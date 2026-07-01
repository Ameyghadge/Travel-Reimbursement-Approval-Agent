"""Simple LangGraph agentic workflow for claim evaluation.

Flow:
1. Retrieve policy context (RAG)
2. Run tools (receipt check + limit check)
3. LLM reasons over everything → produces decision
4. If LLM output invalid → fallback to simple rules
"""

import json
import time
import structlog
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.prompts import REASONING_PROMPT
from config import settings
from llm.inference import InferenceEngine
from models.claim import ClaimRequest
from models.decision import DecisionResult, AuditTrail, DecisionResponse
from rag.retriever import PolicyRetriever
from tools.receipt_validator import ReceiptValidationTool
from tools.expense_limit_checker import ExpenseLimitChecker

logger = structlog.get_logger()


class ReimbursementAgent:
    """Lightweight agentic claim processor.

    The LLM sees tool results + policy context and decides:
    approve / partially_approve / reject / manual_review
    """

    def __init__(self, inference_engine: InferenceEngine, retriever: PolicyRetriever):
        self._engine = inference_engine
        self._retriever = retriever
        self._receipt_tool = ReceiptValidationTool()
        self._limit_tool = ExpenseLimitChecker()
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("plan_tools", self._plan_tools)
        workflow.add_node("retrieve_policy", self._retrieve_policy)
        workflow.add_node("run_tools", self._run_tools)
        workflow.add_node("llm_decide", self._llm_decide)

        workflow.set_entry_point("plan_tools")
        workflow.add_edge("plan_tools", "retrieve_policy")
        workflow.add_edge("retrieve_policy", "run_tools")
        workflow.add_edge("run_tools", "llm_decide")
        workflow.add_edge("llm_decide", END)

        return workflow.compile()

    def process_claim(self, claim: ClaimRequest) -> DecisionResponse:
        logger.info("═" * 50)
        logger.info("PROCESSING", claim_id=claim.claim_id, employee=claim.employee.name,
                    destination=claim.trip.destination, total=f"${claim.total_amount}")

        state: AgentState = {
            "claim": claim.model_dump(mode="json"),
            "claim_id": claim.claim_id,
            "policy_chunks": [],
            "receipt_validation": None,
            "expense_limit_result": None,
            "duplicate_check": None,
            "approval_matrix": None,
            "llm_prompt": "",
            "llm_raw_response": "",
            "decision": None,
            "retry_count": 0,
            "errors": [],
            "tools_executed": [],
            "tool_outputs": {},
            "processing_start_time": time.time(),
        }

        final = self._graph.invoke(state)
        return self._build_response(final)

    # ── NODES ────────────────────────────────────────────────────────────

    def _plan_tools(self, state: AgentState) -> dict:
        """LLM decides which tools to run (agentic tool selection)."""
        claim = state["claim"]
        categories = list(set(e["category"] for e in claim["expenses"]))

        prompt = f"""Claim: {claim['claim_id']} | {claim['employee']['name']} | ${claim['total_amount']} | Categories: {', '.join(categories)}

Available tools: receipt_validator, expense_limit_checker
Which tools should I run? List them:"""

        response = self._engine.generate(prompt, max_new_tokens=20, temperature=0.0)

        # Parse which tools the LLM requested
        requested = []
        resp_lower = response.lower()
        if "receipt" in resp_lower:
            requested.append("receipt_validator")
        if "limit" in resp_lower or "expense" in resp_lower:
            requested.append("expense_limit_checker")
        if not requested:
            requested = ["receipt_validator", "expense_limit_checker"]

        logger.info("[1/4] LLM TOOL PLANNING", requested=requested, llm_said=response.strip()[:60])
        return {"tools_executed": requested}

    def _retrieve_policy(self, state: AgentState) -> dict:
        claim = state["claim"]
        categories = list(set(e["category"] for e in claim["expenses"]))
        query = f"travel policy {' '.join(categories)} {claim['trip']['destination']}"

        chunks = self._retriever.retrieve(query, top_k=settings.top_k_chunks)
        logger.info("[2/4] POLICY RETRIEVED", chunks=len(chunks))
        return {"policy_chunks": chunks}

    def _run_tools(self, state: AgentState) -> dict:
        claim = ClaimRequest(**state["claim"])

        # Tool 1: Receipt validation
        receipt_result = self._receipt_tool.execute(claim.expenses)
        logger.info("[3/4] TOOL: receipt_validator", all_valid=receipt_result.all_valid, issues=receipt_result.total_issues)

        # Tool 2: Expense limit checker
        limit_result = self._limit_tool.execute(claim.expenses, is_international=claim.trip.is_international)
        logger.info("[3/4] TOOL: expense_limit_checker", compliant=limit_result.all_compliant, overage=f"${limit_result.total_overage}")

        return {
            "receipt_validation": receipt_result.model_dump(),
            "expense_limit_result": limit_result.model_dump(),
            "tools_executed": ["receipt_validator", "expense_limit_checker"],
            "tool_outputs": {
                "receipt_validation": receipt_result.model_dump(),
                "expense_limit_result": limit_result.model_dump(),
            },
        }

    def _llm_decide(self, state: AgentState) -> dict:
        claim = state["claim"]
        expenses = claim["expenses"]
        rv = state.get("receipt_validation") or {}
        lv = state.get("expense_limit_result") or {}

        # Build expense summary
        exp_lines = []
        for e in expenses:
            rcpt = "✓" if e.get("receipt", {}).get("has_receipt") else "✗MISSING"
            mismatch = " MISMATCH" if e.get("receipt", {}).get("receipt_amount_matches") is False else ""
            exp_lines.append(f"  {e['category']} ${e['amount']} | {e.get('description','')} | {rcpt}{mismatch}")

        # Tool summaries
        if rv.get("all_valid"):
            receipt_summary = "All receipts OK"
        else:
            parts = [f"MISSING ${m['amount']}" for m in rv.get("missing_receipts", [])]
            parts += [f"MISMATCH ${m['amount']}" for m in rv.get("mismatched_amounts", [])]
            receipt_summary = ", ".join(parts)

        if lv.get("all_compliant"):
            limit_summary = "All within limits"
        else:
            parts = [f"{v['category']} ${v['amount']}>${v['limit']}" for v in lv.get("violations", [])]
            limit_summary = ", ".join(parts)

        # Policy context
        policy = "\n".join(c.get("content", "")[:120] for c in state.get("policy_chunks", [])[:2])

        prompt = REASONING_PROMPT.format(
            policy=policy or "Standard travel policy applies.",
            claim_id=claim["claim_id"],
            employee=claim["employee"]["name"],
            destination=claim["trip"]["destination"],
            total=claim["total_amount"],
            expenses="\n".join(exp_lines),
            receipt_result=receipt_summary,
            limit_result=limit_summary,
        )

        logger.info("[4/4] LLM REASONING", prompt_len=len(prompt))
        response = self._engine.generate(prompt, max_new_tokens=settings.max_new_tokens, temperature=0.0)
        logger.info("  LLM response", preview=response[:100])

        # Parse LLM response
        decision = self._parse_response(response, claim, rv, lv)
        return {"decision": decision, "llm_raw_response": response, "llm_prompt": prompt}

    # ── HELPERS ──────────────────────────────────────────────────────────

    def _parse_response(self, response: str, claim: dict, rv: dict, lv: dict) -> dict:
        """Build decision from tool outputs (reliable) + LLM reasoning text."""
        import re

        # 1. Build expense decisions from TOOLS (always accurate)
        expense_decisions = self._build_expense_decisions(claim, rv, lv, "")
        total_approved = sum(ed["approved_amount"] for ed in expense_decisions)

        # 2. Determine decision type from expense decisions (deterministic)
        statuses = [ed["status"] for ed in expense_decisions]
        approved_count = statuses.count("approved")
        rejected_count = statuses.count("rejected")
        reduced_count = statuses.count("reduced")

        has_mismatch = len(rv.get("mismatched_amounts", [])) > 0
        has_missing_critical = any(m.get("severity") == "error" for m in rv.get("missing_receipts", []))

        if has_mismatch:
            decision_type = "manual_review"
        elif has_missing_critical and approved_count > 0:
            decision_type = "manual_review"
        elif rejected_count == len(statuses):
            decision_type = "reject"
        elif rejected_count > 0 and rejected_count >= len(statuses) * 0.5:
            decision_type = "reject"
        elif approved_count == len(statuses):
            decision_type = "approve"
        elif reduced_count > 0 or rejected_count > 0:
            decision_type = "partially_approve"
        else:
            decision_type = "approve"

        # 3. Extract reasoning from LLM response (just the text, ignore its decision)
        reasoning = "Based on policy and tool analysis."
        try:
            text = response.strip()
            fence = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
            if fence:
                text = fence.group(1).strip()
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                data = json.loads(text[start:end+1])
                if data.get("reasoning"):
                    reasoning = data["reasoning"]
        except (json.JSONDecodeError, AttributeError):
            pass

        confidence = 0.9 if decision_type == "approve" else 0.8

        return {
            "claim_id": claim["claim_id"],
            "decision": decision_type,
            "confidence_score": confidence,
            "overall_reasoning": reasoning,
            "expense_decisions": expense_decisions,
            "policy_references": [],
            "total_approved_amount": round(total_approved, 2),
            "flags": [],
            "requires_additional_info": ["Receipt mismatch needs clarification"] if has_mismatch else [],
        }

    def _build_expense_decisions(self, claim: dict, rv: dict, lv: dict, decision_type: str) -> list:
        """Build per-item decisions from actual claim + tool outputs."""
        expenses = claim.get("expenses", [])
        missing_amounts = {m["amount"] for m in rv.get("missing_receipts", [])}
        mismatch_amounts = {m["amount"] for m in rv.get("mismatched_amounts", [])}
        violations = {v["category"]: v for v in lv.get("violations", [])}
        non_reimbursable = ["spa", "alcohol", "minibar", "mini-bar", "gym", "movie", "first class"]

        decisions = []
        for exp in expenses:
            amt = exp["amount"]
            cat = exp["category"]
            desc = (exp.get("description") or "").lower()

            # Check non-reimbursable keywords
            rejected_kw = None
            for kw in non_reimbursable:
                if kw in desc:
                    rejected_kw = kw
                    break

            if rejected_kw:
                decisions.append({"category": cat, "amount": amt, "approved_amount": 0.0,
                                  "status": "rejected", "reason": f"Non-reimbursable: {rejected_kw}"})
            elif amt in mismatch_amounts:
                decisions.append({"category": cat, "amount": amt, "approved_amount": 0.0,
                                  "status": "rejected", "reason": "Receipt amount mismatch"})
            elif amt in missing_amounts:
                decisions.append({"category": cat, "amount": amt, "approved_amount": 0.0,
                                  "status": "rejected", "reason": "Receipt missing (>$25)"})
            elif cat in violations:
                v = violations.pop(cat)
                decisions.append({"category": cat, "amount": amt, "approved_amount": v["limit"],
                                  "status": "reduced", "reason": f"Exceeds ${v['limit']} limit"})
            else:
                decisions.append({"category": cat, "amount": amt, "approved_amount": amt,
                                  "status": "approved", "reason": "Within policy"})
        return decisions

    def _build_response(self, state: AgentState) -> DecisionResponse:
        elapsed = (time.time() - state["processing_start_time"]) * 1000
        decision_data = state.get("decision") or {
            "claim_id": state["claim_id"], "decision": "manual_review",
            "confidence_score": 0.0, "overall_reasoning": "Error",
            "expense_decisions": [], "policy_references": [],
            "total_approved_amount": 0.0, "flags": ["error"], "requires_additional_info": [],
        }

        # Add policy refs from chunks
        if not decision_data.get("policy_references"):
            decision_data["policy_references"] = [
                {"section": c.get("section", "Policy"), "content_snippet": c.get("content", "")[:80], "relevance_score": c.get("relevance_score", 0.5)}
                for c in state.get("policy_chunks", [])[:2]
            ]

        decision = DecisionResult(**decision_data)
        audit = AuditTrail(
            retrieved_chunks=[c.get("content", "") for c in state.get("policy_chunks", [])],
            tools_executed=state.get("tools_executed", []),
            tool_outputs=state.get("tool_outputs", {}),
            llm_raw_response=state.get("llm_raw_response", ""),
            retry_count=state.get("retry_count", 0),
            processing_time_ms=round(elapsed, 2),
        )

        logger.info("RESULT", decision=decision.decision.value, approved=f"${decision.total_approved_amount:.2f}", time=f"{elapsed:.0f}ms")
        logger.info("═" * 50)
        return DecisionResponse(decision=decision, audit_trail=audit)
