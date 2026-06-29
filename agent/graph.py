"""LangGraph workflow for claim evaluation."""

import json
import time
import structlog
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.prompts import SYSTEM_PROMPT, REASONING_TEMPLATE, CORRECTION_TEMPLATE
from config import settings
from llm.inference import InferenceEngine
from models.claim import ClaimRequest
from models.decision import (
    DecisionResult,
    DecisionType,
    ExpenseDecision,
    AuditTrail,
    DecisionResponse,
)
from tools.policy_lookup import PolicyLookupTool
from tools.receipt_validator import ReceiptValidationTool
from tools.expense_limit_checker import ExpenseLimitChecker
from tools.duplicate_checker import DuplicateClaimChecker
from tools.approval_matrix import ApprovalMatrixTool
from tools.output_validator import OutputValidationTool

logger = structlog.get_logger()


class ReimbursementAgent:
    """LangGraph-based claim processing agent."""

    def __init__(
        self,
        inference_engine: InferenceEngine,
        policy_lookup: PolicyLookupTool,
        receipt_validator: ReceiptValidationTool,
        limit_checker: ExpenseLimitChecker,
        duplicate_checker: DuplicateClaimChecker,
        approval_matrix: ApprovalMatrixTool,
        output_validator: OutputValidationTool,
    ):
        self._engine = inference_engine
        self._policy_lookup = policy_lookup
        self._receipt_validator = receipt_validator
        self._limit_checker = limit_checker
        self._duplicate_checker = duplicate_checker
        self._approval_matrix = approval_matrix
        self._output_validator = output_validator
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("retrieve_policy", self._retrieve_policy)
        workflow.add_node("execute_tools", self._execute_tools)
        workflow.add_node("llm_reasoning", self._llm_reasoning)
        workflow.add_node("validate_output", self._validate_output)
        workflow.add_node("manual_review", self._manual_review)

        workflow.set_entry_point("validate_input")

        workflow.add_conditional_edges(
            "validate_input",
            lambda state: "error" if state["errors"] else "continue",
            {"continue": "retrieve_policy", "error": END},
        )
        workflow.add_edge("retrieve_policy", "execute_tools")
        workflow.add_edge("execute_tools", "llm_reasoning")
        workflow.add_edge("llm_reasoning", "validate_output")
        workflow.add_conditional_edges(
            "validate_output",
            self._should_retry,
            {"accept": END, "retry": "llm_reasoning", "manual": "manual_review"},
        )
        workflow.add_edge("manual_review", END)

        return workflow.compile()

    def process_claim(self, claim: ClaimRequest) -> DecisionResponse:
        """Run the full claim evaluation pipeline."""
        logger.info("processing_claim", claim_id=claim.claim_id)

        initial_state: AgentState = {
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

        final_state = self._graph.invoke(initial_state)
        return self._build_response(final_state)

    # ── Graph Nodes ──────────────────────────────────────────────────────

    def _validate_input(self, state: AgentState) -> dict:
        """Validate the claim input."""
        claim = state["claim"]
        errors = []

        if not claim.get("expenses"):
            errors.append("No expenses in claim")
        if claim.get("total_amount", 0) <= 0:
            errors.append("Total amount must be positive")

        return {"errors": errors}

    def _retrieve_policy(self, state: AgentState) -> dict:
        """Retrieve relevant policy chunks via RAG."""
        claim = state["claim"]
        categories = list(set(e["category"] for e in claim["expenses"]))
        destination = claim.get("trip", {}).get("destination", "")
        is_international = claim.get("trip", {}).get("is_international", False)

        result = self._policy_lookup.execute(
            categories=categories,
            destination=destination,
            is_international=is_international,
        )

        chunks = [c.model_dump() for c in result.chunks]
        return {"policy_chunks": chunks}

    def _execute_tools(self, state: AgentState) -> dict:
        """Execute all validation tools."""
        claim_data = state["claim"]
        claim = ClaimRequest(**claim_data)

        # Receipt validation
        receipt_result = self._receipt_validator.execute(claim.expenses)

        # Expense limits
        limit_result = self._limit_checker.execute(
            claim.expenses, is_international=claim.trip.is_international
        )

        # Duplicate check (no history in prototype)
        dup_result = self._duplicate_checker.execute(claim, history=None)

        # Approval matrix
        categories = list(set(e.category.value for e in claim.expenses))
        approval_result = self._approval_matrix.execute(
            total_amount=claim.total_amount,
            categories=categories,
            is_international=claim.trip.is_international,
        )

        tools_executed = [
            "receipt_validator",
            "expense_limit_checker",
            "duplicate_checker",
            "approval_matrix",
        ]

        tool_outputs = {
            "receipt_validation": receipt_result.model_dump(),
            "expense_limit_result": limit_result.model_dump(),
            "duplicate_check": dup_result.model_dump(),
            "approval_matrix": approval_result.model_dump(),
        }

        return {
            "receipt_validation": receipt_result.model_dump(),
            "expense_limit_result": limit_result.model_dump(),
            "duplicate_check": dup_result.model_dump(),
            "approval_matrix": approval_result.model_dump(),
            "tools_executed": tools_executed,
            "tool_outputs": tool_outputs,
        }

    def _llm_reasoning(self, state: AgentState) -> dict:
        """Build compact prompt and call the LLM."""
        claim = state["claim"]
        expenses = claim.get("expenses", [])

        # Build compact expense summary
        exp_lines = []
        for e in expenses:
            exp_lines.append(f"- {e['category']} ${e['amount']} ({e.get('description','')})")
        expenses_summary = "\n".join(exp_lines)

        # Compact tool summaries
        rv = state.get("receipt_validation") or {}
        if rv.get("all_valid"):
            receipt_summary = "OK"
        else:
            issue_count = rv.get("total_issues", 0)
            receipt_summary = f"{issue_count} issues"
        if rv.get("missing_receipts"):
            missing = [f"${m['amount']}" for m in rv["missing_receipts"]]
            receipt_summary += f" (missing: {', '.join(missing)})"

        lv = state.get("expense_limit_result") or {}
        if lv.get("all_compliant"):
            limit_summary = "all within limits"
        else:
            viols = [f"{v['category']} ${v['amount']}>${v['limit']}" for v in lv.get("violations", [])]
            limit_summary = f"violations: {'; '.join(viols)}"

        dv = state.get("duplicate_check") or {}
        duplicate_summary = "no duplicates" if not dv.get("is_duplicate") else "DUPLICATE DETECTED"

        av = state.get("approval_matrix") or {}
        approval_summary = f"{av.get('required_level','unknown')} level, auto={av.get('auto_approve','?')}"

        if state["retry_count"] > 0 and state["llm_raw_response"]:
            prompt = CORRECTION_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                errors="; ".join(state["errors"][:2]),
                claim_id=claim.get("claim_id", ""),
                total_amount=claim.get("total_amount", 0),
                receipt_summary=receipt_summary,
                limit_summary=limit_summary,
                duplicate_summary=duplicate_summary,
                approval_summary=approval_summary,
            )
        else:
            # Compact policy context — use shorter snippets
            policy_text = " | ".join(
                f"[{c.get('section', '')}] {c.get('content', '')[:100]}"
                for c in state["policy_chunks"][:2]
            )

            prompt = REASONING_TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                policy_chunks=policy_text or "No policy context.",
                claim_id=claim.get("claim_id", ""),
                employee=claim.get("employee", {}).get("name", ""),
                destination=claim.get("trip", {}).get("destination", ""),
                total_amount=claim.get("total_amount", 0),
                expenses_summary=expenses_summary,
                receipt_summary=receipt_summary,
                limit_summary=limit_summary,
                duplicate_summary=duplicate_summary,
                approval_summary=approval_summary,
            )

        logger.info("calling_llm", retry=state["retry_count"], prompt_len=len(prompt))
        response = self._engine.generate(
            prompt,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
        )

        return {"llm_prompt": prompt, "llm_raw_response": response}

    def _validate_output(self, state: AgentState) -> dict:
        """Validate the LLM output."""
        is_valid, result, errors = self._output_validator.execute(state["llm_raw_response"])

        if is_valid and result:
            return {"decision": result.model_dump(), "errors": []}
        else:
            return {"errors": errors, "retry_count": state["retry_count"] + 1}

    def _manual_review(self, state: AgentState) -> dict:
        """Fallback to manual review."""
        claim = state["claim"]
        expenses = claim.get("expenses", [])

        # Build a manual review decision
        expense_decisions = [
            {
                "category": e.get("category", "unknown"),
                "amount": e.get("amount", 0),
                "approved_amount": 0.0,
                "status": "rejected",
                "reason": "Routed to manual review - automated decision not possible",
            }
            for e in expenses
        ]

        decision = {
            "claim_id": state["claim_id"],
            "decision": "manual_review",
            "confidence_score": 0.0,
            "overall_reasoning": f"Automated decision failed after {state['retry_count']} retries. Errors: {'; '.join(state['errors'][:3])}",
            "expense_decisions": expense_decisions,
            "policy_references": [],
            "total_approved_amount": 0.0,
            "flags": ["llm_output_validation_failed"],
            "requires_additional_info": ["Manual review by finance team required"],
        }

        return {"decision": decision}

    # ── Helpers ──────────────────────────────────────────────────────────

    def _should_retry(self, state: AgentState) -> str:
        """Determine next step after output validation."""
        if state["decision"] is not None:
            return "accept"
        elif state["retry_count"] < settings.max_retries:
            return "retry"
        else:
            return "manual"

    def _summarize(self, data) -> str:
        """Create a brief summary of tool output for correction prompt."""
        if data is None:
            return "N/A"
        return json.dumps(data, default=str)[:200]

    def _build_response(self, state: AgentState) -> DecisionResponse:
        """Build the final DecisionResponse from state."""
        elapsed = (time.time() - state["processing_start_time"]) * 1000

        decision_data = state["decision"]
        if decision_data is None:
            # Should not happen, but safety fallback
            decision_data = {
                "claim_id": state["claim_id"],
                "decision": "manual_review",
                "confidence_score": 0.0,
                "overall_reasoning": "No decision produced",
                "expense_decisions": [],
                "policy_references": [],
                "total_approved_amount": 0.0,
                "flags": ["no_decision_produced"],
                "requires_additional_info": [],
            }

        decision = DecisionResult(**decision_data)

        audit = AuditTrail(
            retrieved_chunks=[c.get("content", "") for c in state["policy_chunks"]],
            tools_executed=state["tools_executed"],
            tool_outputs=state["tool_outputs"],
            llm_raw_response=state["llm_raw_response"],
            retry_count=state["retry_count"],
            processing_time_ms=round(elapsed, 2),
        )

        logger.info(
            "claim_processed",
            claim_id=state["claim_id"],
            decision=decision.decision.value,
            confidence=decision.confidence_score,
            time_ms=round(elapsed, 2),
        )

        return DecisionResponse(decision=decision, audit_trail=audit)
