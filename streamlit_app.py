"""Streamlit demo UI for the Travel Reimbursement Approval Agent."""

import json
import time
from pathlib import Path

import streamlit as st

from config import settings
from models.claim import ClaimRequest
from llm.loader import ModelLoader
from llm.inference import InferenceEngine
from rag.embeddings import EmbeddingModel
from rag.vector_store import FAISSStore
from rag.retriever import PolicyRetriever
from tools import (
    PolicyLookupTool,
    ReceiptValidationTool,
    ExpenseLimitChecker,
    DuplicateClaimChecker,
    ApprovalMatrixTool,
    OutputValidationTool,
)
from agent.graph import ReimbursementAgent


@st.cache_resource
def get_agent():
    """Initialize and cache the agent (loaded once)."""
    with st.spinner("Loading Qwen2.5-3B-Instruct model... (this takes a minute on first run)"):
        loader = ModelLoader.get_instance(settings.model_name)
        loader.load()
        engine = InferenceEngine(loader)

    with st.spinner("Initializing RAG pipeline..."):
        embedding_model = EmbeddingModel.get_instance(settings.embedding_model)
        store = FAISSStore(embedding_model)
        retriever = PolicyRetriever(settings.policy_path, store)
        retriever.initialize()

    policy_lookup = PolicyLookupTool(retriever)
    receipt_validator = ReceiptValidationTool()
    limit_checker = ExpenseLimitChecker()
    duplicate_checker = DuplicateClaimChecker()
    approval_matrix = ApprovalMatrixTool()
    output_validator = OutputValidationTool()

    return ReimbursementAgent(
        inference_engine=engine,
        policy_lookup=policy_lookup,
        receipt_validator=receipt_validator,
        limit_checker=limit_checker,
        duplicate_checker=duplicate_checker,
        approval_matrix=approval_matrix,
        output_validator=output_validator,
    )


def load_sample_claims() -> dict[str, dict]:
    """Load available sample claims."""
    claims = {}
    claims_dir = Path("data/claims")
    if claims_dir.exists():
        for path in sorted(claims_dir.glob("*.json")):
            with open(path) as f:
                data = json.load(f)
            label = f"{data['claim_id']} - {data['employee']['name']} ({data['trip']['destination']})"
            claims[label] = data
    return claims


def main():
    st.set_page_config(
        page_title="Travel Reimbursement Agent",
        page_icon="🧳",
        layout="wide",
    )

    st.title("🧳 Travel Reimbursement Approval Agent")
    st.caption("AI-powered claim evaluation using Qwen2.5-3B-Instruct + LangGraph + RAG")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.text(f"Model: {settings.model_name}")
        st.text(f"Embeddings: {settings.embedding_model}")
        st.text(f"Max Retries: {settings.max_retries}")
        st.text(f"Top-K Chunks: {settings.top_k_chunks}")
        st.divider()
        st.header("📋 How it works")
        st.markdown("""
        1. Submit a claim (JSON)
        2. Agent retrieves relevant policy sections (RAG)
        3. Tools validate receipts, limits, duplicates
        4. Qwen2.5 reasons over context + tool outputs
        5. Structured decision is returned
        """)

    # Initialize agent
    agent = get_agent()

    # Input section
    st.header("📄 Submit Claim")

    tab1, tab2 = st.tabs(["📂 Sample Claims", "📝 Paste JSON"])

    claim_data = None

    with tab1:
        samples = load_sample_claims()
        if samples:
            selected = st.selectbox("Select a sample claim:", list(samples.keys()))
            if selected:
                claim_data = samples[selected]
                with st.expander("View claim JSON"):
                    st.json(claim_data)
        else:
            st.warning("No sample claims found in data/claims/")

    with tab2:
        json_input = st.text_area("Paste claim JSON:", height=300, placeholder='{"claim_id": "...", ...}')
        if json_input.strip():
            try:
                claim_data = json.loads(json_input)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Process button
    if st.button("🚀 Evaluate Claim", type="primary", disabled=claim_data is None):
        if claim_data:
            try:
                claim = ClaimRequest(**claim_data)
            except Exception as e:
                st.error(f"Validation error: {e}")
                return

            with st.spinner("Processing claim through agent pipeline..."):
                start = time.time()
                response = agent.process_claim(claim)
                elapsed = time.time() - start

            # Results
            st.header("📊 Decision")
            d = response.decision

            # Decision badge
            colors = {
                "approve": "green",
                "partially_approve": "orange",
                "reject": "red",
                "manual_review": "blue",
            }
            icons = {
                "approve": "✅",
                "partially_approve": "⚠️",
                "reject": "❌",
                "manual_review": "🔍",
            }
            color = colors.get(d.decision.value, "gray")
            icon = icons.get(d.decision.value, "❓")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Decision", f"{icon} {d.decision.value.replace('_', ' ').title()}")
            with col2:
                st.metric("Confidence", f"{d.confidence_score:.0%}")
            with col3:
                st.metric("Approved Amount", f"${d.total_approved_amount:,.2f}")

            st.info(f"**Reasoning:** {d.overall_reasoning}")

            # Expense breakdown
            if d.expense_decisions:
                st.subheader("💰 Expense Decisions")
                rows = []
                for ed in d.expense_decisions:
                    status_icon = {"approved": "✓", "reduced": "~", "rejected": "✗"}.get(ed.status, "?")
                    rows.append({
                        "Category": ed.category,
                        "Claimed": f"${ed.amount:.2f}",
                        "Approved": f"${ed.approved_amount:.2f}",
                        "Status": f"{status_icon} {ed.status}",
                        "Reason": ed.reason,
                    })
                st.table(rows)

            # Policy references
            if d.policy_references:
                st.subheader("📖 Policy References")
                for ref in d.policy_references:
                    st.markdown(f"- **{ref.section}** (relevance: {ref.relevance_score:.2f}): _{ref.content_snippet[:100]}_")

            # Flags
            if d.flags:
                st.subheader("⚠️ Flags")
                for flag in d.flags:
                    st.warning(flag)

            # Audit trail (expandable)
            with st.expander("🔍 Audit Trail"):
                audit = response.audit_trail
                st.text(f"Processing time: {audit.processing_time_ms:.0f}ms")
                st.text(f"Retries: {audit.retry_count}")
                st.text(f"Tools executed: {', '.join(audit.tools_executed)}")

                st.subheader("Retrieved Policy Chunks")
                for i, chunk in enumerate(audit.retrieved_chunks, 1):
                    st.text_area(f"Chunk {i}", chunk, height=80, disabled=True)

                st.subheader("Tool Outputs")
                st.json(audit.tool_outputs)

                st.subheader("LLM Raw Response")
                st.code(audit.llm_raw_response, language="json")

            # Full JSON response
            with st.expander("📋 Full JSON Response"):
                st.json(response.model_dump(mode="json"))


if __name__ == "__main__":
    main()
