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
from agent.graph import ReimbursementAgent


@st.cache_resource
def get_agent():
    """Load models and build agent (cached across reruns)."""
    with st.spinner("Loading LLM model..."):
        loader = ModelLoader.get_instance(settings.model_name)
        loader.load()
        engine = InferenceEngine(loader)

    with st.spinner("Building RAG index..."):
        emb = EmbeddingModel.get_instance(settings.embedding_model)
        store = FAISSStore(emb)
        retriever = PolicyRetriever(settings.policy_path, store)
        retriever.initialize()

    return ReimbursementAgent(inference_engine=engine, retriever=retriever)


def load_sample_claims():
    claims = {}
    for p in sorted(Path("data/claims").glob("*.json")):
        with open(p) as f:
            d = json.load(f)
        label = f"{d['claim_id']} - {d['employee']['name']} ({d['trip']['destination']}, ${d['total_amount']})"
        claims[label] = d
    return claims


def main():
    st.set_page_config(page_title="Travel Reimbursement Agent", page_icon="🧳", layout="wide")
    st.title("🧳 Travel Reimbursement Approval Agent")
    st.caption(f"Agentic AI: LLM selects tools + generates reasoning | Tools: receipt_validator + expense_limit_checker | RAG: FAISS + MiniLM")

    # Sidebar
    with st.sidebar:
        st.header("How it works")
        st.markdown("""
        1. **LLM plans** which tools to run
        2. **RAG** retrieves relevant policy sections
        3. **Tools** execute (receipt + limit check)
        4. **LLM** generates reasoning explanation
        5. **Decision** derived from tool outputs
        6. If **Manual Review** → you decide below
        """)
        st.divider()
        st.caption(f"Model: {settings.model_name}")
        st.caption(f"Embeddings: {settings.embedding_model}")
        st.caption(f"Max tokens: {settings.max_new_tokens}")
        st.caption(f"Top-K chunks: {settings.top_k_chunks}")

    agent = get_agent()

    # Input
    st.header("📄 Submit Claim")
    tab1, tab2 = st.tabs(["Sample Claims", "Paste JSON"])

    claim_data = None
    with tab1:
        samples = load_sample_claims()
        if samples:
            selected = st.selectbox("Select:", list(samples.keys()))
            claim_data = samples[selected]
            with st.expander("View JSON"):
                st.json(claim_data)

    with tab2:
        raw = st.text_area("Paste claim JSON:", height=200)
        if raw.strip():
            try:
                claim_data = json.loads(raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    # Evaluate
    if st.button("🚀 Evaluate Claim", type="primary", disabled=claim_data is None):
        try:
            claim = ClaimRequest(**claim_data)
        except Exception as e:
            st.error(f"Validation error: {e}")
            return

        with st.spinner("Running agent pipeline..."):
            result = agent.process_claim(claim)

        d = result.decision
        st.session_state["last_result"] = result
        st.session_state["last_claim"] = claim_data

    # Show results
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        d = result.decision

        st.header("📊 Decision")

        # Decision display
        icons = {"approve": "✅", "partially_approve": "⚠️", "reject": "❌", "manual_review": "🔍"}
        col1, col2, col3 = st.columns(3)
        col1.metric("Decision", f"{icons.get(d.decision.value, '?')} {d.decision.value.replace('_', ' ').title()}")
        col2.metric("Confidence", f"{d.confidence_score:.0%}")
        col3.metric("Approved", f"${d.total_approved_amount:,.2f}")

        st.info(f"**Reasoning:** {d.overall_reasoning}")

        # Expense breakdown
        if d.expense_decisions:
            st.subheader("💰 Expense Decisions")
            for ed in d.expense_decisions:
                icon = {"approved": "✅", "reduced": "⚠️", "rejected": "❌"}.get(ed.status, "?")
                st.markdown(f"{icon} **{ed.category}** — ${ed.amount:.2f} → ${ed.approved_amount:.2f} ({ed.status}) — _{ed.reason}_")

        # Flags
        if d.flags:
            st.subheader("🚩 Flags")
            for f in d.flags:
                st.warning(f)

        # ── MANUAL REVIEW SECTION ──
        if d.decision.value == "manual_review":
            st.subheader("🔍 Manual Review Required")
            st.markdown("The agent could not make an automated decision. Please review and decide:")

            if d.requires_additional_info:
                st.markdown("**Missing information:**")
                for info in d.requires_additional_info:
                    st.markdown(f"- {info}")

            col_a, col_r = st.columns(2)
            with col_a:
                if st.button("✅ Manually Approve", type="primary", key="manual_approve"):
                    st.success("✅ Claim APPROVED by reviewer.")
                    st.balloons()
            with col_r:
                if st.button("❌ Manually Reject", type="secondary", key="manual_reject"):
                    st.error("❌ Claim REJECTED by reviewer.")

        # Audit trail
        with st.expander("🔍 Audit Trail"):
            audit = result.audit_trail
            st.text(f"Processing time: {audit.processing_time_ms:.0f}ms")
            st.text(f"Tools: {', '.join(audit.tools_executed)}")
            st.text(f"Retries: {audit.retry_count}")
            st.subheader("Tool Outputs")
            st.json(audit.tool_outputs)
            st.subheader("Policy Chunks Retrieved")
            for i, chunk in enumerate(audit.retrieved_chunks, 1):
                st.text_area(f"Chunk {i}", chunk[:200], height=60, disabled=True)
            st.subheader("LLM Raw Response")
            st.code(audit.llm_raw_response)


if __name__ == "__main__":
    main()
