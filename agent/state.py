"""Agent state definition for the LangGraph workflow."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict):
    """Mutable state passed through the LangGraph workflow nodes."""

    # Input
    claim: dict  # Serialized ClaimRequest
    claim_id: str

    # RAG
    policy_chunks: List[dict]

    # Tool results
    receipt_validation: Optional[dict]
    expense_limit_result: Optional[dict]
    duplicate_check: Optional[dict]
    approval_matrix: Optional[dict]

    # LLM
    llm_prompt: str  # kept internally for retry logic, not in output
    llm_raw_response: str

    # Output
    decision: Optional[dict]

    # Control flow
    retry_count: int
    errors: List[str]

    # Audit
    tools_executed: List[str]
    tool_outputs: Dict[str, Any]
    processing_start_time: float
