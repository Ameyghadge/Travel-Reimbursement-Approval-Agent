"""Pydantic models for agent decision output."""



from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    APPROVE = "approve"
    PARTIALLY_APPROVE = "partially_approve"
    REJECT = "reject"
    MANUAL_REVIEW = "manual_review"


class ExpenseDecision(BaseModel):
    category: str
    amount: float
    approved_amount: float
    status: str  # "approved", "rejected", "reduced"
    reason: str


class PolicyReference(BaseModel):
    section: str
    content_snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class DecisionResult(BaseModel):
    claim_id: str
    decision: DecisionType
    confidence_score: float = Field(ge=0.0, le=1.0)
    overall_reasoning: str
    expense_decisions: List[ExpenseDecision]
    policy_references: List[PolicyReference] = []
    total_approved_amount: float = Field(ge=0.0)
    flags: List[str] = []
    requires_additional_info: List[str] = []


class AuditTrail(BaseModel):
    retrieved_chunks: List[str] = []
    tools_executed: List[str] = []
    tool_outputs: Dict[str, dict] = {}
    llm_raw_response: str = ""
    retry_count: int = 0
    processing_time_ms: float = 0.0


class DecisionResponse(BaseModel):
    decision: DecisionResult
    audit_trail: AuditTrail
