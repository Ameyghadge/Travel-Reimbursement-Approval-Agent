"""Policy Lookup Tool — retrieves relevant policy sections via RAG."""



from typing import List

from pydantic import BaseModel
from rag.retriever import PolicyRetriever


class PolicyChunk(BaseModel):
    content: str
    section: str
    relevance_score: float


class PolicyLookupResult(BaseModel):
    chunks: List[PolicyChunk]
    query_used: str


class PolicyLookupTool:
    """Retrieves relevant policy sections using RAG pipeline."""

    def __init__(self, retriever: PolicyRetriever):
        self._retriever = retriever

    def execute(self, categories: List[str], destination: str = "", is_international: bool = False) -> PolicyLookupResult:
        """Look up policy rules for given expense categories."""
        # Build a composite query
        parts = []
        if destination:
            parts.append(f"travel to {destination}")
        parts.append(f"expense categories: {', '.join(categories)}")
        if is_international:
            parts.append("international travel policy")
        else:
            parts.append("domestic travel policy")

        query = " ".join(parts)

        raw_results = self._retriever.retrieve(query, top_k=3)

        chunks = [
            PolicyChunk(
                content=r["content"],
                section=r["section"],
                relevance_score=r["relevance_score"],
            )
            for r in raw_results
        ]

        return PolicyLookupResult(chunks=chunks, query_used=query)
