"""Duplicate Claim Checker — detects potential duplicate submissions."""



from typing import List, Optional

from pydantic import BaseModel
from models.claim import ClaimRequest


class SimilarClaim(BaseModel):
    claim_id: str
    similarity: float
    matching_fields: List[str]


class DuplicateCheckResult(BaseModel):
    is_duplicate: bool
    confidence: float
    similar_claims: List[SimilarClaim]
    reason: str


DUPLICATE_THRESHOLD = 0.7


class DuplicateClaimChecker:
    """Detects potential duplicate or overlapping claims."""

    def execute(
        self, claim: ClaimRequest, history: Optional[List[ClaimRequest]] = None
    ) -> DuplicateCheckResult:
        """Check for duplicate or overlapping claims."""
        if not history:
            return DuplicateCheckResult(
                is_duplicate=False,
                confidence=0.0,
                similar_claims=[],
                reason="No claim history available for comparison",
            )

        similar_claims = []

        for past in history:
            score = 0.0
            matches = []

            # Date overlap check (weight: 0.4)
            if self._dates_overlap(claim, past):
                score += 0.4
                matches.append("date_overlap")

            # Destination match (weight: 0.3)
            if self._destinations_match(claim, past):
                score += 0.3
                matches.append("destination")

            # Amount similarity within 10% (weight: 0.3)
            if self._amounts_similar(claim, past):
                score += 0.3
                matches.append("amount")

            if score >= DUPLICATE_THRESHOLD:
                similar_claims.append(SimilarClaim(
                    claim_id=past.claim_id,
                    similarity=round(score, 2),
                    matching_fields=matches,
                ))

        is_dup = any(s.similarity >= 0.9 for s in similar_claims)
        max_sim = max((s.similarity for s in similar_claims), default=0.0)

        reason = "No duplicates detected"
        if is_dup:
            reason = f"High-confidence duplicate detected (similarity: {max_sim})"
        elif similar_claims:
            reason = f"Potential duplicate found (similarity: {max_sim})"

        return DuplicateCheckResult(
            is_duplicate=is_dup,
            confidence=max_sim,
            similar_claims=similar_claims,
            reason=reason,
        )

    def _dates_overlap(self, a: ClaimRequest, b: ClaimRequest) -> bool:
        return (
            a.trip.start_date <= b.trip.end_date
            and b.trip.start_date <= a.trip.end_date
        )

    def _destinations_match(self, a: ClaimRequest, b: ClaimRequest) -> bool:
        return a.trip.destination.lower().strip() == b.trip.destination.lower().strip()

    def _amounts_similar(self, a: ClaimRequest, b: ClaimRequest) -> bool:
        if max(a.total_amount, b.total_amount) == 0:
            return True
        diff = abs(a.total_amount - b.total_amount) / max(a.total_amount, b.total_amount)
        return diff < 0.10
