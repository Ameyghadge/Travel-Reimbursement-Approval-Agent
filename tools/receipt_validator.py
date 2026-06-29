"""Receipt Validation Tool — checks receipt presence and consistency."""



from typing import List

from pydantic import BaseModel
from models.claim import ExpenseItem


class ReceiptIssue(BaseModel):
    expense_description: str
    amount: float
    issue_type: str  # "missing_receipt", "amount_mismatch"
    severity: str  # "warning", "error"
    reason: str


class ReceiptValidationResult(BaseModel):
    all_valid: bool
    valid_count: int
    missing_receipts: List[ReceiptIssue]
    mismatched_amounts: List[ReceiptIssue]
    total_issues: int


RECEIPT_THRESHOLD = 25.0


class ReceiptValidationTool:
    """Validates receipt data for each expense item."""

    def execute(self, expenses: list[ExpenseItem]) -> ReceiptValidationResult:
        """Check receipt presence and amount consistency."""
        missing = []
        mismatched = []
        valid_count = 0

        for exp in expenses:
            if not exp.receipt.has_receipt:
                if exp.amount > RECEIPT_THRESHOLD:
                    missing.append(ReceiptIssue(
                        expense_description=exp.description or f"{exp.category.value} - {exp.vendor}",
                        amount=exp.amount,
                        issue_type="missing_receipt",
                        severity="error" if exp.amount > 75 else "warning",
                        reason=f"Receipt required for expenses over ${RECEIPT_THRESHOLD:.0f}",
                    ))
                else:
                    valid_count += 1
            elif exp.receipt.receipt_amount_matches is False:
                mismatched.append(ReceiptIssue(
                    expense_description=exp.description or f"{exp.category.value} - {exp.vendor}",
                    amount=exp.amount,
                    issue_type="amount_mismatch",
                    severity="error",
                    reason="Claimed amount does not match receipt amount",
                ))
            else:
                valid_count += 1

        return ReceiptValidationResult(
            all_valid=(len(missing) == 0 and len(mismatched) == 0),
            valid_count=valid_count,
            missing_receipts=missing,
            mismatched_amounts=mismatched,
            total_issues=len(missing) + len(mismatched),
        )
