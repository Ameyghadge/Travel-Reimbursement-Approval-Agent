"""Approval Matrix Tool — determines required approval level."""



import json
from pathlib import Path
from typing import List

from pydantic import BaseModel


class ApprovalMatrixResult(BaseModel):
    total_amount: float
    required_level: str
    auto_approve: bool
    special_rules: List[str]
    reason: str


class ApprovalMatrixTool:
    """Determines required approval level based on amount and categories."""

    def __init__(self, matrix_path: str = "data/approval_matrix.json"):
        self._matrix = self._load_matrix(matrix_path)

    def _load_matrix(self, path: str) -> dict:
        filepath = Path(path)
        if not filepath.exists():
            return self._default_matrix()
        with open(filepath) as f:
            return json.load(f)

    def _default_matrix(self) -> dict:
        return {
            "thresholds": [
                {"max_amount": 500, "required_level": "manager", "auto_approve": True},
                {"max_amount": 2000, "required_level": "director", "auto_approve": True},
                {"max_amount": 5000, "required_level": "vp", "auto_approve": False},
                {"max_amount": None, "required_level": "cfo", "auto_approve": False},
            ],
            "special_rules": {
                "international": {"additional_approval": "travel_desk"},
                "conference": {"requires_pre_approval": True},
            },
        }

    def execute(
        self,
        total_amount: float,
        categories: List[str],
        is_international: bool = False,
    ) -> ApprovalMatrixResult:
        """Determine approval authority and requirements."""
        # Find required level
        required_level = "cfo"
        auto_approve = False

        for threshold in self._matrix["thresholds"]:
            max_amt = threshold["max_amount"]
            if max_amt is None or total_amount <= max_amt:
                required_level = threshold["required_level"]
                auto_approve = threshold["auto_approve"]
                break

        # Check special rules
        special = []
        rules = self._matrix.get("special_rules", {})

        if is_international and "international" in rules:
            additional = rules["international"].get("additional_approval", "")
            if additional:
                special.append(f"International travel requires {additional} approval")

        if "conference" in categories and "conference" in rules:
            if rules["conference"].get("requires_pre_approval"):
                special.append("Conference expenses require pre-approval")

        reason = f"Amount ${total_amount:.2f} requires {required_level} approval"
        if auto_approve:
            reason += " (auto-approve eligible)"

        return ApprovalMatrixResult(
            total_amount=total_amount,
            required_level=required_level,
            auto_approve=auto_approve,
            special_rules=special,
            reason=reason,
        )
