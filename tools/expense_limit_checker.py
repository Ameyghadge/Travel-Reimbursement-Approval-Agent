"""Expense Limit Checker — validates amounts against category limits."""



import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from models.claim import ExpenseItem, ExpenseCategory


class Violation(BaseModel):
    category: str
    amount: float
    limit: float
    overage: float
    severity: str  # "error", "warning"
    reason: str


class ExpenseLimitResult(BaseModel):
    all_compliant: bool
    violations: List[Violation]
    compliant_count: int
    total_overage: float


class ExpenseLimitChecker:
    """Checks each expense against category-specific policy limits."""

    def __init__(self, limits_path: str = "data/limits.json"):
        self._limits = self._load_limits(limits_path)

    def _load_limits(self, path: str) -> dict:
        filepath = Path(path)
        if not filepath.exists():
            return self._default_limits()
        with open(filepath) as f:
            return json.load(f)

    def _default_limits(self) -> dict:
        return {
            "flight": {"domestic": 800, "international": 2500},
            "hotel": {"domestic_per_night": 200, "international_per_night": 350},
            "meals": {"per_day": 75, "single_meal_max": 50},
            "transport": {"per_day": 100},
            "conference": {"registration_max": 1500},
            "miscellaneous": {"per_trip": 150},
        }

    def execute(
        self, expenses: List[ExpenseItem], is_international: bool = False
    ) -> ExpenseLimitResult:
        """Compare each expense to its category limit."""
        violations = []
        compliant_count = 0

        for exp in expenses:
            limit = self._get_limit(exp.category, is_international)

            if limit is None:
                violations.append(Violation(
                    category=exp.category.value,
                    amount=exp.amount,
                    limit=0,
                    overage=0,
                    severity="warning",
                    reason=f"No limit defined for category '{exp.category.value}'",
                ))
            elif exp.amount > limit:
                violations.append(Violation(
                    category=exp.category.value,
                    amount=exp.amount,
                    limit=limit,
                    overage=round(exp.amount - limit, 2),
                    severity="error",
                    reason=f"Exceeds {exp.category.value} limit of ${limit:.0f}",
                ))
            else:
                compliant_count += 1

        return ExpenseLimitResult(
            all_compliant=(len(violations) == 0),
            violations=violations,
            compliant_count=compliant_count,
            total_overage=round(sum(v.overage for v in violations), 2),
        )

    def _get_limit(self, category: ExpenseCategory, is_international: bool) -> Optional[float]:
        cat = category.value
        limits = self._limits.get(cat)
        if limits is None:
            return None

        if cat == "flight":
            return limits.get("international" if is_international else "domestic")
        elif cat == "hotel":
            key = "international_per_night" if is_international else "domestic_per_night"
            return limits.get(key)
        elif cat == "meals":
            return limits.get("per_day")
        elif cat == "transport":
            return limits.get("per_day")
        elif cat == "conference":
            return limits.get("registration_max")
        elif cat == "miscellaneous":
            return limits.get("per_trip")
        return None
