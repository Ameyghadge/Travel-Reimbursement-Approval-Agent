"""Unit tests for tools — can run without heavy ML dependencies."""

import json
import sys
import importlib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.claim import ClaimRequest
from models.decision import DecisionResult

# Import tools directly to avoid triggering rag/structlog chain via __init__.py
import importlib.util

def _import_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_root = Path(__file__).parent.parent
_rv_mod = _import_module("receipt_validator", _root / "tools" / "receipt_validator.py")
_lc_mod = _import_module("expense_limit_checker", _root / "tools" / "expense_limit_checker.py")
_dc_mod = _import_module("duplicate_checker", _root / "tools" / "duplicate_checker.py")
_am_mod = _import_module("approval_matrix", _root / "tools" / "approval_matrix.py")
_ov_mod = _import_module("output_validator", _root / "tools" / "output_validator.py")

ReceiptValidationTool = _rv_mod.ReceiptValidationTool
ExpenseLimitChecker = _lc_mod.ExpenseLimitChecker
DuplicateClaimChecker = _dc_mod.DuplicateClaimChecker
ApprovalMatrixTool = _am_mod.ApprovalMatrixTool
OutputValidationTool = _ov_mod.OutputValidationTool


def load_claim(filename: str) -> ClaimRequest:
    path = Path(__file__).parent.parent / "data" / "claims" / filename
    with open(path) as f:
        return ClaimRequest(**json.load(f))


def test_receipt_validator():
    rv = ReceiptValidationTool()

    # Claim 001: all receipts present
    claim = load_claim("claim_001.json")
    result = rv.execute(claim.expenses)
    assert result.all_valid is True
    assert result.total_issues == 0
    print("  ✓ Claim 001: all receipts valid")

    # Claim 003: 3 hotel receipts missing (each $320 > $75 threshold)
    claim = load_claim("claim_003.json")
    result = rv.execute(claim.expenses)
    assert result.all_valid is False
    assert len(result.missing_receipts) == 3
    assert all(r.severity == "error" for r in result.missing_receipts)
    print("  ✓ Claim 003: 3 missing receipts flagged (error severity)")

    # Claim 004: flight ($480) and dinner ($42) missing
    claim = load_claim("claim_004.json")
    result = rv.execute(claim.expenses)
    assert result.all_valid is False
    assert len(result.missing_receipts) == 2
    # $480 -> error, $42 -> warning
    severities = {r.amount: r.severity for r in result.missing_receipts}
    assert severities[480.0] == "error"
    assert severities[42.0] == "warning"
    print("  ✓ Claim 004: flight=error, dinner=warning severity")


def test_expense_limit_checker():
    lc = ExpenseLimitChecker()

    # Claim 001: all within limits
    claim = load_claim("claim_001.json")
    result = lc.execute(claim.expenses)
    assert result.all_compliant is True
    assert result.total_overage == 0.0
    print("  ✓ Claim 001: all compliant")

    # Claim 002: hotel $289 > $200 limit
    claim = load_claim("claim_002.json")
    result = lc.execute(claim.expenses)
    assert result.all_compliant is False
    hotel_violations = [v for v in result.violations if v.category == "hotel"]
    assert len(hotel_violations) == 3
    assert hotel_violations[0].overage == 89.0
    print(f"  ✓ Claim 002: hotel over limit ($89/night x3 = ${result.total_overage})")

    # Claim 003: flight $950 > $800
    claim = load_claim("claim_003.json")
    result = lc.execute(claim.expenses)
    assert result.all_compliant is False
    flight_v = [v for v in result.violations if v.category == "flight"]
    assert len(flight_v) == 1
    assert flight_v[0].overage == 150.0
    print("  ✓ Claim 003: flight over limit ($150 overage)")


def test_duplicate_checker():
    dc = DuplicateClaimChecker()

    claim = load_claim("claim_001.json")

    # No history
    result = dc.execute(claim, history=None)
    assert result.is_duplicate is False
    assert "No claim history" in result.reason
    print("  ✓ No history: no duplicate")

    # Same claim as history -> duplicate
    result = dc.execute(claim, history=[claim])
    assert result.is_duplicate is True
    assert result.confidence >= 0.9
    print(f"  ✓ Self-match: duplicate detected (similarity={result.confidence})")

    # Different claim
    claim2 = load_claim("claim_005.json")
    result = dc.execute(claim, history=[claim2])
    assert result.is_duplicate is False
    print("  ✓ Different claim: no duplicate")


def test_approval_matrix():
    am = ApprovalMatrixTool()

    # Under $500 -> manager
    r = am.execute(400.0, ["meals"])
    assert r.required_level == "manager"
    assert r.auto_approve is True
    print("  ✓ $400 -> manager (auto-approve)")

    # $501-$2000 -> director
    r = am.execute(1500.0, ["flight", "hotel"])
    assert r.required_level == "director"
    assert r.auto_approve is True
    print("  ✓ $1500 -> director (auto-approve)")

    # $2001-$5000 -> VP
    r = am.execute(3500.0, ["flight"])
    assert r.required_level == "vp"
    assert r.auto_approve is False
    print("  ✓ $3500 -> vp (manual)")

    # Over $5000 -> CFO
    r = am.execute(6000.0, ["flight"])
    assert r.required_level == "cfo"
    print("  ✓ $6000 -> cfo")

    # International adds travel_desk rule
    r = am.execute(1000.0, ["flight"], is_international=True)
    assert any("travel" in s.lower() for s in r.special_rules)
    print("  ✓ International -> travel_desk special rule")

    # Conference adds pre-approval rule
    r = am.execute(800.0, ["conference"])
    assert any("pre-approval" in s.lower() for s in r.special_rules)
    print("  ✓ Conference -> pre-approval rule")


def test_output_validator():
    ov = OutputValidationTool()

    # Valid JSON
    valid = json.dumps({
        "claim_id": "CLM-001",
        "decision": "approve",
        "confidence_score": 0.9,
        "overall_reasoning": "All good",
        "expense_decisions": [],
        "policy_references": [],
        "total_approved_amount": 100.0,
        "flags": [],
        "requires_additional_info": [],
    })
    is_valid, result, errors = ov.execute(valid)
    assert is_valid is True
    assert result is not None
    assert result.decision.value == "approve"
    print("  ✓ Valid JSON: accepted")

    # Invalid JSON
    is_valid, result, errors = ov.execute("not json at all")
    assert is_valid is False
    assert result is None
    assert len(errors) > 0
    print(f"  ✓ Invalid text: rejected ({errors[0][:40]})")

    # JSON with code fences
    fenced = "```json\n" + valid + "\n```"
    is_valid, result, errors = ov.execute(fenced)
    assert is_valid is True
    print("  ✓ JSON in code fences: extracted and accepted")

    # Missing required field
    incomplete = json.dumps({"claim_id": "X", "decision": "approve"})
    is_valid, result, errors = ov.execute(incomplete)
    assert is_valid is False
    assert any("confidence_score" in e for e in errors)
    print("  ✓ Missing fields: validation errors reported")

    # Invalid confidence score
    bad_conf = json.dumps({
        "claim_id": "X",
        "decision": "approve",
        "confidence_score": 1.5,
        "overall_reasoning": "x",
        "expense_decisions": [],
        "total_approved_amount": 0,
    })
    is_valid, result, errors = ov.execute(bad_conf)
    assert is_valid is False
    print("  ✓ confidence_score > 1.0: validation error")


if __name__ == "__main__":
    print("\n🧪 Running tool tests...\n")

    print("[Receipt Validator]")
    test_receipt_validator()

    print("\n[Expense Limit Checker]")
    test_expense_limit_checker()

    print("\n[Duplicate Checker]")
    test_duplicate_checker()

    print("\n[Approval Matrix]")
    test_approval_matrix()

    print("\n[Output Validator]")
    test_output_validator()

    print("\n✅ ALL TESTS PASSED!\n")
