"""Output Validation Tool — validates LLM JSON output against schema."""



import json
import re
from typing import List, Optional, Tuple

from pydantic import ValidationError
from models.decision import DecisionResult


class OutputValidationTool:
    """Validates and parses LLM output into a DecisionResult."""

    def execute(self, raw_response: str) -> Tuple[bool, Optional[DecisionResult], List[str]]:
        """Validate LLM response.

        Returns:
            (is_valid, parsed_result_or_None, list_of_errors)
        """
        errors = []

        # Try to extract JSON from the response
        json_str = self._extract_json(raw_response)
        if json_str is None:
            errors.append("No valid JSON object found in LLM response")
            return False, None, errors

        # Parse JSON
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            errors.append(f"JSON parse error: {e}")
            return False, None, errors

        # Validate against Pydantic model
        try:
            result = DecisionResult(**data)
            return True, result, []
        except ValidationError as e:
            for err in e.errors():
                field = " -> ".join(str(loc) for loc in err["loc"])
                errors.append(f"Validation error at '{field}': {err['msg']}")
            return False, None, errors

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling markdown code fences."""
        # Try to find JSON in code fences
        fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Try to find a JSON object directly
        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]

        return None
