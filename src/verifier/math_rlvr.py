"""
Math RLVR Verifier.

Rule-based verifier for mathematical reasoning tasks.
Extracts numerical answers and compares with ground truth.
"""

import re
from fractions import Fraction

from .base import BaseVerifier
from .registry import register_verifier


@register_verifier("math-rlvr")
class MathRLVRVerifier(BaseVerifier):
    """
    Math verifier using rule-based answer extraction.

    Supports various answer formats:
    - \\boxed{42}, \\boxed{\\frac{1}{2}}
    - Decimal numbers: 3.14159
    - Fractions: 1/2, \\frac{1}{2}
    - Percentages: 50%
    - Scientific notation: 1.5e-3
    """

    # Patterns for extracting answers, ordered by priority
    BOXED_PATTERNS = [
        r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}",  # \boxed{...} with nested braces
        r"\$\\boxed\{([^{}]+)\}\$",  # $\boxed{...}$
    ]

    # Tolerance for floating point comparison
    FLOAT_TOLERANCE = 1e-6

    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify math answer.

        Args:
            response: Model's response string
            metadata: Must contain 'answer' key with correct answer

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        ground_truth = self._get_ground_truth(metadata)
        if ground_truth is None:
            return 0.0

        extracted = self.extract_answer(response)
        if extracted is None:
            return 0.0

        return 1.0 if self._compare_answers(extracted, ground_truth) else 0.0

    def extract_answer(self, response: str) -> str | None:
        """
        Extract answer from response.

        Looks for \\boxed{} first, then falls back to other patterns.
        """
        if not response:
            return None

        # Try boxed patterns first
        for pattern in self.BOXED_PATTERNS:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Take the last match (usually the final answer)
                answer = matches[-1].strip()
                return self._normalize_answer(answer)

        # Fallback: look for a number at the end
        # Pattern matches integers, decimals, fractions, scientific notation
        match = re.search(
            r"(?:answer|result|=)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\d+/\d+)\s*$",
            response,
            re.IGNORECASE | re.MULTILINE,
        )
        if match:
            return match.group(1)

        return None

    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize a mathematical answer.

        Handles LaTeX formatting like \\frac{1}{2}, \\text{}, etc.
        """
        # Remove common LaTeX wrappers
        answer = re.sub(r"\\text\{([^}]*)\}", r"\1", answer)
        answer = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", answer)
        answer = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", answer)

        # Convert LaTeX fractions to simple fractions
        answer = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", answer)

        # Remove dollar signs and spaces
        answer = answer.replace("$", "").strip()

        return answer

    def _compare_answers(self, extracted: str, ground_truth: str) -> bool:
        """
        Compare two answers with numerical tolerance.

        Args:
            extracted: Extracted answer string
            ground_truth: Ground truth answer string

        Returns:
            True if answers match, False otherwise
        """
        # Normalize both answers
        extracted = self._normalize_answer(extracted)
        ground_truth = self._normalize_answer(ground_truth)

        # Direct string comparison (case-insensitive)
        if extracted.lower() == ground_truth.lower():
            return True

        # Try numerical comparison
        try:
            extracted_val = self._parse_number(extracted)
            ground_truth_val = self._parse_number(ground_truth)

            if extracted_val is not None and ground_truth_val is not None:
                # Handle integer comparison
                if isinstance(extracted_val, int) and isinstance(ground_truth_val, int):
                    return extracted_val == ground_truth_val

                # Handle float comparison with tolerance
                return abs(float(extracted_val) - float(ground_truth_val)) < self.FLOAT_TOLERANCE

        except (ValueError, ZeroDivisionError):
            pass

        return False

    def _parse_number(self, s: str) -> int | float | None:
        """
        Parse a string as a number.

        Handles:
        - Integers: 42
        - Floats: 3.14
        - Fractions: 1/2
        - Percentages: 50% (returns 0.5)
        - Scientific notation: 1.5e-3
        """
        s = s.strip()

        # Handle percentage
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100
            except ValueError:
                return None

        # Handle fraction
        if "/" in s:
            try:
                return float(Fraction(s))
            except (ValueError, ZeroDivisionError):
                return None

        # Try parsing as float/int
        try:
            val = float(s)
            # Return int if it's a whole number
            if val.is_integer():
                return int(val)
            return val
        except ValueError:
            return None

    def _get_ground_truth(self, metadata: dict) -> str | None:
        """Extract ground truth answer from metadata."""
        if isinstance(metadata, dict):
            answer = metadata.get("answer") or metadata.get("gold_target")
            if isinstance(answer, str | int | float):
                return str(answer)
        return None
