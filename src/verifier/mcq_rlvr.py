"""
MCQ RLVR Verifier.

Rule-based verifier for multiple choice questions.
Extracts answer from \\boxed{} format (common in R1-style models).
"""

import re

from .base import BaseVerifier
from .registry import register_verifier


@register_verifier("mcq-rlvr")
class MCQRLVRVerifier(BaseVerifier):
    """
    MCQ verifier using rule-based answer extraction.

    Works best with R1-style models that output answers in \\boxed{} format.
    """

    def __init__(self, **kwargs):
        """Initialize verifier. Accepts **kwargs for compatibility."""
        # Rule-based verifier doesn't need any config
        pass

    # Patterns for extracting answers, ordered by priority
    EXTRACT_PATTERNS = [
        r"\\boxed\{([^}]+)\}",  # \boxed{A} or \boxed{\text{A}}
        r"\$\\boxed\{([^}]+)\}\$",  # $\boxed{A}$
        r"\\text\{Answer:\s*([A-Z])\}",  # \text{Answer: A}
        r"\*\*Answer:\s*([A-Z])",  # **Answer: E** (markdown bold)
        r"Answer:\s*([A-Z])\b",  # Answer: E (plain text)
        r"answer:\s*([A-Z])\b",  # answer: E (lowercase)
        r"答案[是为]?\s*[:：]?\s*([A-Z])",  # Chinese: 答案是 A
        r"选项?\s*([A-Z])\s*是?正确",  # 选项 A 正确
    ]

    # Pattern for LaTeX text commands
    LATEX_TEXT_PATTERN = re.compile(
        r"\\(?:text|mathrm|mathbf|textbf)\s*\{\s*([A-Z])\s*\}", re.IGNORECASE
    )

    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify MCQ answer.

        Args:
            response: Model's response string
            metadata: Must contain 'answer' key with correct answer (e.g., "A", "B")

        Returns:
            1.0 if correct, 0.0 if incorrect
        """
        ground_truth = self._get_ground_truth(metadata)
        if ground_truth is None:
            return 0.0

        extracted = self.extract_answer(response)
        if extracted is None:
            return 0.0

        # Case-insensitive comparison
        return 1.0 if extracted.upper() == ground_truth.upper() else 0.0

    def extract_answer(self, response: str) -> str | None:
        """
        Extract answer from response.

        Supports multiple formats:
        - \\boxed{A}, \\boxed{\\text{A}}
        - 答案是 A
        - Fallback: last standalone letter in response
        """
        if not response:
            return None

        # Try each pattern in order
        for pattern in self.EXTRACT_PATTERNS:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Handle LaTeX text commands
                normalized = self._normalize_latex_answer(answer)
                if normalized:
                    return normalized

        # Fallback: look for standalone letter at the end
        # e.g., "Therefore, the answer is A."
        match = re.search(r"\b([A-Z])\s*[.。]?\s*$", response.strip())
        if match:
            return match.group(1)

        return None

    def _normalize_latex_answer(self, answer: str) -> str | None:
        """
        Normalize a LaTeX-formatted answer to a single letter.

        Args:
            answer: Raw extracted answer (may contain LaTeX)

        Returns:
            Single uppercase letter, or None if cannot normalize
        """
        # Check for LaTeX text commands: \text{A}, \mathrm{A}, etc.
        latex_match = self.LATEX_TEXT_PATTERN.search(answer)
        if latex_match:
            return latex_match.group(1).upper()

        # Clean and check if it's a single letter
        cleaned = answer.upper().strip()
        if len(cleaned) == 1 and cleaned.isalpha():
            return cleaned

        # Try to extract a standalone letter
        letter_match = re.search(r"\b([A-Z])\b", cleaned)
        if letter_match:
            return letter_match.group(1)

        return None

    def _get_ground_truth(self, metadata: dict) -> str | None:
        """
        Extract ground truth answer from metadata.

        Supports multiple formats:
        - {"answer": "A"}
        - {"gold_target": "A"}
        """
        if isinstance(metadata, dict):
            answer = metadata.get("answer") or metadata.get("gold_target")
            if isinstance(answer, str):
                return answer
        return None
