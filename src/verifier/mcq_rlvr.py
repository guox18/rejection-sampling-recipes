"""
MCQ RLVR Verifier.

Rule-based verifier for multiple choice questions.
Extracts answer exclusively from \\boxed{} format.
"""

import re

from .base import BaseVerifier
from .registry import register_verifier


@register_verifier("mcq-rlvr")
class MCQRLVRVerifier(BaseVerifier):
    """
    MCQ verifier using \\boxed{} answer extraction.

    Designed for use with R1-style models or models prompted to output
    answers in \\boxed{} format.

    Supported formats:
        - \\boxed{A}
        - \\boxed{\\text{A}}
        - \\boxed{\\textbf{A}}
        - \\boxed{\\mathrm{A}}
        - $\\boxed{A}$
    """

    def __init__(self, **kwargs):
        """Initialize verifier. Accepts **kwargs for compatibility."""
        pass

    # Pattern for extracting content from \boxed{}
    # Uses a more permissive pattern to capture nested braces
    # Matches: \boxed{...} or $\boxed{...}$
    BOXED_PATTERN = re.compile(
        r"\$?\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\$?",
        re.IGNORECASE,
    )

    # Pattern for LaTeX text commands inside boxed
    LATEX_TEXT_PATTERN = re.compile(
        r"\\(?:text|mathrm|mathbf|textbf)\s*\{\s*([A-Za-z])\s*\}",
        re.IGNORECASE,
    )

    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify MCQ answer by extracting from \\boxed{}.

        Args:
            response: Model's response string
            metadata: Must contain 'answer' key with correct answer (e.g., "A", "B")

        Returns:
            1.0 if correct, 0.0 if incorrect or cannot extract
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
        Extract answer from \\boxed{} format.

        Finds all \\boxed{} occurrences and returns the last one
        (typically the final answer in chain-of-thought responses).

        Supported formats:
            - \\boxed{A}           -> "A"
            - \\boxed{\\text{A}}   -> "A"
            - \\boxed{\\textbf{B}} -> "B"
            - \\boxed{\\mathrm{C}} -> "C"
            - $\\boxed{D}$         -> "D"

        Args:
            response: Model's response string

        Returns:
            Extracted answer (single letter), or None if not found
        """
        if not response:
            return None

        # Find all \boxed{} matches
        matches = self.BOXED_PATTERN.findall(response)
        if not matches:
            return None

        # Use the last match (final answer)
        raw_answer = matches[-1].strip()

        # Normalize the answer
        return self._normalize_answer(raw_answer)

    def _normalize_answer(self, raw: str) -> str | None:
        """
        Normalize raw boxed content to a single letter.

        Handles:
            - Direct letters: "A" -> "A"
            - LaTeX commands: "\\text{A}" -> "A"
            - With whitespace: " A " -> "A"

        Args:
            raw: Raw content from inside \\boxed{}

        Returns:
            Single uppercase letter, or None if cannot normalize
        """
        # Try to extract from LaTeX text commands first
        latex_match = self.LATEX_TEXT_PATTERN.search(raw)
        if latex_match:
            return latex_match.group(1).upper()

        # Clean and check if it's a single letter
        cleaned = raw.strip().upper()
        if len(cleaned) == 1 and cleaned.isalpha():
            return cleaned

        # Try to find a standalone letter
        letter_match = re.search(r"^([A-Za-z])$", cleaned)
        if letter_match:
            return letter_match.group(1).upper()

        return None

    def _get_ground_truth(self, metadata: dict) -> str | None:
        """
        Extract ground truth answer from metadata.

        Supports:
            - {"answer": "A"}
            - {"gold_target": "A"}

        Args:
            metadata: Metadata dict containing answer

        Returns:
            Ground truth answer string, or None if not found
        """
        if isinstance(metadata, dict):
            answer = metadata.get("answer") or metadata.get("gold_target")
            if isinstance(answer, str):
                return answer
        return None
