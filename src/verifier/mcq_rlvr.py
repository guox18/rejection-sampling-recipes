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
    MCQ verifier using multiple answer extraction strategies.

    Designed for use with various models that output answers in different formats.
    Supports both LaTeX-style \\boxed{} format and plain text formats.

    Supported formats:
        - \\boxed{A}
        - \\boxed{\\text{A}}
        - \\boxed{\\textbf{A}}
        - \\boxed{\\mathrm{A}}
        - \\boxed{A: option text}
        - $\\boxed{A}$
        - **Answer: A** (deepseek-r1 style)
        - **Final Answer: B** (deepseek-r1 style)
        - "The answer is C" (plain text)

    Note: This is a rule-based verifier that runs quickly.
          Uses default sequential batch verification (no concurrency needed).
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

    # Additional patterns for deepseek-r1 style answers
    # Matches: **Answer: A**, **Final Answer: B**, etc.
    BOLD_ANSWER_PATTERN = re.compile(
        r"\*\*(?:Final\s+)?Answer:\s*([A-Z])\*\*",
        re.IGNORECASE,
    )

    # Matches: "The answer is A", "the correct answer is B", etc.
    PLAIN_ANSWER_PATTERN = re.compile(
        r"(?:the\s+)?(?:correct\s+)?answer\s+is\s+(?:option\s+)?([A-Z])",
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
        Extract answer from various formats.

        Tries multiple extraction strategies in order of preference:
        1. \\boxed{} format (LaTeX style)
        2. **Answer: X** format (deepseek-r1 style)
        3. Plain "the answer is X" format

        Supported formats:
            - \\boxed{A}           -> "A"
            - \\boxed{\\text{A}}   -> "A"
            - \\boxed{\\textbf{B}} -> "B"
            - \\boxed{\\mathrm{C}} -> "C"
            - $\\boxed{D}$         -> "D"
            - **Answer: E**        -> "E"
            - **Final Answer: F**  -> "F"
            - "The answer is G"    -> "G"

        Args:
            response: Model's response string

        Returns:
            Extracted answer (single letter), or None if not found
        """
        if not response:
            return None

        # Strategy 1: Try \boxed{} format first (most reliable)
        matches = self.BOXED_PATTERN.findall(response)
        if matches:
            # Use the last match (final answer)
            raw_answer = matches[-1].strip()
            normalized = self._normalize_answer(raw_answer)
            if normalized:
                return normalized

        # Strategy 2: Try **Answer: X** format (deepseek-r1 style)
        matches = self.BOLD_ANSWER_PATTERN.findall(response)
        if matches:
            # Use the last match (final answer)
            answer = matches[-1].strip().upper()
            if len(answer) == 1 and answer.isalpha():
                return answer

        # Strategy 3: Try plain "the answer is X" format
        matches = self.PLAIN_ANSWER_PATTERN.findall(response)
        if matches:
            # Use the last match (final answer)
            answer = matches[-1].strip().upper()
            if len(answer) == 1 and answer.isalpha():
                return answer

        return None

    def _normalize_answer(self, raw: str) -> str | None:
        """
        Normalize raw boxed content to a single letter.

        Handles:
            - Direct letters: "A" -> "A"
            - LaTeX commands: "\\text{A}" -> "A"
            - With whitespace: " A " -> "A"
            - Option with text: "A: option text" -> "A"

        Args:
            raw: Raw content from inside \\boxed{}

        Returns:
            Single uppercase letter, or None if cannot normalize
        """
        # Try to extract from LaTeX text commands first
        latex_match = self.LATEX_TEXT_PATTERN.search(raw)
        if latex_match:
            return latex_match.group(1).upper()

        # Try to match "A: option text" or "A：选项文本" format
        option_with_text = re.match(r"^([A-Za-z])\s*[:：]", raw.strip())
        if option_with_text:
            return option_with_text.group(1).upper()

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
