"""
Base verifier interface.

All verifiers should inherit from BaseVerifier and implement the verify method.
"""

from abc import ABC, abstractmethod


class BaseVerifier(ABC):
    """Base class for all verifiers."""

    @abstractmethod
    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify a response and return a score.

        Args:
            response: Model's response string
            metadata: Metadata dict containing ground truth answer, etc.

        Returns:
            Score (float). For binary verification, 1.0 = correct, 0.0 = incorrect.
        """
        pass

    def extract_answer(self, response: str) -> str | None:
        """
        Extract the answer from a response.

        Args:
            response: Model's response string

        Returns:
            Extracted answer string, or None if extraction failed.
        """
        return None
