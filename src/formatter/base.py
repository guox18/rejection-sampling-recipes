"""
Base formatter interface.

All formatters should inherit from BaseFormatter and implement the format method.
"""

from abc import ABC, abstractmethod


class BaseFormatter(ABC):
    """Base class for all formatters."""

    def __init__(self, pass_threshold: float = 1.0, fail_threshold: float = 0.0):
        """
        Initialize formatter.

        Args:
            pass_threshold: Score >= this is considered passed
            fail_threshold: Score <= this is considered failed
        """
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold

    @abstractmethod
    def format(self, item: dict, rollouts: list[dict]) -> list[dict]:
        """
        Format rollouts into training data.

        Args:
            item: Original item with "id", "messages", "metadata"
            rollouts: List of rollouts, each with "response" and "score"

        Returns:
            List of formatted training examples
        """
        pass

    @abstractmethod
    def is_satisfied(self, rollouts: list[dict]) -> bool:
        """
        Check if the formatter's requirements are satisfied (for early stopping).

        Args:
            rollouts: List of rollouts collected so far

        Returns:
            True if requirements are satisfied, False otherwise
        """
        pass

    def _get_passed_rollouts(self, rollouts: list[dict]) -> list[dict]:
        """Get rollouts with score >= pass_threshold."""
        return [r for r in rollouts if r["score"] >= self.pass_threshold]

    def _get_failed_rollouts(self, rollouts: list[dict]) -> list[dict]:
        """Get rollouts with score <= fail_threshold."""
        return [r for r in rollouts if r["score"] <= self.fail_threshold]

    def _get_best_rollout(self, rollouts: list[dict]) -> dict | None:
        """Get rollout with highest score."""
        if not rollouts:
            return None
        return max(rollouts, key=lambda r: r["score"])

    def _get_worst_rollout(self, rollouts: list[dict]) -> dict | None:
        """Get rollout with lowest score."""
        if not rollouts:
            return None
        return min(rollouts, key=lambda r: r["score"])
