"""
SFT Formatter.

Formats rollouts for Supervised Fine-Tuning by selecting the best response.
"""

from .base import BaseFormatter
from .registry import register_formatter


@register_formatter("sft")
class SFTFormatter(BaseFormatter):
    """
    SFT formatter - takes the highest scoring response.

    Early stop condition: has at least 1 passed response.
    """

    def format(self, item: dict, rollouts: list[dict]) -> list[dict]:
        """
        Format rollouts into SFT training data.

        Output format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }

        Args:
            item: Original item with "messages"
            rollouts: List of rollouts with "response" and "score"

        Returns:
            List containing single SFT example, or empty list if no passed responses
        """
        # Get passed rollouts
        passed = self._get_passed_rollouts(rollouts)
        if not passed:
            return []

        # Get best response
        best = self._get_best_rollout(passed)
        if best is None:
            return []

        # Build SFT example
        messages = list(item["messages"])  # Copy original messages
        messages.append(
            {
                "role": "assistant",
                "content": best["response"],
            }
        )

        return [
            {
                "messages": messages,
            }
        ]

    def is_satisfied(self, rollouts: list[dict]) -> bool:
        """
        Check if SFT requirements are satisfied.

        Satisfied when: at least 1 passed response.
        """
        passed = self._get_passed_rollouts(rollouts)
        return len(passed) >= 1
