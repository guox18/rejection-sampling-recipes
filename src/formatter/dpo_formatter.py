"""
DPO Formatter.

Formats rollouts for Direct Preference Optimization by selecting best and worst responses.
"""

from .base import BaseFormatter
from .registry import register_formatter


@register_formatter("dpo")
class DPOFormatter(BaseFormatter):
    """
    DPO formatter - takes the highest and lowest scoring responses.

    Early stop condition: has at least 1 passed and 1 failed response.
    """

    def format(self, item: dict, rollouts: list[dict]) -> list[dict]:
        """
        Format rollouts into DPO training data.

        Output format:
        {
            "prompt": [{"role": "user", "content": "..."}],
            "chosen": [{"role": "assistant", "content": "..."}],
            "rejected": [{"role": "assistant", "content": "..."}]
        }

        Args:
            item: Original item with "messages"
            rollouts: List of rollouts with "response" and "score"

        Returns:
            List containing single DPO example, or empty list if requirements not met
        """
        # Get passed and failed rollouts
        passed = self._get_passed_rollouts(rollouts)
        failed = self._get_failed_rollouts(rollouts)

        if not passed or not failed:
            return []

        # Get best passed and worst failed
        best = self._get_best_rollout(passed)
        worst = self._get_worst_rollout(failed)

        if best is None or worst is None:
            return []

        # Build DPO example
        return [
            {
                "prompt": list(item["messages"]),
                "chosen": [{"role": "assistant", "content": best["response"]}],
                "rejected": [{"role": "assistant", "content": worst["response"]}],
            }
        ]

    def is_satisfied(self, rollouts: list[dict]) -> bool:
        """
        Check if DPO requirements are satisfied.

        Satisfied when: at least 1 passed AND at least 1 failed response.
        """
        passed = self._get_passed_rollouts(rollouts)
        failed = self._get_failed_rollouts(rollouts)
        return len(passed) >= 1 and len(failed) >= 1
