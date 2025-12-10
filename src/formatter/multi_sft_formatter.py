"""
Multi-SFT Formatter.

Formats rollouts for Supervised Fine-Tuning by collecting multiple passed responses.
Useful for diversity sampling where each prompt needs N different correct responses.
"""

from .base import BaseFormatter
from .registry import register_formatter


@register_formatter("multi_sft")
class MultiSFTFormatter(BaseFormatter):
    """
    Multi-SFT formatter - collects N unique passed responses for each prompt.

    Early stop condition: has at least `num_responses` unique passed responses.

    Output format:
    {
        "messages": [...],  # original prompt messages
        "responses": ["response1", "response2", ..., "responseN"]
    }
    """

    def __init__(
        self,
        num_responses: int = 32,
        pass_threshold: float = 1.0,
        fail_threshold: float = 0.0,
    ):
        """
        Initialize Multi-SFT formatter.

        Args:
            num_responses: Number of unique passed responses required
            pass_threshold: Score >= this is considered passed
            fail_threshold: Score <= this is considered failed
        """
        super().__init__(pass_threshold=pass_threshold, fail_threshold=fail_threshold)
        self.num_responses = num_responses

    def _get_unique_passed_rollouts(self, rollouts: list[dict]) -> list[dict]:
        """
        Get unique passed rollouts (deduplicated by response content).

        Returns rollouts in order of appearance, keeping first occurrence.
        """
        passed = self._get_passed_rollouts(rollouts)
        seen_responses: set[str] = set()
        unique_rollouts: list[dict] = []

        for rollout in passed:
            response = rollout["response"]
            if response not in seen_responses:
                seen_responses.add(response)
                unique_rollouts.append(rollout)

        return unique_rollouts

    def format(self, item: dict, rollouts: list[dict]) -> list[dict]:
        """
        Format rollouts into Multi-SFT training data.

        Output format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                ...
            ],
            "responses": ["response1", "response2", ..., "responseN"]
        }

        Args:
            item: Original item with "messages"
            rollouts: List of rollouts with "response" and "score"

        Returns:
            List containing single Multi-SFT example, or empty list if
            not enough unique passed responses
        """
        # Get unique passed rollouts
        unique_passed = self._get_unique_passed_rollouts(rollouts)

        # Must have at least num_responses unique passed responses
        if len(unique_passed) < self.num_responses:
            return []

        # Take the first num_responses (or sort by score if needed)
        # Sort by score descending to get the best ones
        sorted_passed = sorted(unique_passed, key=lambda r: r["score"], reverse=True)
        selected = sorted_passed[: self.num_responses]

        # Extract response strings
        responses = [r["response"] for r in selected]

        # Build Multi-SFT example
        return [
            {
                "messages": list(item["messages"]),
                "responses": responses,
            }
        ]

    def is_satisfied(self, rollouts: list[dict]) -> bool:
        """
        Check if Multi-SFT requirements are satisfied.

        Satisfied when: at least `num_responses` unique passed responses.
        """
        unique_passed = self._get_unique_passed_rollouts(rollouts)
        return len(unique_passed) >= self.num_responses
