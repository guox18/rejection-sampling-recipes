"""
Base sampler interface.

All samplers should inherit from BaseSampler and implement the sample_batch method.
"""

from abc import ABC, abstractmethod


class BaseSampler(ABC):
    """Base class for all samplers."""

    async def initialize(self) -> None:
        """
        Initialize sampler and validate configuration.

        - OpenAI: ping API to check connectivity
        - vLLM: create Ray actors and load models

        Raises:
            RuntimeError: if initialization fails
        """
        ...  # Default: no-op, subclasses can override

    @abstractmethod
    async def sample_batch(
        self,
        items: list[dict],
        n: int,
    ) -> dict[str, list[str]]:
        """
        Sample n responses for each item.

        Args:
            items: List of items, each has "id" and "messages"
                   [{"id": "001", "messages": [{"role": "user", "content": "..."}]}, ...]
            n: Number of responses to generate per item

        Returns:
            Dict mapping item_id to list of responses
            {"001": ["response1", "response2", ...], ...}
        """

    async def shutdown(self) -> None:
        """
        Clean up resources.

        - OpenAI: close client
        - vLLM: kill Ray actors and shutdown Ray
        """
        ...  # Default: no-op, subclasses can override
