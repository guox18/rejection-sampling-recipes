"""
OpenAI API Sampler.

Uses asyncio concurrency for high-throughput sampling via OpenAI-compatible APIs.
"""

import asyncio
import os
import time

import httpx
from openai import AsyncOpenAI
from tqdm import tqdm

from .base import BaseSampler


class OpenAISampler(BaseSampler):
    """OpenAI API sampler using asyncio concurrency."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        concurrent_requests: int = 50,
        timeout: int = 300,
        drop_truncated: bool = True,
        extra_params: dict | None = None,
        verbose: bool = False,
    ):
        """
        Initialize OpenAI sampler.

        Args:
            model: Model name (e.g., "gpt-4", "DeepSeek-R1")
            base_url: API base URL (defaults to OpenAI)
            api_key: API key (uses OPENAI_API_KEY env var if None)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            concurrent_requests: Max concurrent requests
            timeout: Request timeout in seconds
            drop_truncated: Whether to drop truncated responses
            extra_params: Extra parameters for API requests (e.g., reasoning_effort)
            verbose: Enable verbose logging (per-request timing)
        """
        super().__init__()
        self.model = model
        self.verbose = verbose
        self.base_url = base_url or "https://api.openai.com/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.concurrent_requests = concurrent_requests
        self.timeout = timeout
        self.drop_truncated = drop_truncated
        self.extra_params = dict(extra_params) if extra_params else {}

        # Configure httpx client with higher connection limits for true concurrency
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrent_requests,
                max_keepalive_connections=concurrent_requests,
            ),
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

        # Use dummy API key as default for local/self-hosted APIs
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy"

        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=resolved_api_key,
            timeout=timeout,
            http_client=http_client,
        )

    async def initialize(self) -> None:
        """Ping API to validate configuration."""
        print("Initializing OpenAI sampler...")
        print(f"  Base URL: {self.base_url}")
        print(f"  Model: {self.model}")
        if self.extra_params:
            print(f"  Extra params: {self.extra_params}")

        try:
            # Simple health check - minimal request
            await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
                timeout=30,
                **self.extra_params,
            )
            print(f"✓ API connection successful (model: {self.model})")
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI sampler: {e}") from e

    async def sample_batch(
        self,
        items: list[dict],
        n: int,
    ) -> dict[str, list[str]]:
        """
        Async concurrent sampling with rate limiting.

        Args:
            items: List of items with "id" and "messages"
            n: Number of responses per item

        Returns:
            Dict mapping item_id to list of (response, is_truncated) tuples
        """
        semaphore = asyncio.Semaphore(self.concurrent_requests)

        async def sample_one(item: dict, pbar: tqdm) -> tuple[str, list[str]]:
            """Sample one item and return (id, [responses])."""
            async with semaphore:
                try:
                    start_time = time.time()
                    if self.verbose:
                        print(f"  [START] {item['id'][:8]}... at {start_time:.2f}")
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=item["messages"],
                        n=n,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        timeout=self.timeout,
                        **self.extra_params,
                    )

                    results = []
                    truncated_in_batch = 0
                    for choice in response.choices:
                        content = choice.message.content or ""
                        is_truncated = choice.finish_reason == "length"

                        # Skip truncated responses if configured
                        if self.drop_truncated and is_truncated:
                            truncated_in_batch += 1
                            continue

                        results.append(content)

                    # Update statistics
                    self.total_generated += len(response.choices)
                    self.truncated_count += truncated_in_batch

                    if self.verbose:
                        elapsed = time.time() - start_time
                        print(f"  [DONE]  {item['id'][:8]}... took {elapsed:.2f}s")

                    pbar.update(1)
                    return item["id"], results

                except Exception as e:
                    pbar.update(1)
                    print(f"Error sampling {item['id']}: {e}")
                    return item["id"], []

        # Use tqdm progress bar
        with tqdm(total=len(items), desc="    Sampling", unit="req", leave=False) as pbar:
            tasks = [sample_one(item, pbar) for item in items]
            results = await asyncio.gather(*tasks)

        return dict(results)

    async def shutdown(self) -> None:
        """Close HTTP client."""
        await self.client.close()
        print("OpenAI sampler shutdown")
