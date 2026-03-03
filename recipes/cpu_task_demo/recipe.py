"""CPU-intensive demo recipe."""

from __future__ import annotations

import logging
import os
import socket
import time

from src.base import BaseRecipe, Stage

from .config import CpuTaskDemoConfig
from .tools import run_prime_workload

logger = logging.getLogger(__name__)


class CpuIntensiveStage(Stage):
    """Run CPU-heavy computation per item."""

    def __init__(self, config: CpuTaskDemoConfig):
        self.config = config

    def process_item(self, item: dict) -> dict:
        prime_limit = int(item.get("prime_limit", self.config.prime_limit))
        rounds = int(item.get("rounds", self.config.rounds))

        if prime_limit < 2:
            raise ValueError(f"prime_limit must be >= 2, got {prime_limit}")
        if rounds < 1:
            raise ValueError(f"rounds must be >= 1, got {rounds}")

        start = time.perf_counter()
        aggregate_prime_count = run_prime_workload(prime_limit, rounds)
        elapsed_seconds = time.perf_counter() - start

        metadata = item.get("metadata") or {}
        metadata.update(
            {
                "worker_host": socket.gethostname(),
                "worker_pid": os.getpid(),
                "prime_limit": prime_limit,
                "rounds": rounds,
                "elapsed_ms": round(elapsed_seconds * 1000, 2),
            }
        )

        logger.debug(
            "[CpuIntensiveStage] Item %s done on %s(pid=%s), elapsed=%.2fms",
            item.get("id", "unknown"),
            metadata["worker_host"],
            metadata["worker_pid"],
            metadata["elapsed_ms"],
        )

        return {
            **item,
            "metadata": metadata,
            "cpu_result": {
                "aggregate_prime_count": aggregate_prime_count,
            },
        }


class CpuTaskDemoRecipe(BaseRecipe):
    """Demo recipe with a single CPU-heavy stage."""

    def stages(self) -> list[Stage]:
        return [CpuIntensiveStage(self.config)]
