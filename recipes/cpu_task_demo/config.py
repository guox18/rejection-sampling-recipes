"""CPU task demo recipe config."""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class CpuTaskDemoConfig:
    """Config for CPU-intensive demo recipe."""

    # CPU workload
    prime_limit: int = 60000
    rounds: int = 2

    # Pipeline
    batch_size: int = 8
    concurrency: int = 4
    cpu_stage_concurrency: int | None = None
    flush_interval: int = 10

    @classmethod
    def from_yaml(cls, path: str) -> CpuTaskDemoConfig:
        """Load config from YAML."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save config to YAML."""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=False)
