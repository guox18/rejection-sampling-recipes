"""Simple text-only SFT recipe config."""

from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass
class TextSFTConfig:
    """Config for the simple text-only recipe."""

    # Sampling
    model: str = "qwen"
    base_url: str | None = None
    api_key: str | None = None
    n_samples: int = 2
    temperature: float = 0.7
    max_tokens: int = 1024
    max_retries: int = 3
    semaphore_per_sampler: int = 10

    # Feasibility (判断 constraints 是否合理)
    feasibility_model: str | None = None
    feasibility_base_url: str | None = None
    feasibility_api_key: str | None = None
    feasibility_temperature: float = 0.6
    feasibility_max_tokens: int = 4096
    feasibility_max_workers: int = 20

    # Verifier
    verifier_max_workers: int = 20

    # Formatting
    pass_threshold: float = 1.0

    # Message normalization
    strip_img_context: bool = True
    text_joiner: str = "\n"
    drop_empty_messages: bool = True

    # Pipeline
    batch_size: int = 8
    concurrency: int = 2
    sampler_concurrency: int | None = None
    verifier_concurrency: int | None = None

    @classmethod
    def from_yaml(cls, path: str) -> "TextSFTConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save config to a YAML file."""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
