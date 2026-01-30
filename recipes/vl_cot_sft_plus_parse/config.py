"""SFT recipe configuration."""

from dataclasses import dataclass

import yaml


@dataclass
class SFTConfig:
    """Default SFT recipe config. YAML values override these defaults."""

    # LLM parser config
    parse_model: str = "qwen"
    parse_base_url: str = None
    parse_api_key: str = "dummy"
    parse_temperature: float = 0.0
    parse_max_tokens: int = 8096
    parse_max_workers: int = 256

    # Sampling config
    model: str = "qwen"
    base_url: str = None
    api_key: str = None
    n_samples: int = 16
    temperature: float = 0.7
    max_tokens: int = 4096
    semaphore_per_sampler: int = 10

    # Data path config
    abs_image_path_field: str = (
        "abs_path"  # Absolute image path field (supports nesting, e.g. meta_info.abs_image_path)
    )
    max_image_size_mb: float = 100.0  # Max image size (MB); larger images are marked failed

    # LLM judge config
    judge_model: str = None  # Defaults to model
    judge_base_url: str = None  # Defaults to base_url
    judge_api_key: str = None  # Defaults to api_key
    judge_temperature: float = 0.0
    judge_max_tokens: int = 10
    verifier_max_workers: int = 20  # VerifierStage thread pool size (per-item in batch)

    # Formatting config
    pass_threshold: float = 1.0

    # Retry config
    max_retries: int = 3

    # Pipeline config
    batch_size: int = 4  # Items per batch
    concurrency: int = 4  # Default concurrency (stage actor count)
    sampler_concurrency: int = None  # SamplerStage concurrency (defaults to concurrency)
    verifier_concurrency: int = None  # VerifierStage concurrency (defaults to concurrency)

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to a YAML file."""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
