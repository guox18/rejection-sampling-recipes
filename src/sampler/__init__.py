"""
Sampler module.

Provides samplers for generating model responses.
"""

from .base import BaseSampler
from .openai_sampler import OpenAISampler
from .vllm_sampler import VLLMSampler


def get_sampler(cfg, verbose: bool = False) -> BaseSampler:
    """
    Factory function to create a sampler based on configuration.

    Args:
        cfg: Sampler configuration (OmegaConf DictConfig or dict)
        verbose: Enable verbose logging

    Returns:
        BaseSampler instance

    Raises:
        ValueError: If sampler type is unknown
    """
    sampler_type = cfg.type

    if sampler_type == "openai-compatible-api":
        return OpenAISampler(
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=cfg.get("api_key"),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            concurrent_requests=cfg.concurrent_requests,
            timeout=cfg.timeout,
            drop_truncated=cfg.drop_truncated,
            extra_params=cfg.extra_params,
            verbose=verbose,
        )
    elif sampler_type == "vllm-offline":
        return VLLMSampler(
            model_path=cfg.model_path,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            data_parallel_size=cfg.get("data_parallel_size"),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.9),
            drop_truncated=cfg.drop_truncated,
            extra_params=cfg.extra_params,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"Unknown sampler type: '{sampler_type}'. "
            "Available: openai-compatible-api, vllm-offline"
        )


__all__ = [
    "BaseSampler",
    "OpenAISampler",
    "VLLMSampler",
    "get_sampler",
]
