"""
Rejection Sampling Recipes.

A framework for reproducible LLM data synthesis using rejection sampling.
"""

from .formatter import (
    BaseFormatter,
    DPOFormatter,
    MultiSFTFormatter,
    SFTFormatter,
    get_formatter,
    list_formatters,
    register_formatter,
)
from .pipeline import Pipeline, run_pipeline
from .preprocessor import DataPreprocessor
from .sampler import (
    BaseSampler,
    OpenAISampler,
    get_sampler,
)
from .state import StateManager
from .verifier import (
    BaseVerifier,
    MCQLLMJudgeVerifier,
    MCQRLVRVerifier,
    get_verifier,
    list_verifiers,
    register_verifier,
)


# Lazy import for VLLMSampler to avoid requiring vllm/ray for basic usage
def __getattr__(name):
    if name == "VLLMSampler":
        from .sampler.vllm_sampler import VLLMSampler

        return VLLMSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Sampler
    "BaseSampler",
    "OpenAISampler",
    "VLLMSampler",
    "get_sampler",
    # Verifier
    "BaseVerifier",
    "MCQLLMJudgeVerifier",
    "MCQRLVRVerifier",
    "get_verifier",
    "list_verifiers",
    "register_verifier",
    # Formatter
    "BaseFormatter",
    "DPOFormatter",
    "MultiSFTFormatter",
    "SFTFormatter",
    "get_formatter",
    "list_formatters",
    "register_formatter",
    # Pipeline
    "DataPreprocessor",
    "Pipeline",
    "StateManager",
    "run_pipeline",
]
