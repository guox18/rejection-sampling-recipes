"""
Rejection Sampling Recipes.

A framework for reproducible LLM data synthesis using rejection sampling.
"""

from .formatter import (
    BaseFormatter,
    DPOFormatter,
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
    VLLMSampler,
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
