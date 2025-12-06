"""
Formatter module.

Provides formatters for converting rollouts into training data.
"""

from .base import BaseFormatter
from .dpo_formatter import DPOFormatter
from .multi_sft_formatter import MultiSFTFormatter
from .registry import FORMATTER_REGISTRY, get_formatter, list_formatters, register_formatter
from .sft_formatter import SFTFormatter

__all__ = [
    "BaseFormatter",
    "DPOFormatter",
    "FORMATTER_REGISTRY",
    "MultiSFTFormatter",
    "SFTFormatter",
    "get_formatter",
    "list_formatters",
    "register_formatter",
]
