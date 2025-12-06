"""
Formatter registry.

Provides decorator-based registration for formatters.
"""

from .base import BaseFormatter

# Global registry
FORMATTER_REGISTRY: dict[str, type[BaseFormatter]] = {}


def register_formatter(name: str):
    """
    Decorator to register a formatter class.

    Usage:
        @register_formatter("sft")
        class SFTFormatter(BaseFormatter):
            ...
    """

    def decorator(cls: type[BaseFormatter]):
        if name in FORMATTER_REGISTRY:
            raise ValueError(f"Formatter '{name}' is already registered")
        FORMATTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_formatter(name: str, **kwargs) -> BaseFormatter:
    """
    Get a formatter instance by name.

    Args:
        name: Registered formatter name (e.g., "sft", "dpo")
        **kwargs: Additional arguments to pass to the formatter constructor

    Returns:
        Formatter instance

    Raises:
        ValueError: If formatter name is not registered
    """
    if name not in FORMATTER_REGISTRY:
        available = list(FORMATTER_REGISTRY.keys())
        raise ValueError(f"Unknown formatter: '{name}'. Available: {available}")
    return FORMATTER_REGISTRY[name](**kwargs)


def list_formatters() -> list[str]:
    """List all registered formatter names."""
    return list(FORMATTER_REGISTRY.keys())
