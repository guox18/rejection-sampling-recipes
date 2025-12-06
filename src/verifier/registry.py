"""
Verifier registry.

Provides decorator-based registration for verifiers.
"""

from .base import BaseVerifier

# Global registry
VERIFIER_REGISTRY: dict[str, type[BaseVerifier]] = {}


def register_verifier(name: str):
    """
    Decorator to register a verifier class.

    Usage:
        @register_verifier("mcq-rlvr")
        class MCQRLVRVerifier(BaseVerifier):
            ...
    """

    def decorator(cls: type[BaseVerifier]):
        if name in VERIFIER_REGISTRY:
            raise ValueError(f"Verifier '{name}' is already registered")
        VERIFIER_REGISTRY[name] = cls
        return cls

    return decorator


def get_verifier(name: str, **kwargs) -> BaseVerifier:
    """
    Get a verifier instance by name.

    Args:
        name: Registered verifier name (e.g., "mcq-rlvr", "mcq-llm-as-judge")
        **kwargs: Additional arguments to pass to the verifier constructor

    Returns:
        Verifier instance

    Raises:
        ValueError: If verifier name is not registered
    """
    if name not in VERIFIER_REGISTRY:
        available = list(VERIFIER_REGISTRY.keys())
        raise ValueError(f"Unknown verifier: '{name}'. Available: {available}")
    return VERIFIER_REGISTRY[name](**kwargs)


def list_verifiers() -> list[str]:
    """List all registered verifier names."""
    return list(VERIFIER_REGISTRY.keys())
