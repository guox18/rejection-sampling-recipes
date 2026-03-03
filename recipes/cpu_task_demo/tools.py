"""Utility functions for CPU task demo recipe."""

from __future__ import annotations

from math import isqrt


def count_primes_up_to(limit: int) -> int:
    """Count primes in [2, limit] using trial division."""
    if limit < 2:
        return 0

    count = 1
    for candidate in range(3, limit + 1, 2):
        is_prime = True
        upper = isqrt(candidate)
        divisor = 3
        while divisor <= upper:
            if candidate % divisor == 0:
                is_prime = False
                break
            divisor += 2
        if is_prime:
            count += 1
    return count


def run_prime_workload(limit: int, rounds: int) -> int:
    """Run repeated prime counting as a CPU-heavy workload."""
    total = 0
    for _ in range(rounds):
        total += count_primes_up_to(limit)
    return total
