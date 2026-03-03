"""Tests for CPU task demo recipe."""

from recipes.cpu_task_demo.config import CpuTaskDemoConfig
from recipes.cpu_task_demo.recipe import CpuIntensiveStage
from recipes.cpu_task_demo.tools import count_primes_up_to, run_prime_workload


def test_count_primes_up_to():
    """Prime counter returns expected values on small inputs."""
    assert count_primes_up_to(1) == 0
    assert count_primes_up_to(2) == 1
    assert count_primes_up_to(10) == 4
    assert count_primes_up_to(30) == 10


def test_run_prime_workload():
    """CPU workload repeats the same computation across rounds."""
    assert run_prime_workload(10, 1) == 4
    assert run_prime_workload(10, 3) == 12


def test_cpu_intensive_stage_process_item():
    """Stage writes cpu_result and worker metadata."""
    config = CpuTaskDemoConfig(prime_limit=10, rounds=2)
    stage = CpuIntensiveStage(config)

    result = stage.process_item({"id": "demo-item"})

    assert result["cpu_result"]["aggregate_prime_count"] == 8
    assert result["metadata"]["prime_limit"] == 10
    assert result["metadata"]["rounds"] == 2
    assert result["metadata"]["worker_pid"] > 0
