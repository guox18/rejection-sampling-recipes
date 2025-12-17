"""Pytest fixtures and configuration."""

import os
import sys
from pathlib import Path

import pytest

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 测试数据路径
TEST_DATA_DIR = PROJECT_ROOT / "data" / "Nemotron-Post-Training-Dataset-v2" / "datasets"
TEST_DATA_FILE = TEST_DATA_DIR / "train_30.jsonl"


@pytest.fixture
def test_data_path():
    """返回测试数据文件路径."""
    return str(TEST_DATA_FILE)


@pytest.fixture
def sample_mcq_item():
    """返回一个示例 MCQ 数据项."""
    return {
        "id": "test-001",
        "messages": [
            {
                "role": "user",
                "content": "What is 2 + 2?\nA: 3\nB: 4\nC: 5\nD: 6",
            }
        ],
        "metadata": {
            "answer": "B",
            "original_uuid": "test-001",
            "source": "test",
        },
    }


@pytest.fixture
def sample_batch(sample_mcq_item):
    """返回一个示例 batch."""
    return [
        sample_mcq_item,
        {
            "id": "test-002",
            "messages": [
                {
                    "role": "user",
                    "content": "Which planet is closest to the Sun?\nA: Venus\nB: Mercury\nC: Mars\nD: Earth",
                }
            ],
            "metadata": {
                "answer": "B",
                "original_uuid": "test-002",
                "source": "test",
            },
        },
    ]


@pytest.fixture
def sample_batch_with_rollouts(sample_batch):
    """返回带 rollouts 的 batch."""
    return [
        {
            **sample_batch[0],
            "rollouts": [
                {"response": "The answer is B.", "score": 1.0},
                {"response": "The answer is A.", "score": 0.0},
            ],
        },
        {
            **sample_batch[1],
            "rollouts": [
                {"response": "The answer is B.", "score": 1.0},
            ],
        },
    ]
