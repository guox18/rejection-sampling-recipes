# Verifier Development Guide

This document explains how to contribute a new Verifier to the project.

## Directory Structure

```
src/verifier/
├── __init__.py          # Export all verifiers
├── base.py              # BaseVerifier abstract class
├── registry.py          # Registry
├── mcq_rlvr.py          # MCQ rule-based verifier (R1 models)
├── mcq_llm_judge.py     # MCQ LLM judge
└── README.md            # This document
```

## Implementation Steps

### 1. Create Verifier File

```python
# src/verifier/my_verifier.py
from .base import BaseVerifier
from .registry import register_verifier


@register_verifier("my-verifier")  # Registration name for config files
class MyVerifier(BaseVerifier):
    """
    My verifier description.

    Supports:
    - Format 1: xxx
    - Format 2: yyy
    """

    def __init__(self, **kwargs):
        """
        Initialize verifier.

        Accept **kwargs for compatibility - pipeline may pass unused config.
        """
        # If configuration is needed, extract from kwargs
        # self.some_option = kwargs.get("some_option", default_value)
        pass

    def verify(self, response: str, metadata: dict) -> float:
        """
        Verify response against ground truth.

        Args:
            response: Model's raw response
            metadata: Dict containing 'answer' key with ground truth

        Returns:
            Score: 1.0 for correct, 0.0 for incorrect
        """
        # 1. Extract ground truth answer
        ground_truth = metadata.get("answer")
        if not ground_truth:
            return 0.0

        # 2. Extract model's answer from response
        extracted = self.extract_answer(response)
        if not extracted:
            return 0.0

        # 3. Compare
        return 1.0 if extracted == ground_truth else 0.0

    def extract_answer(self, response: str) -> str | None:
        """Extract answer from response."""
        # Implement extraction logic
        pass
```

### 2. Export in `__init__.py`

```python
# src/verifier/__init__.py
from .my_verifier import MyVerifier

__all__ = [
    # ... existing exports
    "MyVerifier",
]
```

### 3. Add Test Data

Add test cases in `tests/fixtures/model_outputs.json`:

```json
{
  "my_verifier_responses": [
    {
      "id": "case_correct",
      "model_type": "xxx",
      "raw_response": "Model output...",
      "expected": {
        "extracted_answer": "A"
      },
      "ground_truth": "A",
      "expected_score": 1.0
    },
    {
      "id": "case_wrong",
      "raw_response": "Wrong output...",
      "ground_truth": "A",
      "expected_score": 0.0
    },
    {
      "id": "case_empty",
      "raw_response": "",
      "ground_truth": "A",
      "expected_score": 0.0
    }
  ]
}
```

### 4. Add Test Class

Add in `tests/test_verifier.py`:

```python
class TestMyVerifier:
    """Tests for MyVerifier."""

    @pytest.fixture
    def verifier(self):
        return MyVerifier()

    def test_correct_answer(self, verifier):
        """Correct answer scores 1.0"""
        score = verifier.verify("Correct output", {"answer": "A"})
        assert score == 1.0

    def test_wrong_answer(self, verifier):
        """Wrong answer scores 0.0"""
        score = verifier.verify("Wrong output", {"answer": "A"})
        assert score == 0.0

    def test_empty_response(self, verifier):
        """Empty response scores 0.0"""
        assert verifier.verify("", {"answer": "A"}) == 0.0

    def test_no_answer_in_metadata(self, verifier):
        """Missing ground truth scores 0.0"""
        assert verifier.verify("Output", {}) == 0.0

    def test_with_fixtures(self, verifier, model_outputs):
        """Batch test using fixtures"""
        for item in model_outputs.get("my_verifier_responses", []):
            score = verifier.verify(
                item["raw_response"],
                {"answer": item["ground_truth"]}
            )
            assert score == item["expected_score"], f"Failed: {item['id']}"
```

## Testing Requirements

Before submitting a PR, ensure:

- [ ] At least 5 test cases
- [ ] Coverage: correct/wrong/empty input/missing metadata
- [ ] `pytest tests/test_verifier.py -v` passes
- [ ] `ruff check src/verifier/` passes

## Using the New Verifier

Use the registration name in config files:

```yaml
verifier:
  type: my-verifier  # @register_verifier("my-verifier")
```

## Shared Utility Functions

If you need to handle thinking tags, use shared utilities:

```python
from ..utils import clip_thinking, split_response

# Remove thinking process, keep only final answer
final_answer = clip_thinking(response)

# Separate thinking and answer
thinking, answer = split_response(response)
```
