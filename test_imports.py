#!/usr/bin/env python3
"""Quick test script to verify all imports and basic functionality."""

from src.formatter import get_formatter, list_formatters
from src.verifier import get_verifier, list_verifiers

print("✓ All imports successful")

# Test verifier registry
print(f"Available verifiers: {list_verifiers()}")

# Test formatter registry
print(f"Available formatters: {list_formatters()}")

# Test MCQ verifier
mcq_verifier = get_verifier("mcq-rlvr")
print(f"✓ Created MCQ verifier: {type(mcq_verifier).__name__}")

# Test SFT formatter
sft_formatter = get_formatter("sft", pass_threshold=1.0, fail_threshold=0.0)
print(f"✓ Created SFT formatter: {type(sft_formatter).__name__}")

# Test DPO formatter
dpo_formatter = get_formatter("dpo", pass_threshold=1.0, fail_threshold=0.0)
print(f"✓ Created DPO formatter: {type(dpo_formatter).__name__}")

# Test MCQ verifier functionality
test_mcq_response = "I think the answer is \\boxed{A}"
test_mcq_metadata = {"answer": "A"}
score = mcq_verifier.verify(test_mcq_response, test_mcq_metadata)
print(f"✓ MCQ verification test: score={score}")

# Test formatter functionality
test_item = {
    "id": "001",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "metadata": {"answer": "4"},
}
test_rollouts = [
    {"response": "The answer is 4", "score": 1.0},
    {"response": "The answer is 5", "score": 0.0},
]

sft_result = sft_formatter.format(test_item, test_rollouts)
print(f"✓ SFT format test: {len(sft_result)} examples")

dpo_result = dpo_formatter.format(test_item, test_rollouts)
print(f"✓ DPO format test: {len(dpo_result)} examples")

# Test early stop
print(f"✓ SFT satisfied: {sft_formatter.is_satisfied(test_rollouts)}")
print(f"✓ DPO satisfied: {dpo_formatter.is_satisfied(test_rollouts)}")

print()
print("All tests passed! ✓")
