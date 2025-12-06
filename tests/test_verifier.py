"""
Tests for verifier implementations.
"""

import pytest

from src.verifier import MCQRLVRVerifier, get_verifier, list_verifiers


class TestMCQRLVRVerifier:
    """Tests for MCQ RLVR (rule-based) verifier."""

    @pytest.fixture
    def verifier(self):
        return MCQRLVRVerifier()

    def test_boxed_format(self, verifier):
        """Test \\boxed{X} extraction."""
        cases = [
            ("\\boxed{A}", "A", 1.0),
            ("\\boxed{B}", "A", 0.0),
            ("The answer is \\boxed{E}.", "E", 1.0),
            ("Therefore, \\boxed{\\text{C}}", "C", 1.0),
        ]
        for response, answer, expected_score in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected_score, f"Failed for {response}"

    def test_chinese_format(self, verifier):
        """Test Chinese answer format."""
        cases = [
            ("答案是 A", "A", 1.0),
            ("答案为：B", "B", 1.0),
            ("选项 C 正确", "C", 1.0),
        ]
        for response, answer, expected_score in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected_score, f"Failed for {response}"

    def test_case_insensitive(self, verifier):
        """Test case-insensitive matching."""
        assert verifier.verify("\\boxed{a}", {"answer": "A"}) == 1.0
        assert verifier.verify("\\boxed{A}", {"answer": "a"}) == 1.0

    def test_empty_response(self, verifier):
        """Empty response should score 0."""
        assert verifier.verify("", {"answer": "A"}) == 0.0

    def test_no_answer_in_metadata(self, verifier):
        """Missing answer in metadata should score 0."""
        assert verifier.verify("\\boxed{A}", {}) == 0.0

    def test_extract_answer(self, verifier):
        """Test answer extraction."""
        assert verifier.extract_answer("\\boxed{E}") == "E"
        assert verifier.extract_answer("\\boxed{\\text{B}}") == "B"
        assert verifier.extract_answer("No boxed answer") is None

    def test_r1_style_responses(self, verifier, mcq_responses):
        """Test with R1-style fixture responses."""
        for item in mcq_responses:
            if item["model_type"] == "r1" and item["expected"]["extracted_answer"]:
                # R1 verifier expects \boxed{} format
                response = item["raw_response"]
                if "\\boxed{" in response:
                    score = verifier.verify(response, {"answer": item["ground_truth"]})
                    assert score == item["expected_score"], f"Failed for {item['id']}"


class TestVerifierRegistry:
    """Tests for verifier registry."""

    def test_list_verifiers(self):
        """Should list registered verifiers."""
        verifiers = list_verifiers()
        assert "mcq-rlvr" in verifiers
        assert "mcq-llm-as-judge" in verifiers

    def test_get_verifier(self):
        """Should get verifier by name."""
        verifier = get_verifier("mcq-rlvr")
        assert isinstance(verifier, MCQRLVRVerifier)

    def test_unknown_verifier(self):
        """Should raise for unknown verifier."""
        with pytest.raises(ValueError):
            get_verifier("unknown-verifier")
