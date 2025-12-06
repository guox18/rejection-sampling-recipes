"""
Tests for verifier implementations.
"""

import pytest

from src.verifier import MathRLVRVerifier, MCQRLVRVerifier, get_verifier, list_verifiers


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


class TestMathRLVRVerifier:
    """Tests for Math RLVR (rule-based) verifier."""

    @pytest.fixture
    def verifier(self):
        return MathRLVRVerifier()

    def test_boxed_integer(self, verifier):
        """Test \\boxed{number} extraction for integers."""
        cases = [
            ("\\boxed{42}", "42", 1.0),
            ("\\boxed{42}", "41", 0.0),
            ("The answer is \\boxed{100}.", "100", 1.0),
            ("Therefore, \\boxed{-5}", "-5", 1.0),
        ]
        for response, answer, expected_score in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected_score, f"Failed for {response}"

    def test_boxed_decimal(self, verifier):
        """Test decimal number extraction."""
        cases = [
            ("\\boxed{3.14}", "3.14", 1.0),
            ("\\boxed{3.14159}", "3.14159", 1.0),
            ("\\boxed{0.5}", "0.5", 1.0),
        ]
        for response, answer, expected_score in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected_score, f"Failed for {response}"

    def test_boxed_fraction(self, verifier):
        """Test fraction extraction."""
        cases = [
            ("\\boxed{\\frac{1}{2}}", "0.5", 1.0),
            ("\\boxed{\\frac{1}{2}}", "1/2", 1.0),
            ("\\boxed{1/2}", "0.5", 1.0),
            ("\\boxed{3/4}", "0.75", 1.0),
        ]
        for response, answer, expected_score in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected_score, f"Failed for {response}"

    def test_percentage(self, verifier):
        """Test percentage handling."""
        # Percentage in ground truth
        assert verifier.verify("\\boxed{0.5}", {"answer": "50%"}) == 1.0
        assert verifier.verify("\\boxed{50%}", {"answer": "0.5"}) == 1.0

    def test_empty_response(self, verifier):
        """Empty response should score 0."""
        assert verifier.verify("", {"answer": "42"}) == 0.0

    def test_no_answer_in_metadata(self, verifier):
        """Missing answer in metadata should score 0."""
        assert verifier.verify("\\boxed{42}", {}) == 0.0

    def test_extract_answer(self, verifier):
        """Test answer extraction."""
        assert verifier.extract_answer("\\boxed{42}") == "42"
        assert verifier.extract_answer("\\boxed{\\frac{1}{2}}") == "1/2"
        assert verifier.extract_answer("No boxed answer") is None

    def test_float_tolerance(self, verifier):
        """Test floating point comparison with tolerance."""
        # These should be considered equal within tolerance
        assert verifier.verify("\\boxed{0.333333}", {"answer": "1/3"}) == 1.0


class TestVerifierRegistry:
    """Tests for verifier registry."""

    def test_list_verifiers(self):
        """Should list registered verifiers."""
        verifiers = list_verifiers()
        assert "mcq-rlvr" in verifiers
        assert "mcq-llm-as-judge" in verifiers
        assert "math-rlvr" in verifiers

    def test_get_verifier(self):
        """Should get verifier by name."""
        verifier = get_verifier("mcq-rlvr")
        assert isinstance(verifier, MCQRLVRVerifier)

    def test_get_math_verifier(self):
        """Should get math verifier by name."""
        verifier = get_verifier("math-rlvr")
        assert isinstance(verifier, MathRLVRVerifier)

    def test_unknown_verifier(self):
        """Should raise for unknown verifier."""
        with pytest.raises(ValueError):
            get_verifier("unknown-verifier")
