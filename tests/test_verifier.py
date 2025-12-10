"""
Tests for verifier implementations.
"""

import pytest

from src.verifier import MathRLVRVerifier, MCQRLVRVerifier, get_verifier, list_verifiers


class TestMCQRLVRVerifier:
    """Tests for MCQ RLVR (rule-based) verifier - \\boxed{} extraction only."""

    @pytest.fixture
    def verifier(self):
        return MCQRLVRVerifier()

    # ==========================================================================
    # Basic \boxed{} format tests
    # ==========================================================================

    def test_boxed_basic(self, verifier):
        """Test basic \\boxed{X} extraction."""
        cases = [
            ("\\boxed{A}", "A", 1.0),
            ("\\boxed{B}", "B", 1.0),
            ("\\boxed{C}", "C", 1.0),
            ("\\boxed{D}", "D", 1.0),
            ("\\boxed{E}", "E", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    def test_boxed_wrong_answer(self, verifier):
        """Test wrong answer in \\boxed{} returns 0."""
        cases = [
            ("\\boxed{A}", "B", 0.0),
            ("\\boxed{B}", "A", 0.0),
            ("\\boxed{C}", "E", 0.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    def test_boxed_with_context(self, verifier):
        """Test \\boxed{} extraction with surrounding text."""
        cases = [
            ("The answer is \\boxed{A}.", "A", 1.0),
            ("After analysis, I conclude \\boxed{B}", "B", 1.0),
            ("Therefore, \\boxed{C}.", "C", 1.0),
            ("\\boxed{D} is the correct choice.", "D", 1.0),
            ("Step 1: ... Step 2: ... Therefore \\boxed{E}", "E", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    # ==========================================================================
    # LaTeX text command tests
    # ==========================================================================

    def test_boxed_with_text(self, verifier):
        """Test \\boxed{\\text{X}} format."""
        cases = [
            ("\\boxed{\\text{A}}", "A", 1.0),
            ("\\boxed{\\text{B}}", "B", 1.0),
            ("Therefore, \\boxed{\\text{C}}", "C", 1.0),
            ("\\boxed{\\text{ A }}", "A", 1.0),  # with whitespace
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    def test_boxed_with_textbf(self, verifier):
        """Test \\boxed{\\textbf{X}} format."""
        cases = [
            ("\\boxed{\\textbf{A}}", "A", 1.0),
            ("\\boxed{\\textbf{B}}", "B", 1.0),
            ("Therefore, \\boxed{\\textbf{C}}", "C", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    def test_boxed_with_mathrm(self, verifier):
        """Test \\boxed{\\mathrm{X}} format."""
        cases = [
            ("\\boxed{\\mathrm{A}}", "A", 1.0),
            ("\\boxed{\\mathrm{B}}", "B", 1.0),
            ("The answer is \\boxed{\\mathrm{C}}", "C", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    def test_boxed_with_mathbf(self, verifier):
        """Test \\boxed{\\mathbf{X}} format."""
        cases = [
            ("\\boxed{\\mathbf{A}}", "A", 1.0),
            ("\\boxed{\\mathbf{B}}", "B", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    # ==========================================================================
    # Dollar sign wrapper tests
    # ==========================================================================

    def test_boxed_with_dollar_signs(self, verifier):
        """Test $\\boxed{X}$ format."""
        cases = [
            ("$\\boxed{A}$", "A", 1.0),
            ("$\\boxed{B}$", "B", 1.0),
            ("The answer is $\\boxed{C}$.", "C", 1.0),
            ("$\\boxed{\\text{D}}$", "D", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    # ==========================================================================
    # Case sensitivity tests
    # ==========================================================================

    def test_case_insensitive_answer(self, verifier):
        """Test case-insensitive matching for answers."""
        # lowercase in boxed
        assert verifier.verify("\\boxed{a}", {"answer": "A"}) == 1.0
        assert verifier.verify("\\boxed{b}", {"answer": "B"}) == 1.0

        # uppercase in boxed, lowercase in metadata
        assert verifier.verify("\\boxed{A}", {"answer": "a"}) == 1.0
        assert verifier.verify("\\boxed{B}", {"answer": "b"}) == 1.0

    def test_case_insensitive_latex(self, verifier):
        """Test case-insensitive matching with LaTeX commands."""
        assert verifier.verify("\\boxed{\\text{a}}", {"answer": "A"}) == 1.0
        assert verifier.verify("\\boxed{\\TEXT{A}}", {"answer": "a"}) == 1.0

    # ==========================================================================
    # Multiple boxed tests (should use last one)
    # ==========================================================================

    def test_multiple_boxed_uses_last(self, verifier):
        """Test that last \\boxed{} is used when multiple present."""
        # Last answer is correct
        response = "First I thought \\boxed{A}, but actually \\boxed{B}"
        assert verifier.verify(response, {"answer": "B"}) == 1.0
        assert verifier.verify(response, {"answer": "A"}) == 0.0

        # Chain of thought with correction
        response = """
        Initially: \\boxed{C}
        Wait, let me reconsider...
        Actually: \\boxed{D}
        """
        assert verifier.verify(response, {"answer": "D"}) == 1.0
        assert verifier.verify(response, {"answer": "C"}) == 0.0

    def test_multiple_boxed_in_reasoning(self, verifier):
        """Test multiple boxed in detailed reasoning."""
        response = """
        Let me analyze each option:
        - Option A: \\boxed{A} could be correct because...
        - Option B: \\boxed{B} seems unlikely
        - After full analysis, the answer is \\boxed{C}
        """
        assert verifier.verify(response, {"answer": "C"}) == 1.0

    # ==========================================================================
    # Edge cases
    # ==========================================================================

    def test_empty_response(self, verifier):
        """Empty response should score 0."""
        assert verifier.verify("", {"answer": "A"}) == 0.0
        assert verifier.verify("   ", {"answer": "A"}) == 0.0

    def test_no_boxed_in_response(self, verifier):
        """Response without \\boxed{} should score 0."""
        cases = [
            "The answer is A",
            "A is correct",
            "I think the answer is B.",
            "**Answer: C**",
            "答案是 D",
        ]
        for response in cases:
            assert verifier.verify(response, {"answer": "A"}) == 0.0, f"Failed for {response}"

    def test_no_answer_in_metadata(self, verifier):
        """Missing answer in metadata should score 0."""
        assert verifier.verify("\\boxed{A}", {}) == 0.0
        assert verifier.verify("\\boxed{A}", {"wrong_key": "A"}) == 0.0

    def test_invalid_boxed_content(self, verifier):
        """Invalid content in \\boxed{} should score 0."""
        cases = [
            "\\boxed{}",  # empty
            "\\boxed{123}",  # numbers
            "\\boxed{ABC}",  # multiple letters
            "\\boxed{yes}",  # word
        ]
        for response in cases:
            score = verifier.verify(response, {"answer": "A"})
            assert score == 0.0, f"Should reject: {response}"

    def test_boxed_with_whitespace(self, verifier):
        """Test \\boxed{} with internal whitespace."""
        cases = [
            ("\\boxed{ A }", "A", 1.0),
            ("\\boxed{ B}", "B", 1.0),
            ("\\boxed{C }", "C", 1.0),
        ]
        for response, answer, expected in cases:
            score = verifier.verify(response, {"answer": answer})
            assert score == expected, f"Failed for {response}"

    # ==========================================================================
    # extract_answer method tests
    # ==========================================================================

    def test_extract_answer_basic(self, verifier):
        """Test extract_answer returns correct letter."""
        assert verifier.extract_answer("\\boxed{A}") == "A"
        assert verifier.extract_answer("\\boxed{B}") == "B"
        assert verifier.extract_answer("\\boxed{E}") == "E"

    def test_extract_answer_with_latex(self, verifier):
        """Test extract_answer with LaTeX commands."""
        assert verifier.extract_answer("\\boxed{\\text{A}}") == "A"
        assert verifier.extract_answer("\\boxed{\\textbf{B}}") == "B"
        assert verifier.extract_answer("\\boxed{\\mathrm{C}}") == "C"

    def test_extract_answer_returns_none(self, verifier):
        """Test extract_answer returns None when no valid answer."""
        assert verifier.extract_answer("") is None
        assert verifier.extract_answer("No boxed here") is None
        assert verifier.extract_answer("The answer is A") is None
        assert verifier.extract_answer("\\boxed{}") is None
        assert verifier.extract_answer("\\boxed{ABC}") is None

    def test_extract_answer_lowercase(self, verifier):
        """Test extract_answer normalizes to uppercase."""
        assert verifier.extract_answer("\\boxed{a}") == "A"
        assert verifier.extract_answer("\\boxed{\\text{b}}") == "B"

    # ==========================================================================
    # Metadata format tests
    # ==========================================================================

    def test_metadata_answer_key(self, verifier):
        """Test with 'answer' key in metadata."""
        assert verifier.verify("\\boxed{A}", {"answer": "A"}) == 1.0

    def test_metadata_gold_target_key(self, verifier):
        """Test with 'gold_target' key in metadata."""
        assert verifier.verify("\\boxed{A}", {"gold_target": "A"}) == 1.0

    def test_metadata_priority(self, verifier):
        """Test that 'answer' takes priority over 'gold_target'."""
        metadata = {"answer": "A", "gold_target": "B"}
        assert verifier.verify("\\boxed{A}", metadata) == 1.0
        assert verifier.verify("\\boxed{B}", metadata) == 0.0

    # ==========================================================================
    # Real-world format tests
    # ==========================================================================

    def test_r1_style_response(self, verifier):
        """Test R1-style response with think tags."""
        response = """<think>
Let me analyze this step by step.
Option A: ...
Option B: ...
After careful consideration, B is correct.
</think>

The answer is \\boxed{B}"""
        assert verifier.verify(response, {"answer": "B"}) == 1.0

    def test_chain_of_thought_response(self, verifier):
        """Test chain-of-thought response."""
        response = """
Step 1: Understand the problem...
Step 2: Analyze options...
Step 3: Compare A and B...

Based on my analysis, the correct answer is \\boxed{A}.
"""
        assert verifier.verify(response, {"answer": "A"}) == 1.0

    def test_fixture_responses(self, verifier, mcq_responses):
        """Test with fixture responses that use \\boxed{} format."""
        for item in mcq_responses:
            response = item["raw_response"]
            # Only test items that have \boxed{} in the response
            if "\\boxed{" in response:
                expected_answer = item["expected"]["extracted_answer"]
                if expected_answer:
                    extracted = verifier.extract_answer(response)
                    assert extracted == expected_answer, f"Failed for {item['id']}"


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
