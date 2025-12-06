"""
Tests for verifier module.
"""

import pytest

from src.verifier import MCQRLVRVerifier, get_verifier, list_verifiers


class TestRegistry:
    """Test verifier registry."""

    def test_list_verifiers(self):
        """Test listing registered verifiers."""
        verifiers = list_verifiers()
        assert "mcq-rlvr" in verifiers
        assert "mcq-llm-as-judge" in verifiers

    def test_get_verifier(self):
        """Test getting verifier by name."""
        verifier = get_verifier("mcq-rlvr")
        assert isinstance(verifier, MCQRLVRVerifier)

    def test_get_unknown_verifier(self):
        """Test getting unknown verifier raises error."""
        with pytest.raises(ValueError, match="Unknown verifier"):
            get_verifier("unknown-verifier")


class TestMCQRLVRVerifier:
    """Test MCQ RLVR verifier."""

    @pytest.fixture
    def verifier(self):
        return MCQRLVRVerifier()

    # ==================== extract_answer tests ====================

    def test_extract_boxed_simple(self, verifier):
        """Test extracting from simple \\boxed{A} format."""
        assert verifier.extract_answer("\\boxed{A}") == "A"
        assert verifier.extract_answer("\\boxed{B}") == "B"
        assert verifier.extract_answer("\\boxed{C}") == "C"

    def test_extract_boxed_with_text(self, verifier):
        """Test extracting from \\boxed{\\text{A}} format."""
        assert verifier.extract_answer("\\boxed{\\text{A}}") == "A"
        assert verifier.extract_answer("\\boxed{\\text{B}}") == "B"
        assert verifier.extract_answer("\\boxed{\\mathrm{C}}") == "C"
        assert verifier.extract_answer("\\boxed{\\mathbf{D}}") == "D"

    def test_extract_boxed_with_think_tags(self, verifier):
        """Test extracting answer when response has think tags."""
        response = "</think>The answer is B</think> \\boxed{B}"
        assert verifier.extract_answer(response) == "B"

        response = "<think>Let me think...</think>\\boxed{C}"
        assert verifier.extract_answer(response) == "C"

    def test_extract_chinese_format(self, verifier):
        """Test extracting from Chinese format."""
        assert verifier.extract_answer("答案是A") == "A"
        assert verifier.extract_answer("答案：B") == "B"
        assert verifier.extract_answer("答案为 C") == "C"

    def test_extract_fallback_last_letter(self, verifier):
        """Test fallback extraction from last letter."""
        assert verifier.extract_answer("Therefore, the answer is A.") == "A"
        assert verifier.extract_answer("So the correct option is B") == "B"

    def test_extract_empty_response(self, verifier):
        """Test extraction from empty response."""
        assert verifier.extract_answer("") is None
        assert verifier.extract_answer(None) is None

    def test_extract_no_answer(self, verifier):
        """Test extraction when no answer found."""
        assert verifier.extract_answer("This is some random text") is None

    # ==================== verify tests ====================

    def test_verify_correct(self, verifier):
        """Test verification of correct answer."""
        response = "\\boxed{A}"
        metadata = {"answer": "A"}
        assert verifier.verify(response, metadata) == 1.0

    def test_verify_incorrect(self, verifier):
        """Test verification of incorrect answer."""
        response = "\\boxed{A}"
        metadata = {"answer": "B"}
        assert verifier.verify(response, metadata) == 0.0

    def test_verify_case_insensitive(self, verifier):
        """Test case-insensitive verification."""
        response = "\\boxed{a}"
        metadata = {"answer": "A"}
        assert verifier.verify(response, metadata) == 1.0

    def test_verify_gold_target_format(self, verifier):
        """Test verification with gold_target metadata format."""
        response = "\\boxed{B}"
        metadata = {"gold_target": "B"}
        assert verifier.verify(response, metadata) == 1.0

    def test_verify_no_answer_in_metadata(self, verifier):
        """Test verification when metadata has no answer."""
        response = "\\boxed{A}"
        metadata = {}
        assert verifier.verify(response, metadata) == 0.0

    def test_verify_extraction_failed(self, verifier):
        """Test verification when answer extraction fails."""
        response = "Random text without answer"
        metadata = {"answer": "A"}
        assert verifier.verify(response, metadata) == 0.0


class TestMCQLLMJudgeVerifier:
    """Test MCQ LLM-as-Judge verifier."""

    def test_clip_thinking_r1_format(self):
        """Test clipping thinking from R1 format."""
        from src.verifier.mcq_llm_judge import MCQLLMJudgeVerifier

        verifier = MCQLLMJudgeVerifier.__new__(MCQLLMJudgeVerifier)

        # R1 format
        response = "<think>Let me think about this...</think>The answer is A"
        clipped = verifier._clip_thinking(response)
        assert clipped == "The answer is A"

        # Multiple think tags
        response = "<think>First thought</think>Middle<think>Second</think>Final"
        clipped = verifier._clip_thinking(response)
        assert clipped == "Final"

    def test_clip_thinking_no_tags(self):
        """Test clipping when no thinking tags."""
        from src.verifier.mcq_llm_judge import MCQLLMJudgeVerifier

        verifier = MCQLLMJudgeVerifier.__new__(MCQLLMJudgeVerifier)

        response = "The answer is B"
        clipped = verifier._clip_thinking(response)
        assert clipped == "The answer is B"

    def test_parse_judge_output(self):
        """Test parsing judge output."""
        from src.verifier.mcq_llm_judge import MCQLLMJudgeVerifier

        verifier = MCQLLMJudgeVerifier.__new__(MCQLLMJudgeVerifier)

        assert verifier._parse_judge_output("A") is True
        assert verifier._parse_judge_output("a") is True
        assert verifier._parse_judge_output("CORRECT") is True
        assert verifier._parse_judge_output("correct") is True

        assert verifier._parse_judge_output("B") is False
        assert verifier._parse_judge_output("b") is False
        assert verifier._parse_judge_output("INCORRECT") is False
        assert verifier._parse_judge_output("incorrect") is False

        assert verifier._parse_judge_output("") is False
        assert verifier._parse_judge_output("unknown") is False
