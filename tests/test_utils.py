"""
Tests for utility functions.
"""

from src.utils import clip_thinking, has_final_answer, split_response


class TestSplitResponse:
    """Tests for split_response function."""

    def test_r1_format(self, mcq_responses):
        """Test R1 <think>...</think> format."""
        for item in mcq_responses:
            if item["model_type"] == "r1":
                thinking, response = split_response(item["raw_response"])
                assert thinking == item["expected"]["thinking"]
                assert response == item["expected"]["response"]

    def test_gpt_oss_format(self, mcq_responses):
        """Test GPT-OSS channel format."""
        for item in mcq_responses:
            if item["model_type"] == "gpt-oss":
                thinking, response = split_response(item["raw_response"])
                assert thinking == item["expected"]["thinking"]
                assert response == item["expected"]["response"]

    def test_no_reasoning(self, mcq_responses):
        """Test responses without reasoning tags."""
        for item in mcq_responses:
            if not item["has_reasoning"]:
                thinking, response = split_response(item["raw_response"])
                assert thinking == ""
                assert response == item["raw_response"].strip()

    def test_empty_response(self):
        """Test empty string input."""
        thinking, response = split_response("")
        assert thinking == ""
        assert response == ""

    def test_only_closing_think_tag(self, mcq_responses):
        """Test responses with only </think> tag (some chat templates)."""
        for item in mcq_responses:
            if item["id"] == "only_think_tag_end":
                thinking, response = split_response(item["raw_response"])
                assert "step by step" in thinking
                assert "\\boxed{E}" in response


class TestClipThinking:
    """Tests for clip_thinking function."""

    def test_returns_only_response(self, mcq_responses):
        """clip_thinking should return only the response part."""
        for item in mcq_responses:
            result = clip_thinking(item["raw_response"])
            assert result == item["expected"]["response"]

    def test_empty_input(self):
        """Empty input returns empty string."""
        assert clip_thinking("") == ""

    def test_no_thinking_returns_original(self):
        """Response without thinking tags returns original."""
        text = "The answer is B."
        assert clip_thinking(text) == text


class TestHasFinalAnswer:
    """Tests for has_final_answer function."""

    def test_truncated_has_no_final(self, mcq_responses):
        """Truncated responses should have no final answer."""
        for item in mcq_responses:
            if "truncated" in item["id"].lower():
                assert has_final_answer(item["raw_response"]) is False

    def test_complete_has_final(self, mcq_responses):
        """Complete responses should have final answer."""
        for item in mcq_responses:
            if "truncated" not in item["id"].lower() and item["raw_response"]:
                assert has_final_answer(item["raw_response"]) is True

    def test_empty_has_no_final(self):
        """Empty response has no final answer."""
        assert has_final_answer("") is False
