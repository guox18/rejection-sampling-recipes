"""
Tests for formatter implementations.
"""

import pytest

from src.formatter import DPOFormatter, SFTFormatter, get_formatter, list_formatters


class TestSFTFormatter:
    """Tests for SFT formatter."""

    @pytest.fixture
    def formatter(self):
        return SFTFormatter(pass_threshold=1.0, fail_threshold=0.0)

    @pytest.fixture
    def sample_item(self):
        return {
            "id": "001",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "metadata": {"answer": "4"},
        }

    def test_format_with_pass(self, formatter, sample_item):
        """Should format when there's a passing response."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "Maybe 5?", "score": 0.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 1
        assert result[0]["messages"][-1]["role"] == "assistant"
        assert result[0]["messages"][-1]["content"] == "The answer is 4"

    def test_format_no_pass(self, formatter, sample_item):
        """Should return empty when no passing response."""
        rollouts = [
            {"response": "Maybe 5?", "score": 0.0},
            {"response": "I think 3", "score": 0.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 0

    def test_format_empty_rollouts(self, formatter, sample_item):
        """Should return empty for empty rollouts."""
        result = formatter.format(sample_item, [])
        assert len(result) == 0

    def test_is_satisfied_with_pass(self, formatter):
        """Satisfied when at least 1 pass."""
        rollouts = [{"response": "...", "score": 1.0}]
        assert formatter.is_satisfied(rollouts) is True

    def test_is_satisfied_no_pass(self, formatter):
        """Not satisfied when no pass."""
        rollouts = [{"response": "...", "score": 0.5}]
        assert formatter.is_satisfied(rollouts) is False

    def test_selects_best_score(self, formatter, sample_item):
        """Should select response with highest score."""
        rollouts = [
            {"response": "Good answer", "score": 1.0},
            {"response": "Better answer", "score": 1.0},  # Same score, first wins
        ]
        result = formatter.format(sample_item, rollouts)
        assert result[0]["messages"][-1]["content"] == "Good answer"


class TestDPOFormatter:
    """Tests for DPO formatter."""

    @pytest.fixture
    def formatter(self):
        return DPOFormatter(pass_threshold=1.0, fail_threshold=0.0)

    @pytest.fixture
    def sample_item(self):
        return {
            "id": "001",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "metadata": {"answer": "4"},
        }

    def test_format_with_pass_and_fail(self, formatter, sample_item):
        """Should format when there's both pass and fail."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "The answer is 5", "score": 0.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 1
        assert result[0]["chosen"][0]["content"] == "The answer is 4"
        assert result[0]["rejected"][0]["content"] == "The answer is 5"

    def test_format_only_pass(self, formatter, sample_item):
        """Should return empty when only pass, no fail."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "4 is the answer", "score": 1.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 0

    def test_format_only_fail(self, formatter, sample_item):
        """Should return empty when only fail, no pass."""
        rollouts = [
            {"response": "Maybe 5?", "score": 0.0},
            {"response": "I think 3", "score": 0.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 0

    def test_is_satisfied_with_both(self, formatter):
        """Satisfied when both pass and fail exist."""
        rollouts = [
            {"response": "...", "score": 1.0},
            {"response": "...", "score": 0.0},
        ]
        assert formatter.is_satisfied(rollouts) is True

    def test_is_satisfied_only_pass(self, formatter):
        """Not satisfied with only pass."""
        rollouts = [{"response": "...", "score": 1.0}]
        assert formatter.is_satisfied(rollouts) is False


class TestFormatterRegistry:
    """Tests for formatter registry."""

    def test_list_formatters(self):
        """Should list registered formatters."""
        formatters = list_formatters()
        assert "sft" in formatters
        assert "dpo" in formatters

    def test_get_formatter(self):
        """Should get formatter by name."""
        formatter = get_formatter("sft")
        assert isinstance(formatter, SFTFormatter)

    def test_unknown_formatter(self):
        """Should raise for unknown formatter."""
        with pytest.raises(ValueError):
            get_formatter("unknown-formatter")
