"""
Tests for formatter implementations.
"""

import pytest

from src.formatter import (
    DPOFormatter,
    MultiSFTFormatter,
    SFTFormatter,
    get_formatter,
    list_formatters,
)


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


class TestMultiSFTFormatter:
    """Tests for Multi-SFT formatter."""

    @pytest.fixture
    def formatter(self):
        return MultiSFTFormatter(num_responses=3, pass_threshold=1.0, fail_threshold=0.0)

    @pytest.fixture
    def sample_item(self):
        return {
            "id": "001",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "metadata": {"answer": "4"},
        }

    def test_format_with_enough_unique_passes(self, formatter, sample_item):
        """Should format when there are enough unique passing responses."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "2+2 equals 4", "score": 1.0},
            {"response": "It's 4", "score": 1.0},
            {"response": "Wrong answer", "score": 0.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 1
        assert "messages" in result[0]
        assert "responses" in result[0]
        assert len(result[0]["responses"]) == 3

    def test_format_not_enough_unique_passes(self, formatter, sample_item):
        """Should return empty when not enough unique passing responses."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "2+2 equals 4", "score": 1.0},
            {"response": "Wrong answer", "score": 0.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 0

    def test_format_deduplicates_responses(self, formatter, sample_item):
        """Should deduplicate identical responses."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "The answer is 4", "score": 1.0},  # Duplicate
            {"response": "2+2 equals 4", "score": 1.0},
            {"response": "It's 4", "score": 1.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 1
        assert len(result[0]["responses"]) == 3
        # All responses should be unique
        assert len(set(result[0]["responses"])) == 3

    def test_format_with_duplicates_not_enough(self, formatter, sample_item):
        """Should return empty when duplicates cause not enough unique responses."""
        rollouts = [
            {"response": "The answer is 4", "score": 1.0},
            {"response": "The answer is 4", "score": 1.0},  # Duplicate
            {"response": "The answer is 4", "score": 1.0},  # Duplicate
            {"response": "2+2 equals 4", "score": 1.0},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 0  # Only 2 unique responses, need 3

    def test_format_empty_rollouts(self, formatter, sample_item):
        """Should return empty for empty rollouts."""
        result = formatter.format(sample_item, [])
        assert len(result) == 0

    def test_is_satisfied_with_enough_unique(self, formatter):
        """Satisfied when enough unique passes."""
        rollouts = [
            {"response": "response1", "score": 1.0},
            {"response": "response2", "score": 1.0},
            {"response": "response3", "score": 1.0},
        ]
        assert formatter.is_satisfied(rollouts) is True

    def test_is_satisfied_not_enough(self, formatter):
        """Not satisfied when not enough unique passes."""
        rollouts = [
            {"response": "response1", "score": 1.0},
            {"response": "response2", "score": 1.0},
        ]
        assert formatter.is_satisfied(rollouts) is False

    def test_is_satisfied_with_duplicates(self, formatter):
        """Not satisfied when duplicates reduce unique count below threshold."""
        rollouts = [
            {"response": "response1", "score": 1.0},
            {"response": "response1", "score": 1.0},  # Duplicate
            {"response": "response2", "score": 1.0},
        ]
        assert formatter.is_satisfied(rollouts) is False

    def test_selects_best_scores(self, formatter, sample_item):
        """Should select responses with highest scores when more than needed."""
        rollouts = [
            {"response": "low score", "score": 1.0},
            {"response": "mid score", "score": 1.5},
            {"response": "high score", "score": 2.0},
            {"response": "highest score", "score": 2.5},
        ]
        result = formatter.format(sample_item, rollouts)
        assert len(result) == 1
        # Should have top 3 by score
        assert "highest score" in result[0]["responses"]
        assert "high score" in result[0]["responses"]
        assert "mid score" in result[0]["responses"]
        assert "low score" not in result[0]["responses"]


class TestFormatterRegistry:
    """Tests for formatter registry."""

    def test_list_formatters(self):
        """Should list registered formatters."""
        formatters = list_formatters()
        assert "sft" in formatters
        assert "dpo" in formatters
        assert "multi_sft" in formatters

    def test_get_formatter(self):
        """Should get formatter by name."""
        formatter = get_formatter("sft")
        assert isinstance(formatter, SFTFormatter)

    def test_get_multi_sft_formatter(self):
        """Should get multi_sft formatter with custom params."""
        formatter = get_formatter("multi_sft", num_responses=16)
        assert isinstance(formatter, MultiSFTFormatter)
        assert formatter.num_responses == 16

    def test_unknown_formatter(self):
        """Should raise for unknown formatter."""
        with pytest.raises(ValueError):
            get_formatter("unknown-formatter")
