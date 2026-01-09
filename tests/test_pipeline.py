"""Tests for Pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.base import BaseRecipe, Stage
from src.pipeline import Pipeline


# ============================================================
# Test Stages
# ============================================================


class PassThroughStage(Stage):
    """简单的透传 Stage, 用于测试."""

    def process(self, batch):
        return batch


class AddFieldStage(Stage):
    """添加字段的 Stage."""

    def __init__(self, field_name: str, field_value):
        self.field_name = field_name
        self.field_value = field_value

    def process(self, batch):
        return [{**item, self.field_name: self.field_value} for item in batch]


# ============================================================
# Test Recipes
# ============================================================


class SimpleRecipe(BaseRecipe):
    """简单的测试 Recipe."""

    def stages(self):
        return [PassThroughStage()]


class MultiStageRecipe(BaseRecipe):
    """多 Stage 的测试 Recipe."""

    def stages(self):
        return [
            AddFieldStage("stage1", "done"),
            AddFieldStage("stage2", "done"),
            AddFieldStage("stage3", "done"),
        ]


# ============================================================
# Tests
# ============================================================


class TestPipelineInit:
    """Pipeline 初始化测试."""

    def test_custom_init(self):
        """自定义参数初始化."""
        recipe = SimpleRecipe(config={})
        pipeline = Pipeline(
            recipe=recipe,
            batch_size=64,
            concurrency=20,
            preserve_order=False,
            resume=False,
        )

        assert pipeline.batch_size == 64
        assert pipeline.concurrency == 20
        assert pipeline.preserve_order is False
        assert pipeline.resume is False

    def test_stage_concurrency_validation(self):
        """stage_concurrency 验证测试."""
        recipe = MultiStageRecipe(config={})

        # 有效配置
        pipeline = Pipeline(
            recipe=recipe,
            stage_concurrency={"AddFieldStage": 5},
        )
        assert pipeline.stage_concurrency == {"AddFieldStage": 5}

        # 无效配置 - 不存在的 Stage 名称
        with pytest.raises(ValueError) as exc_info:
            Pipeline(
                recipe=recipe,
                stage_concurrency={"NonExistentStage": 5},
            )
        assert "Invalid stage names" in str(exc_info.value)


class TestPipelineRun:
    """Pipeline 运行测试."""

    @pytest.fixture
    def temp_files(self):
        """创建临时输入输出文件."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.jsonl"
            output_path = Path(tmpdir) / "output.jsonl"
            yield input_path, output_path

    def test_simple_run(self, temp_files):
        """简单运行测试."""
        input_path, output_path = temp_files

        # 写入测试数据
        test_data = [
            {"id": "1", "value": "a", "metadata": {"_uid": "1"}},
            {"id": "2", "value": "b", "metadata": {"_uid": "2"}},
            {"id": "3", "value": "c", "metadata": {"_uid": "3"}},
        ]
        with open(input_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # 运行 Pipeline
        recipe = SimpleRecipe(config={})
        pipeline = Pipeline(
            recipe=recipe,
            batch_size=2,
            concurrency=1,
            resume=False,
        )
        pipeline.run(str(input_path), str(output_path))

        # 验证输出
        assert output_path.exists()
        with open(output_path) as f:
            output_data = [json.loads(line) for line in f]

        assert len(output_data) == 3

    def test_multi_stage_run(self, temp_files):
        """多 Stage 运行测试."""
        input_path, output_path = temp_files

        # 写入测试数据
        test_data = [
            {"id": "1", "metadata": {"_uid": "1"}},
            {"id": "2", "metadata": {"_uid": "2"}},
        ]
        with open(input_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")

        # 运行 Pipeline
        recipe = MultiStageRecipe(config={})
        pipeline = Pipeline(
            recipe=recipe,
            batch_size=10,
            concurrency=1,
            resume=False,
        )
        pipeline.run(str(input_path), str(output_path))

        # 验证输出 - 每个 item 应该有三个 stage 添加的字段
        with open(output_path) as f:
            output_data = [json.loads(line) for line in f]

        assert len(output_data) == 2
        for item in output_data:
            assert item.get("stage1") == "done"
            assert item.get("stage2") == "done"
            assert item.get("stage3") == "done"


class TestPipelineHelpers:
    """Pipeline 辅助方法测试."""

    def test_get_nested_value(self):
        """测试嵌套字段值获取."""
        recipe = SimpleRecipe(config={})
        pipeline = Pipeline(recipe=recipe)

        item = {"id": "1", "metadata": {"_uid": "uid-1", "nested": {"deep": "value"}}}

        assert pipeline._get_nested_value(item, "id") == "1"
        assert pipeline._get_nested_value(item, "metadata._uid") == "uid-1"
        assert pipeline._get_nested_value(item, "metadata.nested.deep") == "value"
        assert pipeline._get_nested_value(item, "nonexistent") is None
        assert pipeline._get_nested_value(item, "metadata.nonexistent") is None

    def test_get_stage_concurrency(self):
        """测试 Stage 并发度获取."""
        recipe = MultiStageRecipe(config={})

        # 默认并发度
        pipeline = Pipeline(recipe=recipe, concurrency=10)
        stage = pipeline._stages[0]
        assert pipeline._get_stage_concurrency(stage) == 10

        # 自定义并发度
        pipeline = Pipeline(
            recipe=recipe,
            concurrency=10,
            stage_concurrency={"AddFieldStage": 5},
        )
        assert pipeline._get_stage_concurrency(stage) == 5
