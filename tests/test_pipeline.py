"""Tests for Pipeline."""

import json
import os
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


class FailingStage(Stage):
    """会失败的 Stage, 用于测试错误处理."""
    
    def __init__(self, fail_ids: set = None):
        self.fail_ids = fail_ids or set()
    
    def process(self, batch):
        results = []
        for item in batch:
            if item.get("id") in self.fail_ids:
                results.append({**item, "_failed": True, "_error": "Intentional failure"})
            else:
                results.append(item)
        return results


class FilterStage(Stage):
    """过滤 Stage, 移除某些 item."""
    
    def __init__(self, keep_ids: set = None):
        self.keep_ids = keep_ids
    
    def process(self, batch):
        if self.keep_ids is None:
            return batch
        return [item for item in batch if item.get("id") in self.keep_ids]


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


class FailingRecipe(BaseRecipe):
    """包含失败 Stage 的 Recipe."""
    
    def __init__(self, config, fail_ids: set = None):
        super().__init__(config)
        self.fail_ids = fail_ids or set()
    
    def stages(self):
        return [
            FailingStage(self.fail_ids),
            PassThroughStage(),
        ]


# ============================================================
# Tests
# ============================================================

class TestPipelineInit:
    """Pipeline 初始化测试."""
    
    def test_basic_init(self):
        """基本初始化."""
        recipe = SimpleRecipe(config={})
        pipeline = Pipeline(recipe=recipe)
        
        assert pipeline.recipe == recipe
        assert pipeline.batch_size == 32
        assert pipeline.concurrency == 10
        assert pipeline.preserve_order is True
        assert pipeline.resume is True
    
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
    
    def test_resume_skip_processed(self, temp_files):
        """断点续传 - 跳过已处理的数据."""
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
        
        # 预先写入一些已处理的数据
        with open(output_path, "w") as f:
            f.write(json.dumps({"id": "1", "value": "a", "metadata": {"_uid": "1"}}) + "\n")
        
        # 运行 Pipeline (resume=True)
        recipe = SimpleRecipe(config={})
        pipeline = Pipeline(
            recipe=recipe,
            batch_size=10,
            concurrency=1,
            resume=True,
        )
        pipeline.run(str(input_path), str(output_path))
        
        # 验证输出 - 应该有 3 条(1 条已存在 + 2 条新处理)
        with open(output_path) as f:
            output_data = [json.loads(line) for line in f]
        
        assert len(output_data) == 3
    
    def test_failed_items_skipped(self, temp_files):
        """失败的 item 会被跳过."""
        input_path, output_path = temp_files
        
        # 写入测试数据
        test_data = [
            {"id": "1", "metadata": {"_uid": "1"}},
            {"id": "2", "metadata": {"_uid": "2"}},
            {"id": "3", "metadata": {"_uid": "3"}},
        ]
        with open(input_path, "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
        
        # 运行 Pipeline - id=2 会失败
        recipe = FailingRecipe(config={}, fail_ids={"2"})
        pipeline = Pipeline(
            recipe=recipe,
            batch_size=10,
            concurrency=1,
            resume=False,
        )
        pipeline.run(str(input_path), str(output_path))
        
        # 验证输出 - 失败的 item 应该被跳过(因为带 _failed 标记)
        with open(output_path) as f:
            output_data = [json.loads(line) for line in f]
        
        # 只有 2 条成功的
        successful = [item for item in output_data if not item.get("_failed")]
        assert len(successful) == 2


class TestPipelineHelpers:
    """Pipeline 辅助方法测试."""
    
    def test_get_nested_value(self):
        """测试嵌套字段值获取."""
        recipe = SimpleRecipe(config={})
        pipeline = Pipeline(recipe=recipe)
        
        item = {
            "id": "1",
            "metadata": {
                "_uid": "uid-1",
                "nested": {
                    "deep": "value"
                }
            }
        }
        
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

