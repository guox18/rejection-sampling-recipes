"""Tests for base classes (Stage, BaseRecipe)."""

import asyncio

import pytest

from src.base import BaseRecipe, Stage


class TestStage:
    """Tests for Stage base class."""

    def test_stage_is_abstract(self):
        """Stage 是抽象类, 不能直接实例化."""
        with pytest.raises(TypeError):
            Stage()

    def test_sync_stage(self):
        """同步 Stage 测试."""

        class SyncStage(Stage):
            def process(self, batch):
                return [{"processed": True, **item} for item in batch]

        stage = SyncStage()
        assert not stage.is_async()

        result = stage.process([{"id": 1}, {"id": 2}])
        assert len(result) == 2
        assert all(item["processed"] for item in result)

    def test_async_stage(self):
        """异步 Stage 测试."""

        class AsyncStage(Stage):
            async def process(self, batch):
                await asyncio.sleep(0.01)
                return [{"processed": True, **item} for item in batch]

        stage = AsyncStage()
        assert stage.is_async()

        result = asyncio.run(stage.process([{"id": 1}, {"id": 2}]))
        assert len(result) == 2
        assert all(item["processed"] for item in result)

    def test_initialize_and_shutdown(self):
        """测试 initialize 和 shutdown 方法."""

        class StatefulStage(Stage):
            def __init__(self):
                self.initialized = False
                self.shutdown_called = False

            def initialize(self):
                self.initialized = True

            def shutdown(self):
                self.shutdown_called = True

            def process(self, batch):
                return batch

        stage = StatefulStage()
        assert not stage.initialized
        assert not stage.shutdown_called

        stage.initialize()
        assert stage.initialized

        stage.shutdown()
        assert stage.shutdown_called


class TestBaseRecipe:
    """Tests for BaseRecipe base class."""

    def test_recipe_is_abstract(self):
        """BaseRecipe 是抽象类, 不能直接实例化."""
        with pytest.raises(TypeError):
            BaseRecipe(config={})

    def test_concrete_recipe(self):
        """具体 Recipe 测试."""

        class DummyStage(Stage):
            def process(self, batch):
                return batch

        class DummyRecipe(BaseRecipe):
            def stages(self):
                return [DummyStage(), DummyStage()]

        recipe = DummyRecipe(config={"key": "value"})
        assert recipe.config == {"key": "value"}

        stages = recipe.stages()
        assert len(stages) == 2
        assert all(isinstance(s, Stage) for s in stages)
