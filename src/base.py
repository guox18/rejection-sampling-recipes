"""基类定义: Stage 和 BaseRecipe."""

import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Union


class Stage(ABC):
    """
    Stage 基类: 流水线中的一个处理阶段.

    用法:
    1. 实现 process_item(item) -> dict  (推荐)
    2. 覆盖 process(batch) -> list[dict]  (高级用法, 完全自定义批处理)

    装饰器:
    - @Stage.async_mode: 异步并发执行 batch 内的 item
    - @Stage.threaded_mode(n): 多线程并发执行 batch 内的 item
    """

    _execution_mode = "sync"

    @classmethod
    def async_mode(cls, stage_class):
        """异步模式: 框架用 asyncio.gather 并发执行 batch 内的 item."""
        stage_class._execution_mode = "async"
        return stage_class

    @classmethod
    def threaded_mode(cls, stage_class):
        """
        多线程模式: 框架用线程池并发执行 batch 内的 item.

        线程池大小通过在 initialize() 中设置 self._thread_pool_size 指定.
        如果不设置, 默认使用 10 个线程.
        """
        stage_class._execution_mode = "threaded"
        return stage_class

    def initialize(self):
        """初始化资源 (Actor 模式下每个 worker 调用一次)."""
        pass

    def shutdown(self):
        """释放资源 (Actor 模式下 worker 销毁时调用)."""
        pass

    def process(self, batch: list[dict]) -> Union[list[dict], "asyncio.coroutine"]:
        """
        处理一个 batch (默认实现: 自动调用 process_item, 自动异常处理).

        可覆盖此方法以完全控制批处理逻辑 (如共享资源、batch inference).
        """
        mode = self._execution_mode
        if mode == "async":
            return self._default_async_process(batch)
        elif mode == "threaded":
            return self._default_threaded_process(batch)
        else:
            return self._default_sync_process(batch)

    def _default_sync_process(self, batch: list[dict]) -> list[dict]:
        """同步模式: 顺序处理每个 item."""
        results = []
        for item in batch:
            if item.get("_failed"):
                results.append(item)
                continue

            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                print(f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed:")
                print(f"  Error: {e}")
                print(f"  Traceback:\n{error_trace}")
                results.append(
                    {
                        **item,
                        "_failed": True,
                        "_error": f"{type(self).__name__}: {e}",
                        "_traceback": error_trace,
                    }
                )
        return results

    def _default_threaded_process(self, batch: list[dict]) -> list[dict]:
        """多线程模式: 用线程池并发处理 batch 内的 item."""

        def safe_process_one(item):
            if item.get("_failed"):
                return item
            try:
                return self.process_item(item)
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                print(f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed:")
                print(f"  Error: {e}")
                print(f"  Traceback:\n{error_trace}")
                return {
                    **item,
                    "_failed": True,
                    "_error": f"{type(self).__name__}: {e}",
                    "_traceback": error_trace,
                }

        # 从实例属性读取线程池大小, 如果未设置则使用默认值 10
        max_workers = getattr(self, "_thread_pool_size", 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(safe_process_one, batch))

    async def _default_async_process(self, batch: list[dict]) -> list[dict]:
        """异步模式: 用 asyncio.gather 并发处理 batch 内的 item."""
        is_async = asyncio.iscoroutinefunction(self.process_item)

        async def safe_process_one(item):
            if item.get("_failed"):
                return item
            try:
                if is_async:
                    result = await self.process_item(item)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, self.process_item, item)
                return result
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                print(f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed:")
                print(f"  Error: {e}")
                print(f"  Traceback:\n{error_trace}")
                return {
                    **item,
                    "_failed": True,
                    "_error": f"{type(self).__name__}: {e}",
                    "_traceback": error_trace,
                }

        return await asyncio.gather(*[safe_process_one(item) for item in batch])

    def process_item(self, item: dict) -> Union[dict, "asyncio.coroutine"]:
        """处理单个 item (子类实现). 框架自动异常处理, 无需 try-catch."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement process_item() or override process()"
        )

    def is_async(self) -> bool:
        """检查是否需要异步执行."""
        if type(self).process != Stage.process:
            return asyncio.iscoroutinefunction(self.process)
        return self._execution_mode == "async"


class BaseRecipe(ABC):
    """Recipe 基类: 定义由哪些 Stage 组成."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def stages(self) -> list[Stage]:
        """返回 Stage 列表 (按执行顺序)."""
        pass
