"""Base classes: Stage and BaseRecipe."""

import asyncio
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Union

logger = logging.getLogger(__name__)


class Stage(ABC):
    """
    Stage base class: one processing step in the pipeline.

    Usage:
    1) implement process_item(item) -> dict
    2) or override process(batch) -> list[dict] (custom batch processing)

    Concurrency modes:
    - sync by default
    - @Stage.threaded_mode: process items in a batch with a thread pool
    - for async execution, override process() as async def and manage resources inside
    """

    _execution_mode = "sync"

    @classmethod
    def threaded_mode(cls, stage_class):
        """
        Threaded mode: the framework uses a thread pool to process items in a batch.

        The pool size is set in initialize() via self._thread_pool_size.
        """
        stage_class._execution_mode = "threaded"
        return stage_class

    def initialize(self):
        """Initialize resources (called once per worker in actor mode)."""
        pass

    def shutdown(self):
        """Release resources (called when a worker is destroyed in actor mode)."""
        pass

    def process(self, batch: list[dict]) -> Union[list[dict], "asyncio.coroutine"]:
        """
        Process a batch (default: call process_item with automatic error handling).

        Override this method to fully control batch processing (shared resources,
        batch inference, async execution).
        """
        mode = self._execution_mode
        if mode == "threaded":
            return self._default_threaded_process(batch)
        else:
            return self._default_sync_process(batch)

    def _default_sync_process(self, batch: list[dict]) -> list[dict]:
        """Sync mode: process items sequentially."""
        results = []
        for item in batch:
            # Only skip when _failed is explicitly True.
            if item.get("_failed") is True:
                results.append(item)
                continue

            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                logger.error(
                    f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed: {e}\n{error_trace}"
                )
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
        """Threaded mode: process items in a batch with a thread pool."""

        def safe_process_one(item):
            # Only skip when _failed is explicitly True.
            if item.get("_failed") is True:
                return item
            try:
                return self.process_item(item)
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                logger.error(
                    f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed: {e}\n{error_trace}"
                )
                return {
                    **item,
                    "_failed": True,
                    "_error": f"{type(self).__name__}: {e}",
                    "_traceback": error_trace,
                }

        # Read thread pool size from instance attribute.
        max_workers = getattr(self, "_thread_pool_size", 10)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(safe_process_one, batch))

    def process_item(self, item: dict) -> Union[dict, "asyncio.coroutine"]:
        """Process a single item (implemented by subclasses)."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement process_item() or override process()"
        )

    def is_async(self) -> bool:
        """Check whether process() is async."""
        if type(self).process != Stage.process:
            return asyncio.iscoroutinefunction(self.process)
        return False


class BaseRecipe(ABC):
    """Recipe base class: defines which stages compose the recipe."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def stages(self) -> list[Stage]:
        """Return stage list (in execution order)."""
        pass
