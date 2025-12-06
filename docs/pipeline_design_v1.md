# Pipeline 设计文档 v1

本文档描述 Sampler 和 Pipeline 的对称设计，支持 OpenAI API (asyncio) 和 vLLM offline (Ray) 两种推理后端。

## 设计目标

1. **对称性**：两种 sampler 接口完全一致，Pipeline 无需关心底层实现
2. **职责分离**：Sampler 只负责采样，Pipeline 控制多轮逻辑
3. **资源管理**：显式的初始化和清理，避免资源泄漏
4. **提前验证**：初始化时检查配置，尽早发现错误

## 两种推理方式的差异

| 特性 | OpenAI API | vLLM offline |
|-----|-----------|--------------|
| 并发方式 | asyncio 并发 HTTP 请求 | Ray 数据并行 (多 GPU) |
| 状态 | 无状态 | 有状态（模型需加载） |
| 初始化 | ping API 检查连通性 | 创建 Ray Actors，加载模型 |
| 资源管理 | 关闭 HTTP client | kill Ray Actors |
| GPU 配置 | N/A | `num_gpus`（可配置或自动检测） |
| extra_params | API 请求参数 | vLLM SamplingParams 参数 |

## 核心原则：职责分离

- **Sampler**：只负责"给一批 items 生成 responses"
- **Pipeline**：负责多轮迭代、验证、早停等流程控制

这样 Pipeline 的多轮逻辑只需要写一次，对两种 sampler 都适用。

---

## Sampler 接口设计

### BaseSampler

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseSampler(ABC):
    """Base sampler interface"""

    async def initialize(self) -> None:
        """
        Initialize sampler and validate configuration.
        - OpenAI: ping API to check connectivity
        - vLLM: create Ray actors and load models

        Raises:
            RuntimeError: if initialization fails
        """
        pass

    @abstractmethod
    async def sample_batch(
        self,
        items: List[dict],
        n: int
    ) -> Dict[str, List[str]]:
        """
        Sample n responses for each item.

        Args:
            items: List of items, each has "id" and "messages"
                   [{"id": "001", "messages": [{"role": "user", "content": "..."}]}, ...]
            n: Number of responses to generate per item

        Returns:
            Dict mapping item_id to list of responses
            {"001": ["response1", "response2", ...], ...}
        """
        pass

    async def shutdown(self) -> None:
        """
        Clean up resources.
        - OpenAI: close client
        - vLLM: kill Ray actors and shutdown Ray
        """
        pass
```

### OpenAISampler 实现

```python
import asyncio
import os
from openai import AsyncOpenAI

class OpenAISampler(BaseSampler):
    """OpenAI API sampler using asyncio concurrency"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = AsyncOpenAI(
            base_url=cfg.base_url or "https://api.openai.com/v1",
            api_key=cfg.api_key or os.getenv("OPENAI_API_KEY"),
            timeout=cfg.timeout
        )
        # Extra params for specific models (e.g., reasoning_effort for OSS models)
        self.extra_params = dict(cfg.extra_params) if cfg.extra_params else {}

    async def initialize(self) -> None:
        """Ping API to validate configuration"""
        print("Initializing OpenAI sampler...")
        print(f"  Base URL: {self.cfg.base_url}")
        print(f"  Model: {self.cfg.model}")
        if self.extra_params:
            print(f"  Extra params: {self.extra_params}")

        try:
            # Simple health check - minimal request
            response = await self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=5,
                timeout=10,
                **self.extra_params  # Pass extra params
            )
            print(f"✓ API connection successful (model: {self.cfg.model})")
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            raise RuntimeError(f"Failed to initialize OpenAI sampler: {e}")

    async def sample_batch(
        self,
        items: List[dict],
        n: int
    ) -> Dict[str, List[str]]:
        """Async concurrent sampling with rate limiting"""

        async def sample_one(item):
            """Sample one item"""
            try:
                response = await self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=item["messages"],
                    n=n,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    top_p=self.cfg.top_p,
                    timeout=self.cfg.timeout,
                    **self.extra_params  # Pass extra params (e.g., reasoning_effort)
                )
                responses = [choice.message.content for choice in response.choices]
                return item["id"], responses
            except Exception as e:
                print(f"Error sampling {item['id']}: {e}")
                return item["id"], []

        # Rate limiting with semaphore
        semaphore = asyncio.Semaphore(self.cfg.concurrent_requests)

        async def sample_with_limit(item):
            async with semaphore:
                return await sample_one(item)

        # Launch all tasks concurrently
        tasks = [sample_with_limit(item) for item in items]
        results = await asyncio.gather(*tasks)

        return dict(results)

    async def shutdown(self) -> None:
        """Close HTTP client"""
        await self.client.close()
        print("OpenAI sampler shutdown")
```

### VLLMSampler 实现

使用 **Ray Actor**（而非 `@ray.remote` 函数）来保持模型持久化，避免每轮重复加载。

```python
import numpy as np

class VLLMSampler(BaseSampler):
    """vLLM offline sampler using Ray data parallelism"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.workers = None  # Ray actors (persistent)
        # Extra params for vLLM SamplingParams
        self.extra_params = dict(cfg.extra_params) if cfg.extra_params else {}

    async def initialize(self) -> None:
        """Create Ray actors and load models (ONCE)"""
        import ray

        print("Initializing vLLM sampler...")
        print(f"  Model path: {self.cfg.model_path}")

        if not ray.is_initialized():
            ray.init()

        # Determine number of GPUs to use
        num_gpus = self._get_num_gpus()
        print(f"  Using {num_gpus} GPUs")
        if self.extra_params:
            print(f"  Extra params: {self.extra_params}")

        # Define Ray Actor (persistent worker with loaded model)
        @ray.remote(num_gpus=1)
        class VLLMWorker:
            def __init__(self, model_path, sampling_params, extra_params):
                from vllm import LLM
                import torch

                gpu_id = torch.cuda.current_device()
                print(f"[GPU {gpu_id}] Loading model from {model_path}...")

                # Load model ONCE when actor is created
                self.llm = LLM(
                    model=model_path,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.9
                )
                self.sampling_params = sampling_params
                self.extra_params = extra_params

                print(f"[GPU {gpu_id}] Model loaded!")

            def sample(self, items, n):
                """Called multiple times WITHOUT reloading model"""
                from vllm import SamplingParams

                # Prepare prompts
                prompts = [
                    self._format_messages(item["messages"])
                    for item in items
                ]

                # Generate with extra params
                outputs = self.llm.generate(
                    prompts=prompts,
                    sampling_params=SamplingParams(
                        n=n,
                        temperature=self.sampling_params["temperature"],
                        max_tokens=self.sampling_params["max_tokens"],
                        top_p=self.sampling_params["top_p"],
                        **self.extra_params  # Pass extra params
                    )
                )

                # Collect results
                results = []
                for item, output in zip(items, outputs):
                    responses = [o.text for o in output.outputs]
                    results.append((item["id"], responses))

                return results

            def _format_messages(self, messages):
                """Convert OpenAI messages to prompt string"""
                # TODO: Use proper chat template from tokenizer
                return "\n".join([
                    f"{m['role']}: {m['content']}"
                    for m in messages
                ])

        # Create actors (one per GPU)
        self.workers = [
            VLLMWorker.remote(
                self.cfg.model_path,
                {
                    "temperature": self.cfg.temperature,
                    "max_tokens": self.cfg.max_tokens,
                    "top_p": self.cfg.top_p
                },
                self.extra_params
            )
            for _ in range(num_gpus)
        ]

        # Wait for all workers to finish loading (blocking)
        print("Waiting for all workers to load models...")
        test_item = {
            "id": "test",
            "messages": [{"role": "user", "content": "hi"}]
        }
        test_futures = [
            worker.sample.remote([test_item], 1)
            for worker in self.workers
        ]
        ray.get(test_futures)  # Block until all loaded

        print(f"✓ All {num_gpus} workers ready!")

    async def sample_batch(
        self,
        items: List[dict],
        n: int
    ) -> Dict[str, List[str]]:
        """Distribute work to existing actors (no reloading!)"""
        import ray

        # Split items across workers
        chunks = np.array_split(items, len(self.workers))

        # Submit tasks to actors
        futures = [
            worker.sample.remote(chunk.tolist(), n)
            for worker, chunk in zip(self.workers, chunks)
            if len(chunk) > 0
        ]

        # Gather results
        results_list = ray.get(futures)

        # Flatten to dict
        results_dict = {}
        for results in results_list:
            for item_id, responses in results:
                results_dict[item_id] = responses

        return results_dict

    async def shutdown(self) -> None:
        """Kill actors and shutdown Ray"""
        import ray

        if self.workers:
            for worker in self.workers:
                ray.kill(worker)
            print(f"Killed {len(self.workers)} workers")

        ray.shutdown()
        print("Ray shutdown")

    def _get_num_gpus(self) -> int:
        """
        Get number of GPUs to use.
        Priority: cfg.num_gpus > auto detect
        """
        if self.cfg.num_gpus is not None:
            return self.cfg.num_gpus

        # Auto detect
        import torch
        return torch.cuda.device_count()
```

---

## Pipeline 设计

Pipeline 统一控制多轮采样逻辑，对两种 sampler 完全一致。

```python
class Pipeline:
    def __init__(self, cfg, sampler, verifier, formatters):
        self.cfg = cfg
        self.sampler = sampler
        self.verifier = verifier
        self.formatters = formatters

    async def run(self):
        """Main entry point"""
        # Initialize sampler (symmetric for both!)
        await self.sampler.initialize()

        try:
            # Load data
            items = self.load_data()

            # Process in shards
            for shard_idx, shard in enumerate(self.shard_items(items)):
                print(f"Processing shard {shard_idx}...")
                shard_results = await self._rollout_shard(shard)
                self.save_shard(shard_results, shard_idx)

        finally:
            # Cleanup (symmetric for both!)
            await self.sampler.shutdown()

    async def _rollout_shard(self, items: List[dict]) -> List[dict]:
        """
        Multi-round sampling for a shard.
        Works for BOTH OpenAI API and vLLM!
        """
        # Initialize rollouts for each item
        rollouts_map = {item["id"]: [] for item in items}
        remaining_items = items.copy()

        for step in range(self.cfg.sampling.max_steps):
            if not remaining_items:
                break

            print(f"  Step {step + 1}/{self.cfg.sampling.max_steps}: "
                  f"sampling {len(remaining_items)} items")

            # Step 1: Sample batch (works for both API and vLLM!)
            responses_map = await self.sampler.sample_batch(
                remaining_items,
                n=self.cfg.sampling.step_size
            )

            # Step 2: Verify and collect
            for item in remaining_items:
                item_id = item["id"]
                responses = responses_map.get(item_id, [])

                for resp in responses:
                    # Filter truncated
                    if self._is_truncated(resp):
                        continue

                    # Verify
                    score = self.verifier.verify(resp, item["metadata"])
                    rollouts_map[item_id].append({
                        "response": resp,
                        "score": score
                    })

            # Step 3: Remove satisfied items (early stopping)
            if self.cfg.sampling.early_stop:
                remaining_items = [
                    item for item in remaining_items
                    if not self._is_satisfied(rollouts_map[item["id"]])
                ]
            else:
                # Just check max_rollouts
                remaining_items = [
                    item for item in remaining_items
                    if len(rollouts_map[item["id"]]) < self.cfg.sampling.max_rollouts
                ]

        # Build final results
        results = []
        for item in items:
            results.append({
                **item,
                "rollouts": rollouts_map[item["id"]]
            })

        return results

    def _is_satisfied(self, rollouts: List[dict]) -> bool:
        """Check if all formatters are satisfied (for early stopping)"""
        for formatter in self.formatters:
            if not formatter.is_satisfied(rollouts):
                return False
        return True

    def _is_truncated(self, response: str) -> bool:
        """Check if response is truncated"""
        # TODO: Implement based on finish_reason or eos_token
        return False

    def load_data(self) -> List[dict]:
        """Load input data"""
        # TODO: Implement
        pass

    def shard_items(self, items: List[dict]):
        """Split items into shards"""
        shard_size = self.cfg.shard.size
        for i in range(0, len(items), shard_size):
            yield items[i:i + shard_size]

    def save_shard(self, results: List[dict], shard_idx: int):
        """Save shard results"""
        # TODO: Implement
        pass
```

---

## 执行流程

```
Pipeline.run()
    │
    ├── sampler.initialize()
    │       │
    │       ├── [OpenAI] ping API, validate config
    │       │
    │       └── [vLLM] create Ray Actors, load models to GPUs
    │                  (model loading happens ONCE here)
    │
    ├── for shard in shards:
    │       │
    │       └── _rollout_shard(shard)
    │               │
    │               └── for step in range(max_steps):
    │                       │
    │                       ├── sampler.sample_batch(items, n)
    │                       │       │
    │                       │       ├── [OpenAI] asyncio concurrent HTTP requests
    │                       │       │
    │                       │       └── [vLLM] distribute to Ray Actors
    │                       │                 (use already-loaded models)
    │                       │
    │                       ├── verifier.verify() for each response
    │                       │
    │                       └── check early_stop, remove satisfied items
    │
    └── sampler.shutdown()
            │
            ├── [OpenAI] close HTTP client
            │
            └── [vLLM] kill Ray Actors, ray.shutdown()
```

---

## 数据流

```
Dataset (e.g., 1700 samples)
    │
    ├── Shard 0 (1000 samples)
    │       │
    │       └── Step 1: sample_batch() → verify() → early_stop check
    │           Step 2: sample_batch() → verify() → early_stop check
    │           ...
    │           Step N: done
    │       │
    │       └── Save to rollout/shard_0000.jsonl
    │
    ├── Shard 1 (700 samples)
    │       │
    │       └── ... same flow ...
    │       │
    │       └── Save to rollout/shard_0001.jsonl
    │
    └── Complete
```

---

## vLLM Ray Actor vs Ray Function

### ❌ 错误做法：Ray Function（每轮重新加载模型）

```python
# 每次调用都创建新 task，重新加载模型！
@ray.remote(num_gpus=1)
def sample_worker(items, model_path, n):
    llm = LLM(model=model_path)  # 每次都加载
    return llm.generate(...)

# Pipeline 每轮调用
for step in range(max_steps):
    futures = [sample_worker.remote(...) for ...]  # 创建新 task
    results = ray.get(futures)  # 等待，模型重新加载
```

**问题**：
- 每轮都重新加载模型
- 模型加载时间 >> 推理时间
- 完全不可接受

### ✅ 正确做法：Ray Actor（模型持久化）

```python
# Actor 初始化时加载模型
@ray.remote(num_gpus=1)
class VLLMWorker:
    def __init__(self, model_path):
        self.llm = LLM(model=model_path)  # 只加载一次

    def sample(self, items, n):
        return self.llm.generate(...)  # 复用已加载的模型

# 初始化时创建 actors
workers = [VLLMWorker.remote(model_path) for _ in range(num_gpus)]

# Pipeline 每轮调用（模型已加载）
for step in range(max_steps):
    futures = [w.sample.remote(...) for w in workers]  # 调用已有 actor
    results = ray.get(futures)  # 快速，无需重新加载
```

**优势**：
- 模型只加载一次
- 每轮调用复用已加载的模型
- 高效
