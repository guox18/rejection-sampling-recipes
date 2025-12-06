"""
vLLM Offline Sampler.

Uses Ray data parallelism with persistent actors for efficient multi-GPU inference.
Supports tensor parallelism (TP) and data parallelism (DP).
"""

import ray

from .base import BaseSampler


# Ray Actor must be defined at module level to avoid serialization issues
@ray.remote
class VLLMWorker:
    """Persistent vLLM worker that keeps model loaded."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        sampling_params: dict,
        extra_params: dict,
    ):
        import os

        from vllm import LLM

        # Get assigned GPU IDs from Ray
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print(f"[Worker] CUDA_VISIBLE_DEVICES={cuda_visible}")
        print(f"[Worker] Loading model with TP={tensor_parallel_size}...")

        # Load model ONCE when actor is created
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )
        self.sampling_params = sampling_params
        self.extra_params = extra_params
        self.tokenizer = self.llm.get_tokenizer()

        print("[Worker] Model loaded!")

    def sample(
        self,
        items: list[dict],
        n: int,
        drop_truncated: bool = True,
    ) -> list[tuple[str, list[str]]]:
        """Called multiple times WITHOUT reloading model."""
        from vllm import SamplingParams

        if not items:
            return []

        # Prepare prompts using tokenizer's chat template
        prompts = []
        for item in items:
            prompt = self.tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        # Generate with extra params
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=SamplingParams(
                n=n,
                temperature=self.sampling_params["temperature"],
                max_tokens=self.sampling_params["max_tokens"],
                top_p=self.sampling_params["top_p"],
                **self.extra_params,
            ),
        )

        # Collect results
        results = []
        eos_token_id = self.tokenizer.eos_token_id
        for item, output in zip(items, outputs, strict=False):
            responses = []
            for o in output.outputs:
                # Check truncation by finish_reason or missing EOS token
                is_truncated = o.finish_reason == "length" or (
                    len(o.token_ids) > 0 and o.token_ids[-1] != eos_token_id
                )

                if drop_truncated and is_truncated:
                    continue

                responses.append(o.text)
            results.append((item["id"], responses))

        return results


class VLLMSampler(BaseSampler):
    """vLLM offline sampler using Ray data parallelism."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        data_parallel_size: int | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.9,
        drop_truncated: bool = True,
        extra_params: dict | None = None,
        verbose: bool = False,
    ):
        """
        Initialize vLLM sampler.

        Args:
            model_path: Path to the model
            tensor_parallel_size: Number of GPUs for tensor parallelism per worker
            data_parallel_size: Number of data parallel workers (None = auto: total_gpus / tp)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            gpu_memory_utilization: GPU memory utilization ratio
            drop_truncated: Whether to drop truncated responses
            extra_params: Extra parameters for vLLM SamplingParams
            verbose: Enable verbose logging
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.data_parallel_size = data_parallel_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.drop_truncated = drop_truncated
        self.extra_params = dict(extra_params) if extra_params else {}
        self.verbose = verbose

        self.workers = None  # Ray actors (persistent)

    async def initialize(self) -> None:
        """Create Ray actors and load models (ONCE)."""
        print("Initializing vLLM sampler...")
        print(f"  Model path: {self.model_path}")
        print(f"  Tensor parallel size: {self.tensor_parallel_size}")

        if not ray.is_initialized():
            # Disable automatic working_dir packaging to avoid creating new venvs
            # that miss optional dependencies (ray, vllm)
            ray.init(ignore_reinit_error=True, runtime_env={"working_dir": None})

        # Calculate data parallel size
        import torch

        total_gpus = torch.cuda.device_count()

        if self.data_parallel_size is not None:
            dp_size = self.data_parallel_size
        else:
            # Auto: use all available GPUs
            dp_size = total_gpus // self.tensor_parallel_size

        required_gpus = dp_size * self.tensor_parallel_size
        if required_gpus > total_gpus:
            raise RuntimeError(
                f"Not enough GPUs: need {required_gpus} (TP={self.tensor_parallel_size} x DP={dp_size}), "
                f"but only {total_gpus} available"
            )

        print(f"  Data parallel size: {dp_size}")
        print(f"  Total GPUs used: {required_gpus}")
        if self.extra_params:
            print(f"  Extra params: {self.extra_params}")

        # Create actors with GPU resources
        tp_size = self.tensor_parallel_size
        gpu_mem_util = self.gpu_memory_utilization

        print(f"Creating {dp_size} workers (each with {tp_size} GPUs)...")

        # Create worker actors with proper GPU allocation
        self.workers = []
        for _ in range(dp_size):
            # Create actor with specific GPU count
            worker = VLLMWorker.options(num_gpus=tp_size).remote(
                self.model_path,
                tp_size,
                gpu_mem_util,
                {
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                },
                self.extra_params,
            )
            self.workers.append(worker)

        # Wait for all workers to finish loading (blocking)
        print("Waiting for all workers to load models...")
        test_item = {
            "id": "test",
            "messages": [{"role": "user", "content": "hi"}],
        }
        test_futures = [
            worker.sample.remote([test_item], 1, self.drop_truncated) for worker in self.workers
        ]
        ray.get(test_futures)  # Block until all loaded

        print(f"✓ All {dp_size} workers ready! (TP={tp_size}, DP={dp_size})")

    async def sample_batch(
        self,
        items: list[dict],
        n: int,
    ) -> dict[str, list[str]]:
        """Distribute work to existing actors (no reloading!)."""
        import numpy as np

        if not self.workers:
            raise RuntimeError("Sampler not initialized. Call initialize() first.")

        if not items:
            return {}

        # Split items across workers
        chunks = np.array_split(items, len(self.workers))

        # Submit tasks to actors
        futures = [
            worker.sample.remote(list(chunk), n, self.drop_truncated)
            for worker, chunk in zip(self.workers, chunks, strict=False)
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
        """Kill actors and shutdown Ray."""
        if self.workers:
            for worker in self.workers:
                ray.kill(worker)
            print(f"Killed {len(self.workers)} workers")
            self.workers = None

        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown")
