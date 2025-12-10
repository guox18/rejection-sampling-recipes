"""
Main Pipeline for Rejection Sampling.

Orchestrates the sampling, verification, and formatting workflow.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .formatter import BaseFormatter, get_formatter
from .preprocessor import DataPreprocessor
from .sampler import BaseSampler, get_sampler
from .state import StateManager
from .verifier import BaseVerifier, get_verifier

# Suppress verbose httpx logging by default
logging.getLogger("httpx").setLevel(logging.WARNING)


class Pipeline:
    """Main rejection sampling pipeline."""

    def __init__(self, cfg):
        """
        Initialize pipeline from configuration.

        Args:
            cfg: OmegaConf configuration object
        """
        self.cfg = cfg

        # Time tracking
        self._start_time = time.time()
        self._sampling_time = 0.0
        self._verify_time = 0.0

        # Setup working directory
        self.work_dir = self._setup_work_dir()
        print(f"Working directory: {self.work_dir}")

        # Initialize components
        self.verbose = cfg.get("verbose", False)
        self.state = StateManager(self.work_dir)
        self.sampler: BaseSampler = get_sampler(cfg.sampler, verbose=self.verbose)
        self.verifier: BaseVerifier = self._create_verifier()
        self.formatters: list[BaseFormatter] = self._create_formatters()

        # Save config
        self._save_config()

    def _setup_work_dir(self) -> Path:
        """Setup and return working directory."""
        if self.cfg.work_dir:
            work_dir = Path(self.cfg.work_dir)
        else:
            # Auto-generate timestamp path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            work_dir = Path("output") / timestamp

        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _create_verifier(self) -> BaseVerifier:
        """Create verifier based on config."""
        verifier_cfg = self.cfg.verifier

        # Collect all optional kwargs from config
        extra_kwargs = {}
        if verifier_cfg.get("model"):
            extra_kwargs["model"] = verifier_cfg.model
        if verifier_cfg.get("base_url"):
            extra_kwargs["base_url"] = verifier_cfg.base_url
        if verifier_cfg.get("api_key"):
            extra_kwargs["api_key"] = verifier_cfg.api_key
        if verifier_cfg.get("temperature") is not None:
            extra_kwargs["temperature"] = verifier_cfg.temperature
        if verifier_cfg.get("max_tokens"):
            extra_kwargs["max_tokens"] = verifier_cfg.max_tokens

        # Verifiers should accept **kwargs and ignore what they don't need
        return get_verifier(verifier_cfg.type, **extra_kwargs)

    def _create_formatters(self) -> list[BaseFormatter]:
        """Create formatters based on config."""
        formatters = []
        for fmt_cfg in self.cfg.formatter:
            # Convert to dict and pass all params (formatters should accept **kwargs)
            fmt_dict = dict(fmt_cfg)
            fmt_type = fmt_dict.pop("type")
            # Ensure defaults for common params
            fmt_dict.setdefault("pass_threshold", 1.0)
            fmt_dict.setdefault("fail_threshold", 0.0)
            formatter = get_formatter(fmt_type, **fmt_dict)
            formatters.append(formatter)
        return formatters

    def _save_config(self) -> None:
        """Save config to working directory."""
        from omegaconf import OmegaConf

        config_path = self.work_dir / "config.yaml"
        if not config_path.exists():
            with open(config_path, "w") as f:
                f.write(OmegaConf.to_yaml(self.cfg))
            print(f"Saved config to {config_path}")

    async def run(self) -> None:
        """Main entry point - run the pipeline."""
        print("=" * 60)
        print("Rejection Sampling Pipeline")
        print("=" * 60)

        # Initialize sampler
        await self.sampler.initialize()

        try:
            # Load/preprocess data
            items = self._load_data()
            self.state.set_total_items(len(items))

            # Process in shards
            shards = list(self._shard_items(items))
            print(f"Processing {len(items)} items in {len(shards)} shards")

            for shard_idx, shard in enumerate(shards):
                # Skip completed shards
                if self.state.is_shard_completed(shard_idx):
                    print(f"Skipping completed shard {shard_idx}")
                    continue

                print(f"\n--- Shard {shard_idx}/{len(shards) - 1} " f"({len(shard)} items) ---")

                self.state.mark_shard_started(shard_idx)

                # Process shard
                shard_results = await self._rollout_shard(shard)

                # Save shard results
                self._save_shard(shard_results, shard_idx)

                self.state.mark_shard_completed(shard_idx, len(shard))
                print(f"Progress: {self.state.get_progress()}")

            # Format all results into training data
            self._format_training_data()

            # Generate summary statistics
            self._generate_summary()

            self.state.mark_completed()
            print("\n" + "=" * 60)
            print("Pipeline completed successfully!")
            print("=" * 60)

        except Exception as e:
            self.state.mark_failed(str(e))
            raise

        finally:
            await self.sampler.shutdown()

    def _load_data(self) -> list[dict]:
        """Load and preprocess input data."""
        preprocessor = DataPreprocessor(
            input_path=self.cfg.data.input_path,
            output_path=self.work_dir / "data" / "input.jsonl",
            transform=self.cfg.data.preprocess.transform,
        )
        return preprocessor.process()

    def _shard_items(self, items: list[dict]):
        """Split items into shards."""
        shard_size = self.cfg.shard.size
        for i in range(0, len(items), shard_size):
            yield items[i : i + shard_size]

    async def _rollout_shard(self, items: list[dict]) -> list[dict]:
        """
        Multi-round sampling for a shard.

        Works for BOTH OpenAI API and vLLM!

        Sampling strategy:
        - max_rollouts: Target number of valid rollouts per prompt (soft limit)
        - step_size: Batch size for each sampling round
        - max_steps: Maximum number of sampling rounds (hard limit)
        - Total sampling budget: max_steps × step_size attempts

        The hard limit (max_steps × step_size) should be ≥ max_rollouts to account
        for truncated responses that are dropped. For example:
        - max_rollouts=16, step_size=4, max_steps=4: 16 attempts (1× buffer)
        - max_rollouts=16, step_size=4, max_steps=8: 32 attempts (2× buffer)
        """
        # Initialize rollouts for each item
        rollouts_map: dict[str, list[dict]] = {item["id"]: [] for item in items}
        remaining_items = items.copy()

        max_steps = self.cfg.sampling.max_steps
        step_size = self.cfg.sampling.step_size
        max_rollouts = self.cfg.sampling.max_rollouts
        early_stop = self.cfg.sampling.early_stop

        for step in range(max_steps):
            if not remaining_items:
                break

            print(
                f"\n  ▶ Step {step + 1}/{max_steps}: "
                f"{len(remaining_items)} items × {step_size} samples"
            )

            # Step 1: Sample batch
            sample_start = time.time()
            responses_map = await self.sampler.sample_batch(
                remaining_items,
                n=step_size,
            )
            sample_elapsed = time.time() - sample_start
            self._sampling_time += sample_elapsed

            # Step 2: Verify and collect (with progress bar)
            # Flatten all (item, response) pairs for verification
            verify_tasks = []
            for item in remaining_items:
                item_id = item["id"]
                responses = responses_map.get(item_id, [])
                for resp in responses:
                    verify_tasks.append((item_id, item["metadata"], resp))

            # Verify with progress bar (less verbose)
            verify_start = time.time()
            for item_id, metadata, resp in tqdm(
                verify_tasks,
                desc=f"    Step {step + 1} verifying",
                leave=False,
                disable=len(verify_tasks) < 10,  # Hide for small batches
                mininterval=1.0,  # Update at most once per second
            ):
                score = self.verifier.verify(resp, metadata)
                rollouts_map[item_id].append(
                    {
                        "response": resp,
                        "score": score,
                    }
                )
            verify_elapsed = time.time() - verify_start
            self._verify_time += verify_elapsed

            # Step 3: Remove satisfied items (early stopping)
            prev_remaining = len(remaining_items)
            if early_stop:
                remaining_items = [
                    item
                    for item in remaining_items
                    if not self._is_satisfied(rollouts_map[item["id"]])
                ]
            else:
                # Just check max_rollouts
                remaining_items = [
                    item for item in remaining_items if len(rollouts_map[item["id"]]) < max_rollouts
                ]

            # Print step summary
            satisfied = prev_remaining - len(remaining_items)
            print(
                f"    ✓ Step {step + 1} done: "
                f"satisfied={satisfied}, remaining={len(remaining_items)} "
                f"(sample: {sample_elapsed:.1f}s, verify: {verify_elapsed:.1f}s)"
            )

        # Build final results
        results = []
        for item in items:
            results.append(
                {
                    **item,
                    "rollouts": rollouts_map[item["id"]],
                }
            )

        return results

    def _is_satisfied(self, rollouts: list[dict]) -> bool:
        """Check if all formatters are satisfied (for early stopping)."""
        for formatter in self.formatters:
            if not formatter.is_satisfied(rollouts):
                return False
        return True

    def _save_shard(self, results: list[dict], shard_idx: int) -> None:
        """Save shard results to file."""
        rollout_dir = self.work_dir / "rollout"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        shard_path = rollout_dir / f"shard_{shard_idx:04d}.jsonl"
        with open(shard_path, "w") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"  Saved to {shard_path}")

    def _format_training_data(self) -> None:
        """Format all rollout results into training data."""
        print("\nFormatting training data...")

        train_dir = self.work_dir / "train"
        train_dir.mkdir(parents=True, exist_ok=True)

        # Collect all rollout results
        rollout_dir = self.work_dir / "rollout"
        all_results = []

        for shard_path in sorted(rollout_dir.glob("shard_*.jsonl")):
            with open(shard_path) as f:
                for line in f:
                    all_results.append(json.loads(line))

        # Format with each formatter
        for formatter in self.formatters:
            formatter_type = type(formatter).__name__.replace("Formatter", "").lower()
            output_path = train_dir / f"{formatter_type}.jsonl"

            count = 0
            with open(output_path, "w") as f:
                for item in all_results:
                    formatted = formatter.format(item, item.get("rollouts", []))
                    for example in formatted:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        count += 1

            print(f"  {formatter_type}: {count} examples → {output_path}")

    def _generate_summary(self) -> None:
        """Generate summary statistics."""
        print("\nGenerating summary...")

        summary_dir = self.work_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Collect stats from rollouts
        rollout_dir = self.work_dir / "rollout"
        total_items = 0
        total_rollouts = 0
        total_passed = 0
        pass_rates = []

        for shard_path in sorted(rollout_dir.glob("shard_*.jsonl")):
            with open(shard_path) as f:
                for line in f:
                    item = json.loads(line)
                    rollouts = item.get("rollouts", [])
                    total_items += 1
                    total_rollouts += len(rollouts)

                    # Count passed rollouts
                    passed = sum(1 for r in rollouts if r.get("score", 0) >= 1.0)
                    total_passed += passed

                    if rollouts:
                        pass_rates.append(passed / len(rollouts))

        # Calculate timing
        total_time = time.time() - self._start_time
        rollout_time = self._sampling_time + self._verify_time

        # Get truncation statistics from sampler
        total_generated = self.sampler.total_generated
        truncated_count = self.sampler.truncated_count
        truncation_rate = truncated_count / total_generated if total_generated > 0 else 0

        # Calculate statistics
        stats = {
            "total_items": total_items,
            "total_rollouts": total_rollouts,
            "total_passed": total_passed,
            "avg_rollouts_per_item": total_rollouts / total_items if total_items > 0 else 0,
            "overall_pass_rate": total_passed / total_rollouts if total_rollouts > 0 else 0,
            "avg_pass_rate_per_item": sum(pass_rates) / len(pass_rates) if pass_rates else 0,
            "items_with_pass": sum(1 for r in pass_rates if r > 0),
            # Truncation statistics
            "total_generated": total_generated,
            "truncated_count": truncated_count,
            "truncation_rate": round(truncation_rate, 4),
            # Timing (in seconds)
            "total_time_sec": round(total_time, 2),
            "rollout_time_sec": round(rollout_time, 2),
            "sampling_time_sec": round(self._sampling_time, 2),
            "verify_time_sec": round(self._verify_time, 2),
        }

        # Save stats
        stats_path = summary_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"  Total items: {stats['total_items']}")
        print(
            f"  Total rollouts: {stats['total_rollouts']} (generated: {total_generated}, truncated: {truncated_count})"
        )
        print(f"  Truncation rate: {truncation_rate:.2%}")
        print(f"  Overall pass rate: {stats['overall_pass_rate']:.2%}")
        print(f"  Items with at least 1 pass: {stats['items_with_pass']}")
        print(
            f"  Timing: total={total_time:.1f}s, rollout={rollout_time:.1f}s "
            f"(sampling={self._sampling_time:.1f}s, verify={self._verify_time:.1f}s)"
        )
        print(f"  Saved stats to {stats_path}")


def run_pipeline(cfg) -> None:
    """Run pipeline (sync wrapper for async pipeline)."""
    pipeline = Pipeline(cfg)
    asyncio.run(pipeline.run())
