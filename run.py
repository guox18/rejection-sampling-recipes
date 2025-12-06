#!/usr/bin/env python3
"""Entry point for rejection sampling pipeline."""

import hydra
from omegaconf import DictConfig

from src.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Run the rejection sampling pipeline.

    Usage:
        # Start new experiment
        python run.py data.input_path=/path/to/data.jsonl

        # Override config
        python run.py data.input_path=/path/to/data.jsonl \\
            sampler.model=deepseek-chat \\
            sampling.max_rollouts=32

        # Resume from checkpoint
        python run.py work_dir=output/20251206_143052/
    """
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
