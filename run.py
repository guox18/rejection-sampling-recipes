#!/usr/bin/env python3
"""Entry point for rejection sampling pipeline."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.pipeline import run_pipeline


def _get_cli_overrides() -> list[str]:
    """Extract CLI overrides from sys.argv."""
    overrides = []
    for arg in sys.argv[1:]:
        # Skip hydra special args
        if arg.startswith("--") or arg.startswith("-"):
            continue
        # Keep key=value overrides
        if "=" in arg:
            overrides.append(arg)
    return overrides


def _apply_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """Apply CLI overrides to config."""
    import yaml

    for override in overrides:
        if override.startswith("++"):
            # Append mode: ++key=value
            override = override[2:]
        key, value = override.split("=", 1)

        # Parse value (handle int, float, bool, dict, list, string)
        try:
            parsed_value = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed_value = value

        OmegaConf.update(cfg, key, parsed_value, merge=True)
    return cfg


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

        # Resume from checkpoint (auto-loads config from work_dir)
        python run.py work_dir=output/20251206_143052/
    """
    # If work_dir is specified, load config from it
    if cfg.work_dir:
        config_path = Path(cfg.work_dir) / "config.yaml"
        if config_path.exists():
            # Load saved config as base
            saved_cfg = OmegaConf.load(config_path)

            # Get CLI overrides (excluding work_dir itself for base config)
            cli_overrides = [o for o in _get_cli_overrides() if not o.startswith("work_dir=")]

            # Use saved config as base, apply only CLI overrides
            cfg = saved_cfg
            cfg.work_dir = str(config_path.parent)  # Keep work_dir

            # Apply CLI overrides on top
            if cli_overrides:
                _apply_overrides(cfg, cli_overrides)
                print(f"Loaded config from {config_path}, with overrides: {cli_overrides}")
            else:
                print(f"Loaded config from {config_path}")

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
