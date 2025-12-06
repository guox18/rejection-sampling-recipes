#!/usr/bin/env python3
"""Entry point for rejection sampling pipeline."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the rejection sampling pipeline."""
    # TODO: Implement pipeline
    print(cfg)


if __name__ == "__main__":
    main()
