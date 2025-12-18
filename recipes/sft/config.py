"""SFT Recipe 配置."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SFTConfig:
    """SFT Recipe 默认配置. yaml 中的配置会覆盖这里的配置"""

    # 采样配置
    model: str = "qwen"
    base_url: str = None
    api_key: str = None
    n_samples: int = 16
    temperature: float = 0.7
    max_tokens: int = 4096
    semaphore_per_sampler: int = 10

    # 数据路径配置
    abs_image_path_field: str = "meta_info.abs_image_path"  # 图片绝对路径字段（支持嵌套，如 meta_info.abs_image_path 或 abs_path）

    # LLM Judge 配置
    judge_model: str = None  # 默认使用 model
    judge_base_url: str = None  # 默认使用 base_url
    judge_api_key: str = None  # 默认使用 api_key
    judge_temperature: float = 0.0
    judge_max_tokens: int = 10
    verifier_max_workers: int = 20  # VerifierStage 线程池大小 (batch 内 item 级别并发)

    # 格式化配置
    pass_threshold: float = 1.0
    verbose: bool = False

    # 重试配置
    max_retries: int = 3

    # Pipeline 配置
    batch_size: int = 4  # 每个 batch 的数据量
    concurrency: int = 4  # 默认并发度(Stage 的 actor 数量)
    sampler_concurrency: int = None  # SamplerStage 并发度(默认与 concurrency 相同)
    verifier_concurrency: int = None  # VerifierStage 并发度(默认与 concurrency 相同)

    @classmethod
    def from_yaml(cls, path: str) -> "SFTConfig":
        """从 yaml 文件加载配置."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """保存配置到 yaml 文件."""
        data = {k: v for k, v in self.__dict__.items() if v is not None}
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
