"""Pipeline: 封装 Ray Data 的流水线框架."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import ray
import ray.data
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from .base import BaseRecipe, Stage
from .utils.data_io import convert_to_python_types, convert_scalar_to_python
from .utils.framework import FRAMEWORK_FIELDS


def clean_nan_values(item: dict, warn: bool = True) -> dict:
    """
    清理字典中的 NaN 值，将其转换为 None. 数据是列式操作的, 所有 item 需要具有相同的字段.
    如果有些项独有某些字段, pandas 会为缺失的字段带上 NaN 值.
    
    Args:
        item: 要清理的数据字典
        warn: 是否在发现 NaN 时打印警告
    
    Returns:
        清理后的字典（NaN 转换为 None）
    """
    import math
    
    has_nan = False
    cleaned_item = {}
    
    for k, v in item.items():
        # 检查是否为 NaN（包括 pandas 的 NaN 和 numpy 的 nan）
        if isinstance(v, float) and math.isnan(v):
            if warn and not has_nan:
                # 第一次发现 NaN 时打印警告（每个 item 只警告一次）
                print(f"⚠️  Warning: Found NaN value in item {item.get('id', 'unknown')}")
                print(f"   This should not happen after framework initialization.")
                print(f"   Please report this as a bug if you see this message.")
                has_nan = True
            cleaned_item[k] = None
        else:
            cleaned_item[k] = v
    
    return cleaned_item


def safe_str(value, max_length: int = 500) -> str:
    """
    安全地将值转换为字符串，避免打印过大的数据导致程序崩溃.
    
    Args:
        value: 要转换的值
        max_length: 字符串的最大长度
    
    Returns:
        安全的字符串表示
    """
    try:
        v_str = str(value)
        if len(v_str) > max_length:
            return v_str[:max_length] + f"... (truncated, total {len(v_str)} chars)"
        return v_str
    except Exception as e:
        return f"<error converting to string: {e}>"


def safe_repr_item(item: dict, max_value_length: int = 200) -> str:
    """
    安全地生成 item 的字符串表示，避免打印过大的数据导致程序崩溃.
    
    Args:
        item: 要表示的数据字典
        max_value_length: 每个字段值的最大长度
    
    Returns:
        安全的字符串表示
    """
    result = []
    for k, v in item.items():
        v_str = safe_str(v, max_value_length)
        result.append(f"  {k}: {v_str}")
    return "{\n" + "\n".join(result) + "\n}"


class Pipeline:
    """
    流水线框架, 封装 Ray Data.

    功能:
    - 流式数据处理
    - 并发控制(通过 batchsize 和 concurrency 参数. 总并发量相当于 batchsize * concurrency)
    - 顺序保证(通过 preserve_order 参数控制, 默认开启. 关闭可提高性能.)
    - 背压控制(Ray Data 内置)
    - 支持断点续传(基于已处理 ID 跳过)
    - Actor 模式：每个 worker 只初始化一次(适合 API client 复用、vLLM 等)

    Usage:
        pipeline = Pipeline(
            recipe=SFTRecipe(config),
            batch_size=32,
            concurrency=2,
        )
        pipeline.run("data/input.jsonl", "output/train.jsonl")
    """

    def __init__(
        self,
        recipe: BaseRecipe,
        batch_size: int = 32,
        concurrency: int = 2,
        stage_concurrency: dict[str, int] = None,
        work_dir: str = None,
        preserve_order: bool = True,
        resume: bool = True,
        flush_interval: int = 10,
    ):
        """
        初始化 Pipeline.

        Args:
            recipe: Recipe 实例
            batch_size: 每个 batch 的数据量, 数据处理的最小单元.
            concurrency: 流水线并行度. 总并发量相当于 batchsize * concurrency
            stage_concurrency: 按 Stage 类名配置并发度, 如 {"SamplerStage": 100}. 优先级高于 concurrency 参数.
            work_dir: 工作目录, 用于存放中间结果. 默认自动生成, 推荐指定一个固定的目录.
            preserve_order: 是否保持数据顺序. 默认 True(保持顺序).
                           设为 False 可提高性能, 但输出顺序可能不一致.
            resume: 是否启用断点续传. 默认 True.
                   启用后会跳过输出文件中已存在的数据.
                   自动基于 "messages" 内容计算哈希值(_resume_id)进行去重.
                   如要关闭此项, 请确保输出文件为空, 避免重复.
            flush_interval: 数据刷新间隔(条数). 每处理 N 条成功数据就刷新到磁盘.
        """
        self.recipe = recipe
        self.batch_size = batch_size
        self.concurrency = concurrency
        self.stage_concurrency = stage_concurrency or {}
        self.work_dir = Path(work_dir) if work_dir else self._auto_work_dir()
        self.preserve_order = preserve_order
        self.resume = resume
        self.flush_interval = flush_interval

        self._stages = recipe.stages()

        # 验证 stage_concurrency 配置
        self._validate_stage_concurrency()

    def _auto_work_dir(self) -> Path:
        """自动生成工作目录."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("output") / timestamp

    def _validate_stage_concurrency(self):
        """
        验证 stage_concurrency 配置.

        确保用户配置的 stage 名称都是有效的, 避免配置错误而不自知.
        """
        if not self.stage_concurrency:
            return

        valid_stage_names = {type(stage).__name__ for stage in self._stages}
        invalid_names = set(self.stage_concurrency.keys()) - valid_stage_names

        if invalid_names:
            raise ValueError(
                f"Invalid stage names in stage_concurrency: {invalid_names}. "
                f"Valid stage names are: {valid_stage_names}"
            )

    def _compute_resume_id(self, item: dict) -> str:
        """
        计算数据项的断点续传 ID.

        基于数据内容（默认使用 messages 字段）生成确定性哈希值.
        这是框架内部实现，用户无需感知.

        Args:
            item: 数据项字典

        Returns:
            16位哈希字符串（非空，保证不为 None）

        Note:
            如果返回值为 None 或空字符串，Pipeline 会在 add_resume_id 阶段抛出异常
        """
        import hashlib

        # 使用 messages 字段计算哈希（如果存在）
        messages = item.get("messages", [])
        if messages:
            # 转换为 Python 原生类型（处理 numpy/pandas 类型）
            messages = convert_to_python_types(messages)
            content = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        else:
            # 如果没有 messages 字段，使用整个 item
            content = json.dumps(item, sort_keys=True, ensure_ascii=False)

        hash_obj = hashlib.sha256(content.encode("utf-8"))
        return hash_obj.hexdigest()[:16]

    def _load_processed_ids(self, output_path: str) -> set:
        """
        从输出文件加载已处理的 ID.

        用于断点续传：跳过已处理的数据.
        使用 _resume_id 字段(基于内容哈希的确定性 ID).

        注意：所有 ID 会加载到内存中.
        """
        processed_ids = set()
        if os.path.exists(output_path):
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            resume_id = item.get("_resume_id")
                            if resume_id is not None:
                                processed_ids.add(resume_id)
                        except json.JSONDecodeError:
                            continue

            if len(processed_ids) > 1_000_000:
                print(f"[Warning] Loaded {len(processed_ids):,} processed IDs into memory.")

        return processed_ids

    def _add_framework_fields(self, batch):
        """
        为每个 item 添加框架字段.

        框架字段：
        - _resume_id: 基于内容哈希的唯一 ID（用于断点续传）
        - _failed: 标记 item 是否处理失败（初始值 None）
        - _error: 错误消息（初始值 None）
        - _traceback: 错误堆栈（初始值 None）

        所有 item 都会有这些字段，避免 pandas DataFrame 填充 NaN。
        """
        # 转为 DataFrame（如果不是的话）
        if isinstance(batch, dict):
            df = pd.DataFrame(batch)
        else:
            df = batch

        if df.empty:
            return df

        # 为每行计算 resume_id
        resume_ids = []
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            resume_id = self._compute_resume_id(row_dict)

            # 检查 resume_id 是否为 None 或空字符串
            if resume_id is None or resume_id == "":
                item_id = row_dict.get("id", "unknown")
                error_msg = (
                    f"❌ Failed to compute _resume_id for item (id={item_id}, row_index={idx})\n"
                    f"   resume_id is None or empty. This usually indicates:\n"
                    f"   1. The data item is completely empty or malformed\n"
                    f"   2. The 'messages' field is missing or invalid\n"
                    f"   Data sample: {str(row_dict)[:200]}..."
                )
                raise ValueError(error_msg)

            resume_ids.append(resume_id)

        # 添加所有框架字段（初始值为 None，确保数据结构一致）
        df["_resume_id"] = resume_ids
        df["_failed"] = None
        df["_error"] = None
        df["_traceback"] = None

        return df

    def _create_filter_processed_fn(self, processed_ids: set):
        """
        创建过滤已处理 item 的函数.

        Args:
            processed_ids: 已处理的 _resume_id 集合

        Returns:
            过滤函数，用于 map_batches
        """

        def filter_processed(batch):
            """过滤掉已处理的 item."""
            # 转为 DataFrame（如果不是的话）
            if isinstance(batch, dict):
                df = pd.DataFrame(batch)
            else:
                df = batch

            if df.empty:
                return df

            # 使用已有的 _resume_id 字段过滤
            mask = ~df["_resume_id"].isin(processed_ids)
            return df[mask]

        return filter_processed

    def run(self, input_path: str, output_path: str = None):
        """
        执行流水线.

        Args:
            input_path: 输入数据路径, 目前只支持 jsonl 格式.
            output_path: 输出数据路径, 默认为 work_dir/output.jsonl

        Raises:
            ValueError: 如果数据项的 _resume_id 计算结果为 None 或空字符串
        """
        # 确保 Ray 已初始化
        if not ray.is_initialized():
            ray.init()
        
        # 设置输出路径
        if output_path is None:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(self.work_dir / "output.jsonl")
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # 设置 preserve_order 和进度条
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options.preserve_order = self.preserve_order
        ctx.enable_rich_progress_bars = True
        ctx.use_ray_tqdm = False

        # 流水线优化：显示详细进度
        ctx.execution_options.verbose_progress = True

        # 加载已处理的 ID(断点续传)
        processed_ids = set()
        if self.resume:
            processed_ids = self._load_processed_ids(output_path)
            if processed_ids:
                print(f"[Resume] Found {len(processed_ids)} processed items, will skip them.")

        # 读取数据（JSONL 格式：每行一个 JSON 对象）
        ds = ray.data.read_json(input_path, lines=True)

        # 添加框架字段（用于断点续传和错误处理）
        ds = ds.map_batches(
            self._add_framework_fields,
            batch_format="pandas",
            batch_size=self.batch_size,
            compute=TaskPoolStrategy(size=self.concurrency),
        )

        # 过滤已处理的数据（断点续传）
        if processed_ids:
            ds = ds.map_batches(
                self._create_filter_processed_fn(processed_ids),
                batch_format="pandas",
                batch_size=self.batch_size,
                compute=TaskPoolStrategy(size=self.concurrency),
            )

        # 依次应用每个 Stage (使用 Actor 模式, 每个 worker 只初始化一次)
        for stage in self._stages:
            concurrency = self._get_stage_concurrency(stage)
            ds = ds.map_batches(
                self._create_map_batches_fn(stage),
                batch_format="pandas",
                batch_size=self.batch_size,
                compute=ActorPoolStrategy(size=concurrency),
            )

        # 写入输出
        self._write_output(ds, output_path, len(processed_ids))

    def _write_output(self, ds, output_path: str, resume_count: int):
        """
        写入输出数据（追加模式，支持断点续传）.

        Args:
            ds: Ray Dataset 对象
            output_path: 输出文件路径
            resume_count: 已跳过的断点续传数据条数
        """
        total_rows = 0
        success_rows = 0
        failed_rows = 0

        with open(output_path, "a") as f:
            try:
                for row in ds.iter_rows():
                    item = dict(row)
                    total_rows += 1

                    # pandas 处理数据时可能引入 NaN 值
                    item = clean_nan_values(item, warn=True)

                    # 统计失败的 item，但仍然保留在输出中
                    if item.get("_failed") is True:
                        failed_rows += 1
                        error_msg = safe_str(item.get("_error") or "Unknown error", max_length=10000)
                        item_id = safe_str(item.get("id", "unknown"), max_length=100)
                        resume_id = safe_str(item.get("_resume_id", "unknown"), max_length=100)
                        traceback_msg = safe_str(
                            item.get("_traceback", "unknown"), max_length=10000
                        )
                        print(f"[Pipeline] ⚠️  Writing failed item to output:")
                        print(f"  - ID: {item_id}")
                        print(f"  - Resume ID: {resume_id}")
                        print(f"  - Error: {error_msg}")
                        print(f"  - Traceback: {traceback_msg}")
                    else:
                        success_rows += 1

                    # 写入所有 item
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    # 定期刷新，确保数据及时写入磁盘
                    if success_rows % self.flush_interval == 0:
                        f.flush()
                        os.fsync(f.fileno())
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                print(f"[Pipeline] ❌ Error writing to output file: {safe_str(e, max_length=200)}")
                print(f"  Traceback:\n{safe_str(error_trace, max_length=2000)}")
                print(f"  Item that caused the error:\n{safe_repr_item(item)}")
            finally:
                # 最后再刷新一次，确保所有数据都写入
                f.flush()
                os.fsync(f.fileno())

        print(f"\n[Pipeline Summary]")
        print(f"  Resume items:  {resume_count}")
        print(f"  New items:     {total_rows}")
        print(f"  Success:       {success_rows}")
        print(f"  Failed:        {failed_rows}")
        print(f"  Output file:   {output_path}")
        print(f"\n[Done] Output written to: {output_path}")

    def _get_stage_concurrency(self, stage: Stage) -> int:
        """
        获取 Stage 的并发度.

        优先使用 stage_concurrency 配置, 否则使用默认 concurrency.
        """
        stage_name = type(stage).__name__
        return self.stage_concurrency.get(stage_name, self.concurrency)

    def _create_map_batches_fn(self, stage: Stage) -> Callable:
        """
        为 Stage 创建 map_batches 可用的函数.

        处理：
        - 支持同步和异步 process 方法
        - 失败 item 的跳过逻辑
        - 异常捕获和 _failed 标记
        """
        is_async = stage.is_async()

        class StageCallable:
            """封装 Stage 为 Ray Data 可调用对象."""

            def __init__(self):
                # Actor 模式：每个 worker 只初始化一次
                self._stage = stage
                self._stage.initialize()

            def __del__(self):
                if hasattr(self, "_stage"):
                    self._stage.shutdown()

            def __call__(self, batch):
                """处理 batch."""
                # 转为 DataFrame
                if isinstance(batch, dict):
                    df = pd.DataFrame(batch)
                else:
                    df = batch

                if df.empty:
                    return df

                # 转为 list of dict
                rows = df.to_dict("records")
                rows = [convert_to_python_types(row) for row in rows]

                # 保存每个 item 的框架字段
                framework_data = {
                    idx: {k: item.get(k) for k in FRAMEWORK_FIELDS}
                    for idx, item in enumerate(rows)
                }

                try:
                    if is_async:
                        results = asyncio.run(self._stage.process(rows))
                    else:
                        results = self._stage.process(rows)

                    # 恢复框架字段
                    for idx, result in enumerate(results):
                        for field, value in framework_data[idx].items():
                            if value is not None and result.get(field) is None:
                                result[field] = value

                except Exception as e:
                    # Stage 级兜底异常处理：整个 batch 都会被标记失败
                    # 正常情况下，Stage.process_item() 应自行处理异常，只标记单个 item 失败
                    # 触发此分支说明 Stage 有未捕获的异常，应尽量避免
                    import traceback

                    error_trace = traceback.format_exc()
                    stage_name = type(self._stage).__name__
                    print(f"[{stage_name}] ❌ Batch processing failed:")
                    print(f"  Error: {safe_str(e, max_length=500)}")
                    print(f"  Traceback:\n{safe_str(error_trace, max_length=2000)}")

                    results = [
                        {
                            **item,
                            **framework_data[idx],
                            "_failed": True,
                            "_error": safe_str(f"{stage_name}: {str(e)}", max_length=1000),
                            "_traceback": safe_str(error_trace, max_length=5000),
                        }
                        for idx, item in enumerate(rows)
                    ]

                # 字段对齐（当不同 item 的字段不一致时）
                if len(results) > 1:
                    all_keys = [set(item.keys()) for item in results]
                    if not all(keys == all_keys[0] for keys in all_keys):
                        all_fields = set().union(*all_keys)
                        results = [
                            {field: item.get(field) for field in all_fields}
                            for item in results
                        ]

                return pd.DataFrame(results)

        return StageCallable
