"""Pipeline: 封装 Ray Data 的流水线框架."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

import ray
import ray.data
from ray.data import ActorPoolStrategy, TaskPoolStrategy

from .base import BaseRecipe, Stage
from .utils.data_io import convert_to_python_types, convert_scalar_to_python


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

    def _get_nested_value(self, item: dict, field_path: str):
        """
        获取嵌套字段的值.

        Args:
            item: 数据字典
            field_path: 字段路径, 如 "metadata._uid" 或 "id"

        Returns:
            字段值, 如果不存在则返回 None
        """
        parts = field_path.split(".")
        value = item
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None
        return value

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

        # 添加 _resume_id 字段（框架内部字段，用于断点续传）
        def add_resume_id(batch):
            """为每个 item 添加 _resume_id（基于内容哈希）."""
            import pandas as pd

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

            df["_resume_id"] = resume_ids
            return df

        ds = ds.map_batches(
            add_resume_id,
            batch_format="pandas",
            batch_size=self.batch_size,
            compute=TaskPoolStrategy(size=self.concurrency),
        )

        # 过滤已处理的数据（断点续传）
        if processed_ids:

            def filter_processed(batch):
                """过滤掉已处理的 item."""
                import pandas as pd

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

            ds = ds.map_batches(
                filter_processed,
                batch_format="pandas",
                batch_size=self.batch_size,
                compute=TaskPoolStrategy(size=self.concurrency),
            )

        # 依次应用每个 Stage(使用 Actor 模式, 每个 worker 只初始化一次)
        for stage in self._stages:
            concurrency = self._get_stage_concurrency(stage)
            map_fn = self._create_map_batches_fn(stage)
            ds = ds.map_batches(
                map_fn,
                batch_size=self.batch_size,
                batch_format="pandas",
                compute=ActorPoolStrategy(size=concurrency),
            )

        # 写入输出（追加模式，支持断点续传）
        # 使用 iter_rows 以便逐行写入
        total_rows = 0
        success_rows = 0
        failed_rows = 0

        with open(output_path, "a") as f:
            try:
                for row in ds.iter_rows():
                    # row 是字典格式
                    item = dict(row)
                    total_rows += 1

                    # 跳过失败的 item（如果最后一个 Stage 没有过滤掉的话）
                    if item.get("_failed"):
                        failed_rows += 1
                        error_msg = item.get("_error", "Unknown error")
                        # 处理 NaN 值
                        if error_msg != error_msg:  # NaN check (NaN != NaN)
                            error_msg = "No error message (NaN)"
                        item_id = item.get("id", "unknown")
                        resume_id = item.get("_resume_id", "unknown")
                        print(f"[Pipeline] ❌ Skipping failed item:")
                        print(f"  - ID: {item_id}")
                        print(f"  - Resume ID: {resume_id}")
                        print(f"  - Error: {error_msg}")
                        # 打印 item 的部分内容用于调试（避免输出过多）
                        debug_info = {
                            k: v
                            for k, v in item.items()
                            if k in ["id", "_resume_id", "_error", "_failed"]
                        }
                        print(f"  - Debug info: {debug_info}")
                        continue

                    success_rows += 1
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

                    # 定期刷新，确保数据及时写入磁盘（断点续传）
                    if success_rows % self.flush_interval == 0:
                        f.flush()
                        os.fsync(f.fileno())  # 强制写入磁盘
            finally:
                # 最后再刷新一次，确保所有数据都写入
                f.flush()
                os.fsync(f.fileno())

        print(f"\n[Pipeline Summary]")
        print(f"  Resume items:  {len(processed_ids)}")
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
                """
                处理 batch.

                Ray Data 使用 pandas DataFrame 格式.
                """
                import pandas as pd

                # 转为 DataFrame(如果不是的话)
                if isinstance(batch, dict):
                    df = pd.DataFrame(batch)
                else:
                    df = batch

                if df.empty:
                    return df

                # 转为 list of dict, 并转换为 Python 原生类型
                rows = df.to_dict("records")
                rows = [convert_to_python_types(row) for row in rows]

                # 框架内部字段（需要自动保留）
                FRAMEWORK_FIELDS = {"_resume_id", "_failed", "_error", "_traceback"}

                # 保存每个 item 的框架字段
                framework_data = {}
                for idx, item in enumerate(rows):
                    framework_data[idx] = {k: item.get(k) for k in FRAMEWORK_FIELDS}

                # 分离失败和正常的 item
                failed_items = []
                normal_items = []
                normal_items_indices = []  # 记录正常 item 的原始索引

                for idx, item in enumerate(rows):
                    if item.get("_failed"):
                        failed_items.append(item)
                    else:
                        normal_items.append(item)
                        normal_items_indices.append(idx)

                # 处理正常的 item
                if normal_items:
                    try:
                        if is_async:
                            # 异步模式：使用 asyncio 运行
                            results = asyncio.run(self._stage.process(normal_items))
                        else:
                            # 同步模式
                            results = self._stage.process(normal_items)

                        # 自动恢复框架字段到处理结果中
                        for result_idx, result in enumerate(results):
                            original_idx = normal_items_indices[result_idx]
                            saved_fields = framework_data[original_idx]

                            # 只恢复非 None 的框架字段（除非 Stage 明确设置了它们）
                            for field, value in saved_fields.items():
                                if value is not None and field not in result:
                                    result[field] = value

                    except Exception as e:
                        # 失败处理兜底, 标记所有 item 为失败.
                        import traceback

                        error_trace = traceback.format_exc()
                        stage_name = type(self._stage).__name__
                        print(f"[{stage_name}] ❌ Batch processing failed:")
                        print(f"  Error: {e}")
                        print(f"  Traceback:\n{error_trace}")
                        results = []
                        for result_idx, item in enumerate(normal_items):
                            original_idx = normal_items_indices[result_idx]
                            saved_fields = framework_data[original_idx]
                            # 保留原有的 _resume_id，添加失败标记
                            result = {
                                **item,
                                **saved_fields,  # 恢复框架字段
                                "_failed": True,
                                "_error": f"{stage_name}: {str(e)}",
                                "_traceback": error_trace,
                            }
                            results.append(result)
                else:
                    results = []

                # 合并结果：失败的 item + 处理结果
                all_results = failed_items + results

                # 转回 DataFrame
                if not all_results:
                    return pd.DataFrame()

                return pd.DataFrame(all_results)

        return StageCallable
