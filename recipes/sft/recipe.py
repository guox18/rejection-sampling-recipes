"""
SFT Recipe: 采样 → 验证 → 格式化.

Stage 支持三种执行模式:
- 同步模式: 顺序执行 (默认)
- 异步模式: asyncio 并发执行 (@Stage.async_mode)
- 多线程模式: 线程池并发执行 (@Stage.threaded_mode)

Stage 支持两种实现方式:
- 只实现 process_item(): 框架自动批处理和异常处理 (推荐)
- 覆盖 process(): 完全自定义批处理 (高级)

本文件示例:
- SamplerStage: 异步 + process_item + AsyncOpenAIClient (API 调用)
- VerifierStage: 多线程 + process_item + SyncOpenAIClient (LLM Judge)
- FormatterStage: 同步 + process_item (数据格式化)

框架内部字段:
- _resume_id: 用于断点续传的唯一标识符（基于内容哈希）
- _failed: 标记处理失败的数据项
- _error: 失败原因
- _traceback: 失败时的堆栈跟踪

注意：
- 这些字段由 Pipeline 框架自动添加和保留
- Stage 实现时无需手动处理这些字段
- 即使 Stage 不返回这些字段，框架也会自动恢复它们
"""

import asyncio
import os

import aiohttp

from src.base import BaseRecipe, Stage

from .config import SFTConfig
from .tools import AsyncOpenAIClient, SyncOpenAIClient, DEFAULT_JUDGE_TEMPLATE, clip_thinking

# ============================================================
# 示例 1: 异步模式 (使用装饰器)
# ============================================================


@Stage.async_mode
class SamplerStage(Stage):
    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: AsyncOpenAIClient = None

    def initialize(self):
        """Actor 创建时调用一次, 创建共享的客户端配置."""
        self.client = AsyncOpenAIClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
            semaphore_size=self.config.semaphore_per_sampler,
        )

    async def process(self, batch: list[dict]) -> list[dict]:
        """
        覆盖 process() 方法, 在 batch 级别管理 session 和 semaphore.

        这样 session 和 semaphore 可以在整个 batch 内共享,
        而不是每个 item 都创建一次.
        """
        # 在 batch 级别创建 session 和 semaphore (此时在 event loop 中)
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.client.semaphore_size)

            # 并发处理 batch 内的所有 items
            async def process_one(item: dict) -> dict:
                if item.get("_failed"):
                    return item

                try:
                    messages = item.get("messages", [])
                    # 调用 API (共享 session 和 semaphore)
                    responses = await self.client.chat_completion(
                        session=session,
                        semaphore=semaphore,
                        messages=messages,
                        model=self.config.model,
                        n=self.config.n_samples,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    return {**item, "responses": responses}
                except Exception as e:
                    import traceback

                    error_trace = traceback.format_exc()
                    item_id = item.get("id", "unknown")
                    print(f"[SamplerStage] ❌ Item {item_id} failed:")
                    print(f"  Error: {e}")
                    print(f"  Traceback:\n{error_trace}")
                    return {
                        **item,
                        "_failed": True,
                        "_error": f"SamplerStage: {e}",
                        "_traceback": error_trace,
                    }

            return await asyncio.gather(*[process_one(item) for item in batch])


# ============================================================
# 示例 2: 多线程模式 (使用装饰器)
# ============================================================


@Stage.threaded_mode
class VerifierStage(Stage):
    """
    验证阶段: 使用 LLM-as-Judge 为每个 response 打分.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: SyncOpenAIClient = None

    def initialize(self):
        """
        Actor 创建时调用一次, 设置线程池大小并初始化客户端. 通过设置 self._thread_pool_size 指定线程池大小.
        """
        self._thread_pool_size = self.config.verifier_max_workers

        # 初始化同步客户端
        api_key = self.config.judge_api_key or self.config.api_key
        base_url = self.config.judge_base_url or self.config.base_url

        self.client = SyncOpenAIClient(api_key=api_key, base_url=base_url)
        self.client.initialize()

    def _extract_question_from_messages(self, messages: list[dict]) -> str:
        """
        提取问题 (支持多模态数据)
        """
        if not messages:
            return ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for content_item in content:
                        if isinstance(content_item, dict) and content_item.get("type") == "text":
                            question = content_item.get("text", "")
                            break
                else:
                    question = content
                break
        return question

    def process_item(self, item: dict) -> dict:
        """
        处理单个 item, 为所有 responses 打分.
        """
        responses = item.get("responses", [])
        metadata = item.get("metadata", {})
        messages = item.get("messages", [])

        question = self._extract_question_from_messages(messages)
        rollouts, first_judge_prompt, first_judge_output = self._verify_llm_judge(
            responses, metadata, question
        )

        # 更换字段名为 rollouts
        result = {k: v for k, v in item.items() if k != "responses"}
        result["rollouts"] = rollouts

        if self.config.verbose and first_judge_prompt is not None:
            if "metadata" not in result:
                result["metadata"] = {}
            result["metadata"]["judge_prompt_sample"] = first_judge_prompt
            result["metadata"]["judge_output_sample"] = first_judge_output

        return result

    def _verify_llm_judge(
        self, responses: list[str], metadata: dict, question: str
    ) -> tuple[list[dict], str, str]:
        """
        使用 LLM Judge 验证多个 responses.

        对 1 个 item 的 N 个 responses, 顺序调用 judge 验证.

        Returns:
            tuple: (rollouts, first_judge_prompt, first_judge_output)
                - rollouts: 验证结果列表
                - first_judge_prompt: 第一条 judge 提示词 (用于调试)
                - first_judge_output: 第一条 judge 输出结果 (用于调试)
        """
        if not responses:
            return [], None, None

        gold_target = metadata.get("answer") or metadata.get("gold_target", "")
        rollouts = []
        first_judge_prompt = None
        first_judge_output = None

        for idx, response in enumerate(responses):
            clipped_response = clip_thinking(response)
            prompt = DEFAULT_JUDGE_TEMPLATE.format(
                question=question,
                gold_target=gold_target,
                predicted_answer=clipped_response,
            )

            try:
                judge_output = self._call_judge(prompt)
                is_correct = self._parse_judge_output(judge_output)
                score = 1.0 if is_correct else 0.0

                if idx == 0:
                    first_judge_prompt = prompt
                    first_judge_output = judge_output
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                print(f"[VerifierStage] ⚠️  Judge error on response {idx}:")
                print(f"  Error: {e}")
                print(f"  Traceback:\n{error_trace}")
                score = 0.0

                # 第一条失败时也要记录
                if idx == 0:
                    first_judge_prompt = prompt
                    first_judge_output = f"ERROR: {e}\n\nTraceback:\n{error_trace}"

            rollouts.append({"response": response, "score": score})

        return rollouts, first_judge_prompt, first_judge_output

    def _call_judge(self, prompt: str) -> str:
        """调用 judge model."""
        model = self.config.judge_model or self.config.model
        return self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=self.config.judge_max_tokens,
            temperature=self.config.judge_temperature,
        )

    def _parse_judge_output(self, output: str) -> bool:
        """解析 judge 输出 (A=正确, B=错误)."""
        if not output:
            return False
        cleaned = output.strip().lower()
        if cleaned in ("a", "correct"):
            return True
        elif cleaned in ("b", "incorrect"):
            return False
        else:
            return False


# ============================================================
# 示例 3: 同步模式 - 只实现 process_item (推荐)
# ============================================================


class FormatterStage(Stage):
    """
    格式化阶段: 选择通过验证的 response, 输出 SFT 格式.
    """

    def __init__(self, config: SFTConfig):
        self.config = config

    def _restore_image_urls(self, messages: list, original_urls: list) -> list:
        """
        将 messages 中的 base64 图像 URL 恢复为原始相对路径.

        Args:
            messages: 包含 base64 图像的消息列表
            original_urls: 原始的相对路径列表

        Returns:
            恢复相对路径后的消息列表
        """
        if not original_urls:
            return messages

        # 复制 messages 避免修改原始数据
        restored_messages = []
        url_index = 0

        for msg in messages:
            restored_msg = {"role": msg["role"]}
            content = msg.get("content")

            # 如果是 user 消息且 content 是列表（可能包含图像）
            if msg.get("role") == "user" and isinstance(content, list):
                restored_content = []
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                        # 恢复为原始相对路径
                        if url_index < len(original_urls):
                            restored_item = {
                                "type": "image_url",
                                "image_url": {"url": original_urls[url_index]},
                            }
                            # 保留 image_wh 信息（如果原始数据有的话）
                            image_url_data = content_item.get("image_url", {})
                            if "image_wh" in image_url_data:
                                restored_item["image_url"]["image_wh"] = image_url_data["image_wh"]

                            restored_content.append(restored_item)
                            url_index += 1
                        else:
                            # 如果没有对应的原始 URL，保持原样
                            restored_content.append(content_item)
                    else:
                        restored_content.append(content_item)
                restored_msg["content"] = restored_content
            else:
                # 非 user 消息或纯文本，直接复制
                restored_msg["content"] = content

            restored_messages.append(restored_msg)

        return restored_messages

    def process_item(self, item: dict) -> dict:
        """
        处理单个 item, 格式化为 SFT 训练数据.
        """
        messages = item.get("messages", [])
        rollouts = item.get("rollouts", [])
        metadata = item.get("metadata", {})
        item_id = item.get("id", "unknown")

        # 选择通过验证的 responses
        passed = [r for r in rollouts if r.get("score", 0) >= self.config.pass_threshold]

        print(f"[FormatterStage] Item {item_id}: {len(passed)}/{len(rollouts)} passed")

        # 选择 best response
        if passed:
            best_response = passed[0]["response"]
            used_gt = False
        else:
            # 回退到 ground truth
            gt = metadata.get("answer") or metadata.get("gold_target", "")
            if gt:
                best_response = gt
                used_gt = True
                print(f"[FormatterStage] Item {item_id}: Using ground truth")
            else:
                # 主动标记失败
                print(f"[FormatterStage] Item {item_id}: No valid response")
                return {
                    **item,
                    "_failed": True,
                    "_error": "No response passed and no ground truth",
                }

        sft_messages = messages + [{"role": "assistant", "content": best_response}]

        # 清理 metadata（移除内部调试信息）
        clean_metadata = {
            k: v
            for k, v in metadata.items()
            if k not in ["judge_prompt_sample", "judge_output_sample"]
        }
        clean_metadata.update(
            {
                "n_passed": len(passed),
                "n_total": len(rollouts),
                "used_ground_truth": used_gt,  # 在模型无法答对时, 保留原始 ground_truth
            }
        )

        result = {}
        if "id" in item:
            result["id"] = item["id"]

        result = {
            **result,
            "messages": sft_messages,
            "metadata": clean_metadata,
        }

        return result


# ============================================================
# Recipe 定义
# ============================================================


class SFTRecipe(BaseRecipe):
    """
    SFT Recipe: 组合 3 个 Stage 完成 SFT 数据生成.

    Stage 流水线:
    1. SamplerStage (异步): 生成 N 个 responses
    2. VerifierStage (多线程): 验证每个 response
    3. FormatterStage (同步): 输出 SFT 格式

    并发控制参数说明:
    - stage_concurrency: 控制 Stage 的 Actor 数量 (多少个 batch 并发处理)
    - semaphore_per_sampler: 控制单个 SamplerStage Actor 内部的并发请求数
    - verifier_max_workers: 控制 VerifierStage batch 内有多少个 item 并发处理
    """

    config_class = SFTConfig

    def __init__(self, config: SFTConfig):
        super().__init__(config)

    def stages(self) -> list[Stage]:
        """返回 Stage 列表 (按执行顺序)."""
        return [
            SamplerStage(self.config),
            VerifierStage(self.config),
            FormatterStage(self.config),
        ]


if __name__ == "__main__":
    pass
