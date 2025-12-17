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

注意: 
- _resume_id 字段由 Pipeline 框架自动管理，用于断点续传
- 用户无需手动添加或处理 _resume_id，它是框架内部实现细节
"""

import asyncio
import os

import aiohttp

from src.base import BaseRecipe, Stage

from .config import SFTConfig
from .tools import AsyncOpenAIClient, SyncOpenAIClient, DEFAULT_JUDGE_TEMPLATE, clip_thinking


# ============================================================
# 示例 1: 异步模式 - 使用装饰器 (推荐)
# ============================================================

@Stage.async_mode
class SamplerStage(Stage):
    """
    采样阶段: 为每个 item 生成 n 个 responses.
    
    模式: 异步模式 (@Stage.async_mode)
    - 覆盖 process() 方法，在 batch 级别管理 session
    - Session 和 semaphore 在每个 batch 创建，处理完后关闭
    - 避免事件循环不一致问题，代码更简洁
    """
    
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
        处理一个 batch，在 batch 级别创建和管理 session.
        """
        # 创建 batch 级别的 session 和 semaphore
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(self.client.semaphore_size)
            
            # 并发处理 batch 内的所有 items
            async def process_one(item: dict) -> dict:
                if item.get("_failed"):
                    return item
                
                try:
                    messages = item.get("messages", [])
                    item_id = item.get("id", "unknown")
                    
                    # 调用 API
                    responses = await self.client.chat_completion(
                        session=session,
                        semaphore=semaphore,
                        messages=messages,
                        model=self.config.model,
                        n=self.config.n_samples,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    print(f"[SamplerStage] Item {item_id}: Generated {len(responses)} responses")
                    
                    return {**item, "responses": responses}
                except Exception as e:
                    print(f"[SamplerStage] Item {item.get('id', 'unknown')} failed: {e}")
                    return {**item, "_failed": True, "_error": f"SamplerStage: {e}"}
            
            return await asyncio.gather(*[process_one(item) for item in batch])


# ============================================================
# 示例 2: 多线程模式 - 使用装饰器 
# ============================================================

@Stage.threaded_mode
class VerifierStage(Stage):
    """
    验证阶段: 使用 LLM-as-Judge 为每个 response 打分.
    
    模式: 多线程
    - 使用 @Stage.threaded_mode 装饰器
    - 在 initialize() 中通过设置 self._thread_pool_size 指定线程池大小
    - 实现 process_item(), 框架自动并发处理 batch 内的多个 item
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: SyncOpenAIClient = None
    
    def initialize(self):
        """
        Actor 创建时调用一次, 设置线程池大小并初始化客户端.
        
        通过设置 self._thread_pool_size 指定线程池大小.
        """
        self._thread_pool_size = self.config.verifier_max_workers
        
        # 初始化同步客户端
        api_key = self.config.judge_api_key or self.config.api_key
        base_url = self.config.judge_base_url or self.config.base_url
        
        self.client = SyncOpenAIClient(api_key=api_key, base_url=base_url)
        self.client.initialize()
    
    def process_item(self, item: dict) -> dict:
        """
        处理单个 item, 为所有 responses 打分.
        
        框架保证:
        - 自动用线程池并发处理 batch 内的 item (由 verifier_max_workers 控制)
        - 自动捕获异常, 失败的 item 标记 _failed=True
        - 无需手动 try-catch
        """
        responses = item.get("responses", [])
        metadata = item.get("metadata", {})
        messages = item.get("messages", [])
        item_id = item.get("id", "unknown")
        
        print(f"[VerifierStage] Item {item_id}: Verifying {len(responses)} responses")
        
        # 提取问题
        question = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    question = msg.get("content", "")
                    break
        
        # 验证所有 responses
        rollouts = self._verify_llm_judge(responses, metadata, question)
        
        # 返回结果
        result = {k: v for k, v in item.items() if k != "responses"}
        result["rollouts"] = rollouts
        return result
    
    def _verify_llm_judge(self, responses: list[str], metadata: dict, question: str) -> list[dict]:
        """
        使用 LLM Judge 验证多个 responses.
        
        对 1 个 item 的 N 个 responses, 顺序调用 judge 验证.
        """
        if not responses:
            return []
        
        gold_target = metadata.get("answer") or metadata.get("gold_target", "")
        rollouts = []
        
        for response in responses:
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
            except Exception as e:
                print(f"⚠️  Judge error: {e}")
                score = 0.0
            
            rollouts.append({"response": response, "score": score})
        
        return rollouts
    
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
    
    模式: 同步
    - 只实现 process_item()
    - 框架自动处理异常, 顺序执行
    - 适合: 轻量计算, 无需并发
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config
    
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
        
        # 构建 SFT 格式
        sft_messages = messages + [{"role": "assistant", "content": best_response}]
        result = {}
        if "id" in item:
            result["id"] = item["id"]

        result = {
            **result,
            "_resume_id": item.get("_resume_id"),
            "messages": sft_messages,
            "metadata": {
                **metadata,
                "n_passed": len(passed),
                "n_total": len(rollouts),
                "used_ground_truth": used_gt,
            },
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
