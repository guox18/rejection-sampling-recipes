"""Utility functions and client classes for SFT recipe."""

import asyncio
import os
from typing import Optional

import aiohttp

# ============================================================
# LLM Judge Prompt Template
# ============================================================

DEFAULT_JUDGE_TEMPLATE = """You are a helpful assistant who evaluates the correctness of models' outputs.
Please judge whether the candidate's answer matches the standard answer.

Evaluation criteria:
1. The standard answer is definitely correct. You only need to judge if the candidate's answer matches it.
2. Answers may be expressed differently but mean the same thing.
3. For multiple choice questions, the candidate needs to select the correct option(s).
4. Ignore formatting differences like \\boxed{{}}.

Grade the answer as:
A: CORRECT
B: INCORRECT

Just return "A" or "B", nothing else.

<Question>
{question}
</Question>

<Standard Answer>
{gold_target}
</Standard Answer>

<Candidate's Answer>
{predicted_answer}
</Candidate's Answer>

Your judgment (A or B):"""


# ============================================================
# Response processor: split thinking and final response
# ============================================================


def split_response(raw_response: str) -> tuple[str, str]:
    """
    Split raw response into thinking and final response.

    Supports multiple formats:
    - <think>...</think> format (R1, QwQ, etc.)
    - <|channel|>analysis<|message|> format (GPT-OSS, etc.)

    Args:
        raw_response: Model's raw output

    Returns:
        Tuple of (thinking, response):
        - thinking: The thinking/analysis process (empty string if none)
        - response: The final response for judging
    """
    thinking = ""
    response = raw_response.strip()

    # Format 1: <think>...</think> (R1, QwQ, etc.)
    if "<think>" in raw_response:
        if "</think>" in raw_response:
            parts = raw_response.split("</think>", 1)
            # Extract thinking content (remove <think> tag)
            thinking_part = parts[0]
            if "<think>" in thinking_part:
                thinking = thinking_part.split("<think>", 1)[-1].strip()
            else:
                thinking = thinking_part.strip()
            response = parts[-1].strip()
        else:
            # Truncated - thinking incomplete, no final response
            thinking = raw_response.replace("<think>", "").strip()
            response = ""

    # Format 2: GPT-OSS <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>
    elif "<|channel|>analysis<|message|>" in raw_response:
        separator = "<|end|><|start|>assistant<|channel|>final<|message|>"
        if separator in raw_response:
            parts = raw_response.split(separator, 1)
            # Extract thinking (remove channel tags)
            thinking_part = parts[0]
            if "<|channel|>analysis<|message|>" in thinking_part:
                thinking = thinking_part.split("<|channel|>analysis<|message|>", 1)[-1].strip()
            else:
                thinking = thinking_part.strip()
            response = parts[-1].strip()
        else:
            # Truncated - no final response
            thinking = raw_response.replace("<|channel|>analysis<|message|>", "").strip()
            response = ""

    # Fallback: check for </think> without <think> (some chat templates)
    elif "</think>" in raw_response:
        parts = raw_response.split("</think>", 1)
        thinking = parts[0].strip()
        response = parts[-1].strip()

    return thinking, response


def clip_thinking(raw_response: str) -> str:
    """
    Remove thinking process, return only final response.

    This is a convenience function that returns just the response part.

    Args:
        raw_response: Model's raw output

    Returns:
        Final response (for judging)
    """
    _, response = split_response(raw_response)
    return response


# ============================================================
# OpenAI-compatible API Clients
# ============================================================


class AsyncOpenAIClient:
    """
    异步 OpenAI-compatible API 客户端.

    特性:
    - 使用 aiohttp.ClientSession 实现异步调用
    - 内置重试机制
    - Session 在 batch 级别创建和回收
    - 信号量控制单个 session 的并发请求数
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        semaphore_size: int = 10,
    ):
        """
        初始化异步客户端.

        Args:
            api_key: API key (如果为 None, 从环境变量读取)
            base_url: API base URL (如果为 None, 使用 OpenAI 默认)
            max_retries: 最大重试次数
            semaphore_size: 单个 session 的并发请求数
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.max_retries = max_retries
        self.semaphore_size = semaphore_size

    async def chat_completion(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        messages: list[dict],
        model: str,
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> list[str]:
        """
        调用 chat completion API.

        Args:
            session: aiohttp session (由调用方在 batch 级别创建)
            semaphore: 并发控制信号量 (由调用方创建)
            messages: 对话消息列表
            model: 模型名称
            n: 生成的回复数量
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            生成的回复列表
        """
        async with semaphore:
            import json as json_module

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": model,
                "messages": messages,
                "n": n,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            # 记录请求大小（用于调试）
            payload_size = len(json_module.dumps(payload))
            if payload_size > 1_000_000:  # > 1MB
                print(f"[AsyncOpenAIClient] Large payload: {payload_size / 1_000_000:.2f} MB")

            url = f"{self.base_url}/chat/completions"

            # 重试逻辑
            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            # 检查响应的 Content-Type
                            content_type = resp.headers.get("Content-Type", "")

                            # 尝试解析 JSON，处理错误的 Content-Type
                            try:
                                data = await resp.json(content_type=None)  # 忽略 Content-Type 检查
                                return [choice["message"]["content"] for choice in data["choices"]]
                            except (ValueError, KeyError, aiohttp.ContentTypeError) as e:
                                # 获取原始响应文本用于调试
                                error_text = await resp.text()
                                raise RuntimeError(
                                    f"Failed to parse API response. "
                                    f"Content-Type: {content_type}, "
                                    f"Response preview: {error_text[:500]}"
                                )
                        else:
                            error_text = await resp.text()
                            if attempt == self.max_retries - 1:
                                raise RuntimeError(f"API error {resp.status}: {error_text}")
                            await asyncio.sleep(2**attempt)
                except aiohttp.ClientError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2**attempt)

            return []


class SyncOpenAIClient:
    """
    同步 OpenAI API 客户端.

    特性:
    - 使用 openai SDK 实现同步调用
    - 适合多线程模式
    - 封装初始化逻辑
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化同步客户端.

        Args:
            api_key: API key (如果为 None, 从环境变量读取)
            base_url: API base URL (如果为 None, 使用 OpenAI 默认)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        self.base_url = base_url
        self._client = None

    def initialize(self):
        """初始化 OpenAI client (在 Actor 的 initialize() 中调用)."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("pip install openai")

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 10,
    ) -> str:
        """
        调用 chat completion API.

        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            生成的回复内容
        """
        if self._client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
