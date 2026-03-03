"""Helpers for the simple text-only SFT recipe."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import aiohttp

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

logger = logging.getLogger(__name__)

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

# 可行性判断 prompt: 判断 constraints 下 query 是否可能完成
FEASIBILITY_TEMPLATE = """你是一个逻辑分析助手。请判断在给定的约束条件下，用户的任务是否有可能完成。

你只需要排除明显不合理的组合，例如:
- 约束要求回复只包含几个字母，但任务要求写一篇文章
- 约束要求使用特定语言，但任务内容与该语言不兼容
- 约束之间相互矛盾

如果任务在约束下基本可行，即使有一定难度，也应该判断为 PASS。

<Query>
{query}
</Query>

<Constraints>
{constraints}
</Constraints>

请判断这个任务在给定约束下是否可行:
- 如果可行，回复: PASS
- 如果明显不合理，回复: FAIL

你的判断 (PASS 或 FAIL):"""

# 单个 constraint 验证 prompt
CONSTRAINT_VERIFY_TEMPLATE = """你是一个严格的约束验证助手。请判断给定的回复是否满足指定的约束条件。

<Constraint>
{constraint}
</Constraint>

<Response>
{response}
</Response>

请仔细检查回复是否满足这个约束条件:
- 如果满足约束，回复: PASS
- 如果不满足约束，回复: FAIL

你的判断 (PASS 或 FAIL):"""


def _remove_trailing_commas(json_str: str) -> str:
    """移除 JSON 字符串中的尾随逗号"""
    import re

    json_str = re.sub(r",\s*]", "]", json_str)
    json_str = re.sub(r",\s*}", "}", json_str)
    return json_str


# 尝试导入 regex 模块（支持递归匹配）
try:
    import regex

    _REGEX_AVAILABLE = True
except ImportError:
    _REGEX_AVAILABLE = False


def extract_json_from_response(response: str) -> dict | None:
    """Extract JSON from model response, handling markdown code blocks and common errors.

    当存在多个 JSON 时，默认返回最后一个。
    """
    import ast
    import json
    import re

    def _try_parse_json(json_string: str) -> dict | None:
        """尝试多种方式解析 JSON 字符串"""
        json_string = json_string.strip()
        json_string_clean = _remove_trailing_commas(json_string)

        # 尝试标准 JSON 解析
        for candidate in [json_string_clean, json_string, json_string.replace("'", '"')]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # 尝试使用 ast.literal_eval
        try:
            parsed = ast.literal_eval(json_string)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            pass

        # 尝试使用 json5（如果可用）
        try:
            import json5

            return json5.loads(json_string_clean)
        except Exception:
            pass

        return None

    if not response:
        return None

    llm_output = response

    # 预处理：移除特殊标记和控制字符
    llm_output = re.sub(r"<\|.*?\|>", "", llm_output)  # 移除<|im_end|>等标记
    llm_output = re.sub(r"[\x00-\x1F\x7F]", "", llm_output)  # 移除控制字符

    # 处理中文引号和逗号，以及特殊连字符
    llm_output = llm_output.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    llm_output = re.sub(r"，", ",", llm_output)
    llm_output = llm_output.replace("‑", "-")  # 处理特殊连字符（U+2011）

    # 尝试从 markdown 代码块中提取 JSON
    json_pattern = r"```(?:json)?\s*({[\s\S]*?})\s*```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # 回退模式：使用 regex 模块进行递归匹配（如果可用）
        if _REGEX_AVAILABLE:
            try:
                json_pattern = r"({(?:[^{}]*|(?R))*})"
                matches = regex.findall(json_pattern, llm_output, regex.DOTALL)
            except Exception:
                matches = []

    if not matches:
        # 终极回退：提取疑似 JSON 结构
        matches = re.findall(r"{.+}", llm_output, re.DOTALL)

    if not matches:
        return None

    # 从最后一个开始尝试解析，返回第一个成功解析的
    for json_string in reversed(matches):
        parsed = _try_parse_json(json_string)
        if parsed is not None:
            return parsed

    return None


def clean_text(text: str, strip_img_context: bool = True) -> str:
    """Normalize a text string."""
    if not text:
        return ""
    cleaned = text
    if strip_img_context:
        cleaned = cleaned.replace(IMG_CONTEXT_TOKEN, "")
    return cleaned.strip()


def extract_text_from_content(content, joiner: str = "\n", strip_img_context: bool = True) -> str:
    """Extract text from message content.

    Supports:
    - str content
    - list content with {"type": "text", "text": "..."}
    - dict content with {"type": "text", "text": "..."}
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return clean_text(content, strip_img_context=strip_img_context)

    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    texts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if text:
                    texts.append(text)
        joined = joiner.join(texts)
        return clean_text(joined, strip_img_context=strip_img_context)

    if isinstance(content, dict) and content.get("type") == "text":
        return clean_text(content.get("text", ""), strip_img_context=strip_img_context)

    return ""


def normalize_messages(
    messages: list[dict],
    joiner: str = "\n",
    strip_img_context: bool = True,
    drop_empty: bool = True,
) -> list[dict]:
    """Normalize messages to text-only format."""
    normalized: list[dict] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        text = extract_text_from_content(
            message.get("content"),
            joiner=joiner,
            strip_img_context=strip_img_context,
        )
        if drop_empty and not text:
            continue
        normalized.append({"role": role, "content": text})
    return normalized


def split_prompt_and_gold(
    messages: list[dict],
    joiner: str = "\n",
    strip_img_context: bool = True,
    drop_empty: bool = True,
) -> tuple[list[dict], str]:
    """Split prompt messages and the gold answer (if the last message is assistant)."""
    if not messages:
        return [], ""

    gold = ""
    if isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        gold = extract_text_from_content(
            messages[-1].get("content"),
            joiner=joiner,
            strip_img_context=strip_img_context,
        )
        prompt_messages = messages[:-1]
    else:
        prompt_messages = messages

    normalized = normalize_messages(
        prompt_messages,
        joiner=joiner,
        strip_img_context=strip_img_context,
        drop_empty=drop_empty,
    )
    return normalized, gold


def extract_question_text(
    messages: list[dict],
    joiner: str = "\n",
    strip_img_context: bool = True,
) -> str:
    """Extract the last user text from messages."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        return extract_text_from_content(
            msg.get("content"),
            joiner=joiner,
            strip_img_context=strip_img_context,
        )
    return ""


def get_gold_answer(item: dict) -> str:
    """Get gold answer from metadata or fallback to last assistant message."""
    metadata = item.get("metadata") or {}
    for key in ("short_answer", "answer", "gold_target", "gold_answer"):
        value = metadata.get(key)
        if value:
            return value

    messages = item.get("messages") or []
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        return extract_text_from_content(messages[-1].get("content"))
    return ""


def clip_thinking(raw_response: str) -> str:
    """Remove common thinking tags and return the final response."""
    if not raw_response:
        return ""

    if "<think>" in raw_response:
        if "</think>" in raw_response:
            return raw_response.split("</think>", 1)[-1].strip()
        return ""

    if "<|channel|>analysis<|message|>" in raw_response:
        separator = "<|end|><|start|>assistant<|channel|>final<|message|>"
        if separator in raw_response:
            return raw_response.split(separator, 1)[-1].strip()
        return ""

    if "</think>" in raw_response:
        return raw_response.split("</think>", 1)[-1].strip()

    return raw_response.strip()


class SyncOpenAIClient:
    """Small OpenAI-compatible sync client wrapper."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        self.base_url = base_url
        self._client = None

    def initialize(self) -> None:
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_completion(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        n: int = 1,
    ) -> list[str]:
        if self._client is None:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        response = self._client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
        )
        return [choice.message.content or "" for choice in response.choices]


class AsyncOpenAIClient:
    """OpenAI-compatible async client using aiohttp."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        semaphore_size: int = 10,
    ) -> None:
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
    ) -> list[dict]:
        async with semaphore:
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
            url = f"{self.base_url}/chat/completions"

            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            return [
                                {
                                    "text": choice["message"]["content"] or "",
                                    "finish_reason": choice.get("finish_reason"),
                                }
                                for choice in data.get("choices", [])
                            ]

                        error_text = await resp.text()
                        if attempt < self.max_retries - 1:
                            logger.warning(
                                "[AsyncOpenAIClient] API error %s on attempt %s/%s: %s",
                                resp.status,
                                attempt + 1,
                                self.max_retries,
                                error_text[:200],
                            )
                            await asyncio.sleep(2**attempt)
                        else:
                            raise RuntimeError(f"API error {resp.status}: {error_text}")
                except aiohttp.ClientError as exc:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            "[AsyncOpenAIClient] Connection error on attempt %s/%s: %s",
                            attempt + 1,
                            self.max_retries,
                            exc,
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        raise

            return []
