"""Utility functions and client classes for SFT recipe."""

import asyncio
import logging
import os
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ============================================================
# LLM Parse Prompt Template
# ============================================================
EXTRACT_ANSWER_TEMPLATE = """You are a helpful assistant tasked with extracting the final concise answer from a model's output.
Please extract the `short_answer` based on the provided `question` and `answer`.

Extraction criteria:
1. **Remove Reasoning:** Strip away all analysis, calculation steps, or chain-of-thought. We only need the final result.
2. **Preserve Integrity:** Do not attempt to verify or correct the answer. Even if the reasoning logic appears flawed, extract the final conclusion exactly as stated.
3. **Complex Formats:** If the final answer is a code block (e.g., Mermaid, Python), a JSON object, or a list, extract the entire block/object without modification.
4. **Key Patterns:** Look for phrases like "The answer is...", "\\boxed{{...}}", or "Therefore...".
5. **Multiple Choice:** For selection questions, extract the chosen option (e.g., "A" or "Option C").

Just return the extracted content, nothing else.

<Question>
{question_text}
</Question>

<Model Answer>
{answer_text}
</Model Answer>

Extracted Short Answer:"""


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
    Async OpenAI-compatible API client.

    Features:
    - async calls via aiohttp.ClientSession
    - built-in retry
    - session created/closed per batch
    - semaphore limits concurrent requests per session
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        semaphore_size: int = 10,
    ):
        """
        Initialize async client.

        Args:
            api_key: API key (if None, read from env)
            base_url: API base URL (if None, use OpenAI default)
            max_retries: max retry attempts
            semaphore_size: concurrent requests per session
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
        Call the chat completion API.

        Args:
            session: aiohttp session (created by caller per batch)
            semaphore: concurrency semaphore (created by caller)
            messages: chat messages
            model: model name
            n: number of responses
            temperature: sampling temperature
            max_tokens: max tokens

        Returns:
            list of responses
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

            # Log payload size for debugging.
            payload_size = len(json_module.dumps(payload))
            if payload_size > 1_000_000:  # > 1MB
                logger.warning(
                    f"[AsyncOpenAIClient] Large payload: {payload_size / 1_000_000:.2f} MB"
                )

            url = f"{self.base_url}/chat/completions"

            # Retry logic.
            # Note: 413 (request too large) is not retried; caller should resize images.
            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status == 200:
                            # Success: parse JSON.
                            content_type = resp.headers.get("Content-Type", "")

                            try:
                                data = await resp.json(content_type=None)  # ignore Content-Type
                                if attempt > 0:
                                    logger.info(
                                        f"[AsyncOpenAIClient] Request succeeded after {attempt + 1} attempt(s)"
                                    )
                                return [choice["message"]["content"] for choice in data["choices"]]
                            except (ValueError, KeyError, aiohttp.ContentTypeError) as e:
                                error_text = await resp.text()
                                raise RuntimeError(
                                    f"Failed to parse API response. "
                                    f"Content-Type: {content_type}, "
                                    f"Response preview: {error_text[:500]}"
                                ) from e

                        elif resp.status == 413:
                            # 413: request too large, raise immediately (caller should resize).
                            error_text = await resp.text()
                            logger.warning(
                                f"[AsyncOpenAIClient] 413 error (request too large), needs input length adjustment: {error_text[:200]}"
                            )
                            raise RuntimeError(f"API error {resp.status}: {error_text}")

                        else:
                            # Other HTTP errors: retry.
                            error_text = await resp.text()
                            if attempt < self.max_retries - 1:
                                logger.warning(
                                    f"[AsyncOpenAIClient] API error {resp.status} on attempt {attempt + 1}/{self.max_retries}, "
                                    f"retrying in {2**attempt}s... Error: {error_text[:200]}"
                                )
                                await asyncio.sleep(2**attempt)
                            else:
                                logger.error(
                                    f"[AsyncOpenAIClient] API error {resp.status} after {self.max_retries} attempts"
                                )
                                raise RuntimeError(f"API error {resp.status}: {error_text}")

                except aiohttp.ClientError as e:
                    # Connection errors (e.g. 104 reset): retry.
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"[AsyncOpenAIClient] Connection error on attempt {attempt + 1}/{self.max_retries}: {type(e).__name__}: {e}, "
                            f"retrying in {2**attempt}s..."
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            f"[AsyncOpenAIClient] Connection error after {self.max_retries} attempts: {type(e).__name__}: {e}"
                        )
                        raise

            return []


class SyncOpenAIClient:
    """
    Sync OpenAI API client.

    Features:
    - synchronous calls via openai SDK
    - suitable for threaded mode
    - wraps initialization logic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize sync client.

        Args:
            api_key: API key (if None, read from env)
            base_url: API base URL (if None, use OpenAI default)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        self.base_url = base_url
        self._client = None

    def initialize(self):
        """Initialize OpenAI client (called from Actor initialize())."""
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
        Call the chat completion API.

        Args:
            messages: chat messages
            model: model name
            temperature: sampling temperature
            max_tokens: max tokens

        Returns:
            response content
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
