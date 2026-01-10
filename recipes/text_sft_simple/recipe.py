"""Simple text-only SFT recipe with sampling, verification, and formatting."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from src.base import BaseRecipe, Stage

from .config import TextSFTConfig
from .tools import (
    DEFAULT_JUDGE_TEMPLATE,
    AsyncOpenAIClient,
    SyncOpenAIClient,
    clip_thinking,
    extract_question_text,
    get_gold_answer,
    split_prompt_and_gold,
)

logger = logging.getLogger(__name__)


class SamplerStage(Stage):
    """Sample candidate answers for each item."""

    def __init__(self, config: TextSFTConfig):
        self.config = config
        self.client: AsyncOpenAIClient | None = None

    def initialize(self) -> None:
        self.client = AsyncOpenAIClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
            semaphore_size=self.config.semaphore_per_sampler,
        )

    async def process(self, batch: list[dict]) -> list[dict]:
        timeout = aiohttp.ClientTimeout(total=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            semaphore = asyncio.Semaphore(self.client.semaphore_size)

            async def process_one(item: dict) -> dict:
                if item.get("_failed") is True:
                    return item

                messages = item.get("messages", [])
                prompt_messages, gold_from_messages = split_prompt_and_gold(
                    messages,
                    joiner=self.config.text_joiner,
                    strip_img_context=self.config.strip_img_context,
                    drop_empty=self.config.drop_empty_messages,
                )

                metadata = item.get("metadata") or {}
                if gold_from_messages:
                    has_gold = any(
                        metadata.get(key)
                        for key in ("short_answer", "answer", "gold_target", "gold_answer")
                    )
                    if not has_gold:
                        metadata["short_answer"] = gold_from_messages

                try:
                    responses = await self.client.chat_completion(
                        session=session,
                        semaphore=semaphore,
                        messages=prompt_messages,
                        model=self.config.model,
                        n=self.config.n_samples,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    return {
                        **item,
                        "messages": prompt_messages,
                        "metadata": metadata,
                        "responses": responses,
                    }
                except Exception as exc:
                    logger.error(
                        "[SamplerStage] ❌ Item %s failed: %s",
                        item.get("id", "unknown"),
                        exc,
                    )
                    return {**item, "_failed": True, "_error": f"SamplerStage: {exc}"}

            return await asyncio.gather(*[process_one(item) for item in batch])


@Stage.threaded_mode
class VerifierStage(Stage):
    """Use an LLM judge to score candidate answers."""

    def __init__(self, config: TextSFTConfig):
        self.config = config
        self.client: SyncOpenAIClient | None = None

    def initialize(self) -> None:
        self._thread_pool_size = self.config.verifier_max_workers
        api_key = self.config.judge_api_key or self.config.api_key
        base_url = self.config.judge_base_url or self.config.base_url
        self.client = SyncOpenAIClient(api_key=api_key, base_url=base_url)
        self.client.initialize()

    def process_item(self, item: dict) -> dict:
        responses = item.get("responses", [])
        messages = item.get("messages", [])
        metadata = item.get("metadata") or {}

        gold_target = get_gold_answer(item)
        if not gold_target:
            raise ValueError("Gold answer missing: set metadata.short_answer or include assistant.")

        question = extract_question_text(
            messages,
            joiner=self.config.text_joiner,
            strip_img_context=self.config.strip_img_context,
        )

        rollouts = []
        for response in responses:
            clipped_response = clip_thinking(response)
            prompt = DEFAULT_JUDGE_TEMPLATE.format(
                question=question,
                gold_target=gold_target,
                predicted_answer=clipped_response,
            )
            judge_output = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.config.judge_model or self.config.model,
                temperature=self.config.judge_temperature,
                max_tokens=self.config.judge_max_tokens,
                n=1,
            )
            is_correct = self._parse_judge_output(judge_output[0] if judge_output else "")
            rollouts.append({"response": response, "score": 1.0 if is_correct else 0.0})

        result = {k: v for k, v in item.items() if k != "responses"}
        if result.get("metadata") is None:
            result["metadata"] = metadata
        result["rollouts"] = rollouts
        return result

    @staticmethod
    def _parse_judge_output(output: str) -> bool:
        if not output:
            return False
        cleaned = output.strip().lower()
        return cleaned in ("a", "correct")


class FormatterStage(Stage):
    """Select a response and emit the final SFT example."""

    def __init__(self, config: TextSFTConfig):
        self.config = config

    def process_item(self, item: dict) -> dict:
        messages = item.get("messages", [])
        rollouts = item.get("rollouts") or []
        metadata = item.get("metadata") or {}
        item_id = item.get("id", "unknown")

        passed = [r for r in rollouts if r.get("score", 0) >= self.config.pass_threshold]
        if passed:
            best_response = passed[0]["response"]
            used_ground_truth = False
        else:
            gold_target = get_gold_answer(item)
            if not gold_target:
                logger.warning("[FormatterStage] Item %s: no valid response", item_id)
                return {**item, "_failed": True, "_error": "No response passed and no gold answer"}
            best_response = gold_target
            used_ground_truth = True

        sft_messages = messages + [{"role": "assistant", "content": best_response}]
        metadata.update(
            {
                "n_passed": len(passed),
                "n_total": len(rollouts),
                "used_ground_truth": used_ground_truth,
            }
        )

        result = {**item, "messages": sft_messages, "metadata": metadata}

        ordered_result = {
            "id": result.get("id"),
            "messages": result.get("messages", []),
            "metadata": result.get("metadata", {}),
        }
        for key in result:
            if key not in ["id", "messages", "metadata"]:
                ordered_result[key] = result[key]
        return ordered_result


class TextSFTRecipe(BaseRecipe):
    """Text-only recipe: sample -> verify -> format."""

    def stages(self) -> list[Stage]:
        return [
            SamplerStage(self.config),
            VerifierStage(self.config),
            FormatterStage(self.config),
        ]
