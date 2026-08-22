"""IFBench recipe with parsing, feasibility checking, sampling, and constraint verification."""

from __future__ import annotations

import asyncio
import logging

import aiohttp

from src.base import BaseRecipe, Stage

from ._vendor import instructions_registry
from .config import TextSFTConfig
from .tools import (
    FEASIBILITY_TEMPLATE,
    AsyncOpenAIClient,
    SyncOpenAIClient,
    clip_thinking,
    extract_question_text,
    get_gold_answer,
    split_prompt_and_gold,
)

logger = logging.getLogger(__name__)


def build_instruction_constraints(
    prompt: str,
    instruction_id_list: list[str],
    instruction_kwargs: list[dict],
) -> list[str]:
    """Build human-readable constraints from IFBench instruction definitions."""
    constraints: list[str] = []
    for instruction_id, kwargs in zip(instruction_id_list, instruction_kwargs, strict=True):
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            continue
        instruction = instruction_cls(instruction_id)
        cleaned_kwargs = {key: value for key, value in (kwargs or {}).items() if value is not None}
        description = instruction.build_description(**cleaned_kwargs)
        instruction_args = instruction.get_instruction_args()
        if instruction_args and "prompt" in instruction_args:
            description = instruction.build_description(prompt=prompt)
        if description:
            constraints.append(description)
    return constraints


def extract_response_text(response_entry: str | dict) -> str:
    """Return response text from a sampler entry."""
    if isinstance(response_entry, dict):
        return response_entry.get("text", "")
    return response_entry


def extract_finish_reason(response_entry: str | dict) -> str | None:
    """Return finish_reason from a sampler entry."""
    if isinstance(response_entry, dict):
        return response_entry.get("finish_reason")
    return None


class PrepareStage(Stage):
    """Normalize IFBench input into messages and instruction metadata."""

    def __init__(self, config: TextSFTConfig):
        self.config = config

    def process_item(self, item: dict) -> dict:
        metadata = item.get("metadata") or {}
        prompt = item.get("prompt")
        messages = item.get("messages", [])

        if not prompt:
            prompt = extract_question_text(
                messages,
                joiner=self.config.text_joiner,
                strip_img_context=self.config.strip_img_context,
            )

        if not prompt:
            logger.warning("[PrepareStage] Item %s: no prompt found", item.get("id", "unknown"))
            return {**item, "_failed": True, "_error": "PrepareStage: no prompt found"}

        instruction_id_list = item.get("instruction_id_list") or metadata.get("instruction_id_list")
        instruction_kwargs = item.get("kwargs") or metadata.get("instruction_kwargs")

        if instruction_id_list is None or instruction_kwargs is None:
            return {
                **item,
                "_failed": True,
                "_error": "PrepareStage: missing instruction_id_list or kwargs",
            }

        if len(instruction_id_list) != len(instruction_kwargs):
            return {
                **item,
                "_failed": True,
                "_error": "PrepareStage: instruction_id_list and kwargs length mismatch",
            }

        if not messages:
            messages = [{"role": "user", "content": prompt}]

        item_id = item.get("id") or item.get("key")
        if item_id is not None:
            item = {**item, "id": item_id}

        metadata["instruction_id_list"] = instruction_id_list
        metadata["instruction_kwargs"] = instruction_kwargs
        metadata["parsed_constraints"] = build_instruction_constraints(
            prompt,
            instruction_id_list,
            instruction_kwargs,
        )
        return {**item, "prompt": prompt, "messages": messages, "metadata": metadata}


@Stage.threaded_mode
class FeasibilityStage(Stage):
    """Check if task is feasible under the given constraints."""

    def __init__(self, config: TextSFTConfig):
        self.config = config
        self.client: SyncOpenAIClient | None = None

    def initialize(self) -> None:
        self._thread_pool_size = self.config.feasibility_max_workers
        api_key = self.config.feasibility_api_key or self.config.api_key
        base_url = self.config.feasibility_base_url or self.config.base_url
        self.client = SyncOpenAIClient(api_key=api_key, base_url=base_url)
        self.client.initialize()

    def process_item(self, item: dict) -> dict:
        messages = item.get("messages", [])
        metadata = item.get("metadata") or {}
        constraints = metadata.get("parsed_constraints", [])

        # 使用原始 user content 作为 query
        query = extract_question_text(
            messages,
            joiner=self.config.text_joiner,
            strip_img_context=self.config.strip_img_context,
        )

        if not query:
            logger.warning(
                "[FeasibilityStage] Item %s: no user content found", item.get("id", "unknown")
            )
            return {**item, "_failed": True, "_error": "FeasibilityStage: no user content"}

        # If no constraints, pass through
        if not constraints:
            logger.warning(
                "[FeasibilityStage] Item %s: no constraints found", item.get("id", "unknown")
            )
            return {**item, "_failed": True, "_error": "FeasibilityStage: no constraints"}

        # Format constraints for prompt
        constraints_text = "\n".join(f"- {c}" for c in constraints)

        prompt = FEASIBILITY_TEMPLATE.format(query=query, constraints=constraints_text)
        try:
            model = self.config.feasibility_model or self.config.model
            responses = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=self.config.feasibility_temperature,
                max_tokens=self.config.feasibility_max_tokens,
                n=1,
            )
            raw_response = responses[0] if responses else ""

            # Clip thinking content for judgment
            clipped_response = clip_thinking(raw_response)

            # Parse pass/fail
            is_feasible = "PASS" in clipped_response.upper()

            # 保存判断日志到 metadata (包括完整 think 内容)
            metadata["feasibility_pass"] = is_feasible
            metadata["feasibility_prompt"] = prompt  # 完整响应包含 think
            metadata["feasibility_raw_response"] = raw_response  # 完整响应包含 think

            if not is_feasible:
                logger.info(
                    "[FeasibilityStage] Item %s: query infeasible under constraints",
                    item.get("id", "unknown"),
                )
                return {**item, "metadata": metadata, "_failed": True, "_error": "Infeasible"}

            return {**item, "metadata": metadata}

        except Exception as exc:
            import traceback

            error_trace = traceback.format_exc()
            logger.error(
                "[FeasibilityStage] ❌ Item %s failed: %s\n%s",
                item.get("id", "unknown"),
                exc,
                error_trace,
            )
            return {
                **item,
                "_failed": True,
                "_error": f"FeasibilityStage: {exc}",
                "_traceback": error_trace,
            }


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
                    response_meta = [
                        {
                            "finish_reason": extract_finish_reason(response),
                            "text": extract_response_text(response),
                        }
                        for response in responses
                    ]
                    metadata["response_meta"] = response_meta
                    return {
                        **item,
                        "messages": prompt_messages,
                        "metadata": metadata,
                        "responses": responses,
                    }
                except Exception as exc:
                    import traceback

                    error_trace = traceback.format_exc()
                    logger.error(
                        "[SamplerStage] ❌ Item %s failed: %s\n%s",
                        item.get("id", "unknown"),
                        exc,
                        error_trace,
                    )
                    return {
                        **item,
                        "_failed": True,
                        "_error": f"SamplerStage: {exc}",
                        "_traceback": error_trace,
                    }

            return await asyncio.gather(*[process_one(item) for item in batch])


@Stage.threaded_mode
class ScriptVerifierStage(Stage):
    """Verify instruction following with the IFBench script checkers."""

    def __init__(self, config: TextSFTConfig):
        self.config = config

    def initialize(self) -> None:
        self._thread_pool_size = self.config.verifier_max_workers

    def _check_instruction(
        self, prompt: str, response: str, instruction_id: str, kwargs: dict
    ) -> dict:
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            return {
                "instruction_id": instruction_id,
                "pass": False,
                "description": "",
                "kwargs": kwargs,
                "error": "Unknown instruction_id",
            }

        instruction = instruction_cls(instruction_id)
        cleaned_kwargs = {key: value for key, value in (kwargs or {}).items() if value is not None}
        description = instruction.build_description(**cleaned_kwargs)
        instruction_args = instruction.get_instruction_args()
        if instruction_args and "prompt" in instruction_args:
            description = instruction.build_description(prompt=prompt)

        is_pass = bool(response.strip()) and instruction.check_following(response)
        return {
            "instruction_id": instruction_id,
            "pass": is_pass,
            "description": description,
            "kwargs": cleaned_kwargs,
        }

    def process_item(self, item: dict) -> dict:
        responses = item.get("responses", [])
        metadata = item.get("metadata") or {}
        instruction_id_list = (
            metadata.get("instruction_id_list") or item.get("instruction_id_list") or []
        )
        instruction_kwargs = metadata.get("instruction_kwargs") or item.get("kwargs") or []
        prompt = item.get("prompt") or extract_question_text(
            item.get("messages", []),
            joiner=self.config.text_joiner,
            strip_img_context=self.config.strip_img_context,
        )
        item_id = item.get("id", "unknown")

        if not instruction_id_list:
            rollouts = []
            for response in responses:
                response_text = extract_response_text(response)
                finish_reason = extract_finish_reason(response)
                rollouts.append(
                    {
                        "response": response_text,
                        "finish_reason": finish_reason,
                        "score": 1.0 if finish_reason == "stop" else 0.0,
                        "constraint_results": [],
                    }
                )
            result = {k: v for k, v in item.items() if k != "responses"}
            result["rollouts"] = rollouts
            return result

        if len(instruction_id_list) != len(instruction_kwargs):
            return {
                **item,
                "_failed": True,
                "_error": "ScriptVerifierStage: instruction_id_list and kwargs length mismatch",
            }

        rollouts = []
        verify_samples = []

        for response in responses:
            response_text = extract_response_text(response)
            finish_reason = extract_finish_reason(response)
            clipped_response = clip_thinking(response_text)
            constraint_results = []
            all_pass = True

            if finish_reason != "stop":
                all_pass = False
                for instruction_id, kwargs in zip(
                    instruction_id_list, instruction_kwargs, strict=True
                ):
                    constraint_results.append(
                        {
                            "instruction_id": instruction_id,
                            "pass": False,
                            "description": "",
                            "kwargs": kwargs,
                            "error": "truncated",
                        }
                    )
            else:
                for instruction_id, kwargs in zip(
                    instruction_id_list, instruction_kwargs, strict=True
                ):
                    try:
                        result = self._check_instruction(
                            prompt, clipped_response, instruction_id, kwargs
                        )
                        constraint_results.append(result)
                        if not result.get("pass"):
                            all_pass = False
                    except Exception as exc:
                        import traceback

                        error_trace = traceback.format_exc()
                        logger.warning(
                            "[ScriptVerifierStage] Item %s: instruction check failed: %s\n%s",
                            item_id,
                            exc,
                            error_trace,
                        )
                        constraint_results.append(
                            {
                                "instruction_id": instruction_id,
                                "pass": False,
                                "description": "",
                                "kwargs": kwargs,
                                "error": str(exc),
                                "traceback": error_trace,
                            }
                        )
                        all_pass = False

            rollouts.append(
                {
                    "response": response_text,
                    "finish_reason": finish_reason,
                    "score": 1.0 if all_pass else 0.0,
                    "constraint_results": constraint_results,
                }
            )

        if rollouts:
            first_rollout = rollouts[0]
            verify_samples.append(
                {
                    "response": clip_thinking(first_rollout["response"]),
                    "finish_reason": first_rollout.get("finish_reason"),
                    "instruction_id_list": instruction_id_list,
                    "constraint_results": first_rollout.get("constraint_results", []),
                    "overall_pass": first_rollout.get("score", 0) >= 1.0,
                }
            )

        result = {k: v for k, v in item.items() if k != "responses"}
        if result.get("metadata") is None:
            result["metadata"] = metadata
        result["metadata"]["verify_samples"] = verify_samples
        result["rollouts"] = rollouts
        return result


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

        # Fix missing <think> prefix when </think> is present
        if "</think>" in best_response and "<think>" not in best_response:
            best_response = "<think>" + best_response

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
    """IFBench recipe: prepare -> feasibility -> sample -> verify -> format."""

    def stages(self) -> list[Stage]:
        return [
            PrepareStage(self.config),
            FeasibilityStage(self.config),
            SamplerStage(self.config),
            ScriptVerifierStage(self.config),
            FormatterStage(self.config),
        ]


if __name__ == "__main__":
    config = TextSFTConfig.from_yaml("config.yaml")
    recipe = TextSFTRecipe(config)
    prepare_stage = recipe.stages()[0]
    prepare_stage.initialize()

    import json

    with open(
        "/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/data/P~Instruct~en~IF_nlp~1.0.0~0.0/jsonl/part-694b932b7c79-000086_test10.jsonl"
    ) as f:
        input_list = [json.loads(line) for line in f]
    for i, item in enumerate(input_list):
        result = prepare_stage.process_item(item)
        if result.get("_failed"):
            print(f"Failed to process item {i + 1}")
            print(result)
        else:
            print(f"Processed {i + 1} items")
            print(result)
