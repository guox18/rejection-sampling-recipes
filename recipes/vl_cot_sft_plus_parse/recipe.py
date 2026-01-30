import asyncio
import base64
import logging
import os
from io import BytesIO

import aiohttp
from PIL import Image

from src.base import BaseRecipe, Stage
from src.utils import get_nested_value
from src.utils.qwen3_vl_util import IMAGE_MAX_TOKEN_NUM, SPATIAL_MERGE_SIZE, smart_resize

from .config import SFTConfig
from .tools import (
    DEFAULT_JUDGE_TEMPLATE,
    EXTRACT_ANSWER_TEMPLATE,
    AsyncOpenAIClient,
    SyncOpenAIClient,
    clip_thinking,
)

logger = logging.getLogger(__name__)


def resize_messages_images(messages: list[dict], max_pixels: int = None) -> list[dict]:
    """
    Resize all images inside messages.

    Args:
        messages: chat messages list
        max_pixels: max pixel count (defaults to IMAGE_MAX_TOKEN_NUM * factor^2)

    Returns:
        processed messages (modified in place and returned)
    """
    # Compute default max_pixels.
    patch_factor = int(14 * SPATIAL_MERGE_SIZE)
    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * patch_factor**2
    import copy

    def to_rgb(image: Image.Image) -> Image.Image:
        """Convert image to RGB."""
        if image.mode == "RGB":
            return image
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background.convert("RGB")

    def process_base64_image(
        base64_str: str, max_pixels: int, patch_factor: int, max_base64_kb: int = 800
    ) -> str:
        """
        Process a base64-encoded image and keep the final size under a limit.

        Args:
            base64_str: original base64 string
            max_pixels: max pixels (used for initial resize)
            patch_factor: patch factor (must be a multiple)
            max_base64_kb: max base64 size in KB (default 800KB)
        """
        try:
            # Parse base64 string.
            if "base64," in base64_str:
                header, base64_data = base64_str.split("base64,", 1)
            else:
                return base64_str  # Not base64, skip.

            # Decode image.
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))

            # Convert to RGB.
            image = to_rgb(image_obj)
            width, height = image.size
            original_size_mb = len(data) / (1024 * 1024)

            # Compute new size.
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_factor,
                min_pixels=4 * patch_factor**2,
                max_pixels=max_pixels,
            )

            # Resize.
            resized_image = image.resize((resized_width, resized_height))

            # Adjust JPEG quality to keep base64 size under the limit.
            max_base64_bytes = max_base64_kb * 1024
            quality = 85
            attempts = 0

            while quality >= 20 and attempts < 10:
                buffer = BytesIO()
                resized_image.save(buffer, format="JPEG", quality=quality)
                jpeg_bytes = buffer.getvalue()

                # Base64 size is roughly 4/3 of the binary size.
                estimated_base64_size = len(jpeg_bytes) * 4 / 3

                if estimated_base64_size <= max_base64_bytes:
                    # Size is within limit.
                    jpeg_size_kb = len(jpeg_bytes) / 1024
                    new_base64_data = base64.b64encode(jpeg_bytes).decode()
                    base64_size_kb = len(new_base64_data) / 1024

                    logger.info(
                        f"[resize_image] {width}x{height} ({original_size_mb:.2f}MB) → "
                        f"{resized_width}x{resized_height} quality={quality} "
                        f"JPEG={jpeg_size_kb:.1f}KB base64={base64_size_kb:.1f}KB"
                    )

                    return f"data:image/jpeg;base64,{new_base64_data}"

                # Too large: lower quality and retry.
                quality -= 10
                attempts += 1

            # Still too large: last attempt with quality=20.
            buffer = BytesIO()
            resized_image.save(buffer, format="JPEG", quality=20)
            jpeg_bytes = buffer.getvalue()
            jpeg_size_kb = len(jpeg_bytes) / 1024
            new_base64_data = base64.b64encode(jpeg_bytes).decode()
            base64_size_kb = len(new_base64_data) / 1024

            logger.warning(
                f"[resize_image] Had to use low quality: {width}x{height} → "
                f"{resized_width}x{resized_height} quality=20 base64={base64_size_kb:.1f}KB"
            )

            return f"data:image/jpeg;base64,{new_base64_data}"

        except Exception as e:
            logger.warning(f"Failed to resize image: {e}, keeping original")
            return base64_str

    # Walk messages and process images.
    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if not content:
            continue

        # If content is a list (multimodal message).
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue

                # Check for image.
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                        if url.startswith("data:image") and "base64," in url:
                            # Process base64 image.
                            new_url = process_base64_image(url, max_pixels, patch_factor)
                            image_url["url"] = new_url

    return messages


def _restore_image_urls(messages: list, original_urls: list) -> list:
    """
    Restore base64 image URLs back to original relative paths in messages.

    Args:
        messages: message list containing base64 images
        original_urls: original relative paths

    Returns:
        messages with restored relative paths
    """
    if not original_urls:
        return messages

    # Copy messages to avoid mutating originals.
    restored_messages = []
    url_index = 0

    for msg in messages:
        restored_msg = {"role": msg["role"]}
        content = msg.get("content")

        # If user message and content is a list (may contain images).
        if msg.get("role") == "user" and isinstance(content, list):
            restored_content = []
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                    # Restore original relative path.
                    if url_index < len(original_urls):
                        restored_item = {
                            "type": "image_url",
                            "image_url": {"url": original_urls[url_index]},
                        }
                        # Keep image_wh if present in original data.
                        image_url_data = content_item.get("image_url", {})
                        if "image_wh" in image_url_data:
                            restored_item["image_url"]["image_wh"] = image_url_data["image_wh"]

                        restored_content.append(restored_item)
                        url_index += 1
                    else:
                        # If no matching original URL, keep as-is.
                        restored_content.append(content_item)
                else:
                    restored_content.append(content_item)
            restored_msg["content"] = restored_content
        else:
            # Non-user message or plain text: copy directly.
            restored_msg["content"] = content

        restored_messages.append(restored_msg)

    return restored_messages


def restore_original_images_in_place(item: dict) -> dict:
    """Restore base64 images back to original paths when metadata carries them."""
    original_urls = (item.get("metadata") or {}).get("original_image_urls")
    if original_urls:
        item["messages"] = _restore_image_urls(item.get("messages", []), original_urls)
    return item


def should_skip_for_final_output(item: dict) -> bool:
    """Skip rest of pipeline when first-round already produced valid output."""
    return (item.get("metadata") or {}).get("used_ground_truth") is False


def extract_question_text(messages: list[dict]) -> str:
    """Extract user text from messages (supports multimodal lists)."""
    for msg in reversed(messages or []):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    return content_item.get("text", "")
        else:
            return content
    return ""


class DataConverterStage(Stage):
    """Data conversion stage: convert raw data into SFT training format.

    Supported input formats:
    1. Multimodal data (with images)
    2. Text-only data (no images)

    Example input 1 - multimodal:
    {
        "id": -1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image.jpg", "image_wh": [1000,1000]}},
                    {"type": "text", "text": "<IMG_CONTEXT>\\nquestion text"}
                ]
            },
            {"role": "assistant", "content": "answer"}
        ],
        "doc_loc": "s3://.../P~xxx~1.0.0~0.0/jsonl/part-001.jsonl"
    }

    Example input 2 - text-only:
    {
        "id": 123,
        "messages": [
            {"role": "user", "content": "question text"},
            {"role": "assistant", "content": "answer"}
        ],
        "doc_loc": "s3://..."
    }

    Conversion rules:
    1. Multimodal: read local image files and encode to base64 (OpenAI format for vLLM).
    2. Text-only: convert string content to the standard content list format.
    3. Remove <IMG_CONTEXT> marker.
    4. Extract assistant answer into metadata.answer.
    5. Keep only user messages.
    6. Preserve original image_wh if present.

    Output format:
    {
        "id": "xxx",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "image_wh": [w, h]}},  # if image present
                    {"type": "text", "text": "question text"}
                ]
            }
        ],
        "metadata": {"answer": "answer"}
    }
    """

    def __init__(self, config: SFTConfig):
        """
        Initialize the data converter.

        Args:
            config: SFT config with image path settings

        Image path config priority:
            1. image_base_path: full path shared by all datasets
            2. image_base_dir: base dir to infer full paths per dataset
        """
        self.config = config
        self.abs_image_path_field = config.abs_image_path_field

    def _convert_image_url_to_base64(
        self, content_item: dict, image_base_path: str
    ) -> tuple[dict, str, str]:
        """
        Convert image_url content to base64 data URL format.

        Reads local image files and encodes them to base64 (OpenAI format for vLLM).
        Preserves original image_wh (width/height).

        Args:
            content_item: {"type": "image_url", "image_url": {"url": "...", "image_wh": [w, h]}}
            image_base_path: image base path (absolute)

        Returns:
            tuple: (converted_content_item, full_image_path, relative_path)
                - converted_content_item: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "image_wh": [w, h]}}
                - full_image_path: full image path (for debugging)
                - relative_path: original relative path (for output restore)

        Raises:
            ValueError: if image path is missing or image file is too large
            FileNotFoundError: if image file does not exist
        """
        import base64
        import mimetypes

        if not image_base_path:
            raise ValueError(
                "Image data found but no absolute image path provided.\n"
                "Run preprocess_images.py to add absolute image paths first."
            )

        image_url_data = content_item.get("image_url", {})
        relative_path = image_url_data.get("url", "")
        full_path = os.path.join(image_base_path, relative_path)

        # Check file exists.
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # Check file size (avoid overly large images).
        file_size = os.path.getsize(full_path)
        max_image_size = getattr(self.config, "max_image_size_mb", 10) * 1024 * 1024
        if file_size > max_image_size:
            size_mb = file_size / (1024 * 1024)
            max_size_mb = max_image_size / (1024 * 1024)
            raise ValueError(
                f"Image file too large: {size_mb:.2f}MB (max: {max_size_mb:.2f}MB)\n"
                f"File: {full_path}\n"
                f"Large images may cause output to be too long or exceed token limits."
            )

        # Read image file.
        with open(full_path, "rb") as f:
            image_data = f.read()

        # Encode to base64.
        base64_encoded = base64.b64encode(image_data).decode("utf-8")

        # Infer MIME type.
        mime_type, _ = mimetypes.guess_type(full_path)
        if mime_type is None:
            # Default to jpeg.
            mime_type = "image/jpeg"

        # Preserve original image_wh if present.
        image_wh = image_url_data.get("image_wh")

        # Return base64 data URL, full path, and relative path.
        converted_item = {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{base64_encoded}"},
        }

        # Keep size info if provided in the original data.
        if image_wh:
            converted_item["image_url"]["image_wh"] = image_wh

        return converted_item, full_path, relative_path

    def _normalize_text_content(self, content_item: dict) -> dict:
        """
        Normalize text content and remove special markers.

        Args:
            content_item: {"type": "text", "text": "..."}

        Returns:
            normalized text content
        """
        text = content_item.get("text", "")
        # Remove <IMG_CONTEXT> marker.
        text = text.replace("<IMG_CONTEXT>\n", "").strip()

        return {"type": "text", "text": text}

    def _process_user_content(
        self, content, image_base_path: str = None
    ) -> tuple[list, list, list]:
        """
        Process the content field of a user message.

        Args:
            content: may be a string or list
            image_base_path: image base path (if images are present)

        Returns:
            tuple: (content_list, image_paths, relative_paths)
                - content_list: normalized content list
                - image_paths: full image paths (for debugging)
                - relative_paths: original relative paths (for output restore)
        """
        # Case 1: plain text string.
        if isinstance(content, str):
            return [{"type": "text", "text": content}], [], []

        # Case 2: structured list.
        new_content = []
        image_paths = []
        relative_paths = []
        for content_item in content:
            if not isinstance(content_item, dict):
                continue  # Skip non-dict items.

            content_type = content_item.get("type")

            if content_type == "image_url":
                converted_item, full_path, relative_path = self._convert_image_url_to_base64(
                    content_item, image_base_path
                )
                new_content.append(converted_item)
                image_paths.append(full_path)
                relative_paths.append(relative_path)
            elif content_type == "text":
                new_content.append(self._normalize_text_content(content_item))
            else:
                # Keep other types.
                new_content.append(content_item)

        return new_content, image_paths, relative_paths

    def process(self, batch: list[dict]) -> list[dict]:
        """Restore image URLs first, then use base class error handling."""
        restored_batch = [restore_original_images_in_place(item) for item in batch]
        return self._default_sync_process(restored_batch)

    def process_item(self, item: dict) -> dict:
        """
        Process one item and convert it to SFT training format.

        Args:
            item: raw item (should include absolute image path after preprocessing)

        Returns:
            converted item

        Raises:
            ValueError: if images exist but absolute path field is missing
        """
        if should_skip_for_final_output(item):
            return item
        # 1. Get absolute image base path (from configured field).
        image_base_path = get_nested_value(item, self.abs_image_path_field)

        result = {
            "id": item.get("id", "unknown"),
            "messages": [],
            "metadata": {},
        }
        if item.get("abs_path") is not None:
            result["abs_path"] = item.get("abs_path")

        # 3. Extract assistant answer into metadata.
        # If metadata already has answer, prefer it.
        existing_answer = (item.get("metadata") or {}).get("answer")
        if existing_answer:
            result["metadata"]["answer"] = existing_answer
        else:
            # Otherwise extract from messages.
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    answer = msg.get("content", "")
                    if answer:
                        result["metadata"]["answer"] = answer
                    break

        # 4. Process user messages, collecting image paths and relative paths.
        all_image_paths = []
        all_relative_paths = []
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                new_content, image_paths, relative_paths = self._process_user_content(
                    content, image_base_path
                )
                all_image_paths.extend(image_paths)
                all_relative_paths.extend(relative_paths)

                result["messages"].append({"role": "user", "content": new_content})

        # 5. Store image paths and relative paths in metadata.
        if all_image_paths:
            result["metadata"]["image_paths"] = all_image_paths  # For debugging.
        if all_relative_paths:
            result["metadata"]["original_image_urls"] = all_relative_paths  # For output restore.

        return result


# ============================================================
# Example 2: threaded mode via decorator
# ============================================================


@Stage.threaded_mode
class ParseStage(Stage):
    """
    Parsing stage: parse the answer.

    Skip condition: if metadata.short_answer exists, skip.

    Output:
        Parse answer to short_answer. This only affects verification behavior
        and does not change failure handling (keeps the original, longer answer).

    Downstream: verify uses metadata.short_answer, not metadata.answer.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: SyncOpenAIClient = None

    def initialize(self):
        """
        Called once when the actor is created; set thread pool size and init client.

        Set thread pool size via self._thread_pool_size.
        """
        self._thread_pool_size = self.config.verifier_max_workers  # Reuse verifier model.

        # Initialize sync client.
        api_key = self.config.judge_api_key or self.config.api_key
        base_url = self.config.judge_base_url or self.config.base_url

        self.client = SyncOpenAIClient(api_key=api_key, base_url=base_url)
        self.client.initialize()

    def process_item(self, item: dict) -> dict:
        """
        Process a single item and parse short_answer.

        Framework guarantees:
        - thread pool concurrency within batch (verifier_max_workers)
        - automatic exception capture; failed items marked _failed=True
        - failed items are skipped before process_item
        - no manual try-catch needed
        """
        if should_skip_for_final_output(item):
            return item

        # Skip if short_answer already exists.
        if (item.get("metadata") or {}).get("short_answer") is not None:
            return item

        question = extract_question_text(item.get("messages", []))

        answer = item.get("metadata", {}).get("answer")

        prompt = EXTRACT_ANSWER_TEMPLATE.format(question_text=question, answer_text=answer)
        short_answer = self._call_parse(prompt)
        item["metadata"]["short_answer"] = short_answer

        logger.info(f"[ParseStage] Item {item.get('id', 'unknown')}: Short answer: {short_answer}")

        return item

    def _call_parse(self, prompt: str) -> str:
        """Call parse model."""
        return self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.config.parse_model,
            max_tokens=self.config.parse_max_tokens,
            temperature=self.config.parse_temperature,
        )


# Async sampling
class SamplerStage(Stage):
    """
    Sampling stage: generate n responses per item.
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: AsyncOpenAIClient = None

    def initialize(self):
        """Called once when actor is created; build shared client config."""
        self.client = AsyncOpenAIClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
            semaphore_size=self.config.semaphore_per_sampler,
        )

    async def process(self, batch: list[dict]) -> list[dict]:
        """
        Process a batch; create and manage the session at batch scope.
        """
        # Create batch-level session and semaphore.
        timeout = aiohttp.ClientTimeout(total=1800)  # 30-minute total timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            semaphore = asyncio.Semaphore(self.client.semaphore_size)

            # Process all items concurrently within the batch.
            async def process_one(item: dict) -> dict:
                # If already failed, return as-is.
                if item.get("_failed") is True:
                    return item

                if should_skip_for_final_output(item):
                    return item

                messages = item.get("messages", [])
                item_id = item.get("id", "unknown")

                try:
                    # First attempt: normal API call.
                    # tools.py retries all errors except 413 automatically.
                    responses = await self.client.chat_completion(
                        session=session,
                        semaphore=semaphore,
                        messages=messages,
                        model=self.config.model,
                        n=self.config.n_samples,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    logger.info(
                        f"[SamplerStage] Item {item_id}: Generated {len(responses)} responses"
                    )
                    return {**item, "responses": responses}

                except Exception as e:
                    import traceback

                    error_str = str(e)

                    # Only handle 413 (request too large): resize images and retry.
                    if "413" in error_str:
                        logger.warning(
                            f"[SamplerStage] Item {item_id}: 413, resizing images and retrying..."
                        )
                        try:
                            # Compute more aggressive resize parameters.
                            patch_factor = int(14 * SPATIAL_MERGE_SIZE)
                            aggressive_max_pixels = (IMAGE_MAX_TOKEN_NUM // 2) * patch_factor**2
                            logger.info(
                                f"[SamplerStage] Item {item_id}: Using aggressive resize with max_pixels={aggressive_max_pixels}"
                            )

                            # Resize images.
                            resized_messages = resize_messages_images(
                                messages, max_pixels=aggressive_max_pixels
                            )

                            # Retry once.
                            responses = await self.client.chat_completion(
                                session=session,
                                semaphore=semaphore,
                                messages=resized_messages,
                                model=self.config.model,
                                n=self.config.n_samples,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                            )
                            logger.info(
                                f"[SamplerStage] Item {item_id}: Retry successful after resizing, generated {len(responses)} responses"
                            )

                            # Update item messages with resized version.
                            return {**item, "messages": resized_messages, "responses": responses}

                        except Exception as retry_e:
                            # Still failed after resize.
                            error_trace = traceback.format_exc()
                            logger.error(
                                f"[SamplerStage] ❌ Item {item_id} failed after resizing: {retry_e}\n{error_trace}"
                            )
                            return {
                                **item,
                                "_failed": True,
                                "_error": f"SamplerStage (after resize): {retry_e}",
                                "_traceback": error_trace,
                            }

                    else:
                        # Other errors (500, 104, etc.) already retried in tools.py.
                        error_trace = traceback.format_exc()
                        logger.error(
                            f"[SamplerStage] ❌ Item {item_id} failed (already retried in client): {e}\n{error_trace}"
                        )
                        return {
                            **item,
                            "_failed": True,
                            "_error": f"SamplerStage: {e}",
                            "_traceback": error_trace,
                        }

            return await asyncio.gather(*[process_one(item) for item in batch])


# ============================================================
# Example 2: threaded mode via decorator
# ============================================================


@Stage.threaded_mode
class VerifierStage(Stage):
    """
    Verification stage: use an LLM-as-Judge to score each response.

    Mode: threaded
    - use @Stage.threaded_mode
    - set self._thread_pool_size in initialize()
    - implement process_item(); framework handles per-item concurrency
    """

    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: SyncOpenAIClient = None

    def initialize(self):
        """
        Called once when actor is created; set thread pool size and init client.

        Set thread pool size via self._thread_pool_size.
        """
        self._thread_pool_size = self.config.verifier_max_workers

        # Initialize sync client.
        api_key = self.config.judge_api_key or self.config.api_key
        base_url = self.config.judge_base_url or self.config.base_url

        self.client = SyncOpenAIClient(api_key=api_key, base_url=base_url)
        self.client.initialize()

    def process_item(self, item: dict) -> dict:
        """
        Process a single item and score all responses.

        Framework guarantees:
        - thread pool concurrency within batch (verifier_max_workers)
        - automatic exception capture; failed items marked _failed=True
        - failed items are skipped before process_item
        - no manual try-catch needed
        """
        if should_skip_for_final_output(item):
            return item

        responses = item.get("responses", [])
        metadata = item.get("metadata", {})
        messages = item.get("messages", [])
        item_id = item.get("id", "unknown")

        logger.info(f"[VerifierStage] Item {item_id}: Verifying {len(responses)} responses")

        question = extract_question_text(messages)

        # Verify all responses and return the first judge prompt/output for debugging.
        rollouts, first_judge_prompt, first_judge_output = self._verify_llm_judge(
            responses, metadata, question
        )

        # Build result (exclude responses field).
        result = {k: v for k, v in item.items() if k != "responses"}
        result["rollouts"] = rollouts

        # Ensure metadata is a dict to avoid downstream errors.
        if result.get("metadata") is None:
            logger.warning(
                f"[VerifierStage][BUG] Item {item_id}: metadata is None, setting to empty dictionary"
            )
            logger.warning(f"item: {item}")
            result["metadata"] = {}

        # Save first judge prompt/output to metadata (debugging).
        if first_judge_prompt is not None:
            result["metadata"]["judge_prompt_sample"] = first_judge_prompt
            result["metadata"]["judge_output_sample"] = first_judge_output

        return result

    def _verify_llm_judge(
        self, responses: list[str], metadata: dict, question: str
    ) -> tuple[list[dict], str, str]:
        """
        Use LLM judge to verify multiple responses.

        For one item, call the judge sequentially for N responses.

        Returns:
            tuple: (rollouts, first_judge_prompt, first_judge_output)
                - rollouts: verification results
                - first_judge_prompt: first judge prompt (debugging)
                - first_judge_output: first judge output (debugging)
        """
        if not responses:
            return [], None, None

        if "short_answer" not in metadata:
            raise ValueError(
                f"[VerifierStage] ⚠️  Question: {question}, short_answer not in metadata"
            )
        else:
            gold_target = metadata["short_answer"]
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

            judge_output = None
            try:
                judge_output = self._call_judge(prompt)
                is_correct = self._parse_judge_output(judge_output)
                score = 1.0 if is_correct else 0.0
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                logger.warning(
                    f"[VerifierStage] ⚠️  Judge error on response {idx}: {e}\n{error_trace}"
                )
                score = 0.0
                judge_output = f"ERROR: {e}\n\nTraceback:\n{error_trace}"
            finally:
                # Save the first prompt/output for debugging.
                if idx == 0:
                    first_judge_prompt = prompt
                    first_judge_output = judge_output

            rollouts.append({"response": response, "score": score})

        return rollouts, first_judge_prompt, first_judge_output

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model."""
        model = self.config.judge_model or self.config.model
        return self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=self.config.judge_max_tokens,
            temperature=self.config.judge_temperature,
        )

    def _parse_judge_output(self, output: str) -> bool:
        """Parse judge output (A=correct, B=incorrect)."""
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
# Example 3: sync mode - only implement process_item (recommended)
# ============================================================


class FormatterStage(Stage):
    """
    Formatting stage: select verified responses and emit SFT format.

    Mode: sync
    - implement process_item() only
    - framework handles exceptions and runs sequentially
    - suitable for light compute with no extra concurrency
    """

    def __init__(self, config: SFTConfig):
        self.config = config

    def process(self, batch: list[dict]) -> list[dict]:
        """Restore image URLs then use base class error handling."""
        restored_batch = [restore_original_images_in_place(item) for item in batch]
        return self._default_sync_process(restored_batch)

    def process_item(self, item: dict) -> dict:
        """
        Process a single item and format it for SFT training.
        """
        if should_skip_for_final_output(item):
            logger.info(
                f"[FormatterStage] Item {item.get('id', 'unknown')}: Already has valid output, skipping"
            )
            item["metadata"]["skipped"] = True
            return item

        messages = item.get("messages", [])
        rollouts = item.get("rollouts") or []
        metadata = item.get("metadata", {})
        item_id = item.get("id", "unknown")

        # Select responses that passed verification.
        passed = [r for r in rollouts if r.get("score", 0) >= self.config.pass_threshold]
        logger.info(f"[FormatterStage] Item {item_id}: {len(passed)}/{len(rollouts)} passed")

        # Choose the best response.
        if passed:
            best_response = passed[0]["response"]
            used_gt = False
        else:
            # Fallback to ground truth.
            gt = metadata.get("answer") or metadata.get("gold_target", "")
            if gt:
                best_response = gt
                used_gt = True
                logger.info(f"[FormatterStage] Item {item_id}: Using ground truth")
            else:
                # Mark as failed.
                logger.warning(f"[FormatterStage] Item {item_id}: No valid response")
                return {
                    **item,
                    "_failed": True,
                    "_error": "No response passed and no ground truth",
                }

        # Build SFT format.
        sft_messages = messages + [{"role": "assistant", "content": best_response}]

        # Clean metadata (remove internal debug fields).
        clean_metadata = {
            k: v
            for k, v in metadata.items()
            if k
            not in [
                "image_paths",
                "original_image_urls",
                "judge_prompt_sample",
                "judge_output_sample",
            ]
        }
        clean_metadata.update(
            {
                "n_passed": len(passed),
                "n_total": len(rollouts),
                "used_ground_truth": used_gt,
            }
        )

        result = {}
        result = {
            **item,
            "messages": sft_messages,
            "metadata": clean_metadata,
        }

        # Ensure id/messages/metadata are ordered before output.
        ordered_result = {
            "id": result["id"],
            "messages": result["messages"],
            "metadata": result["metadata"],
        }
        # Add other fields (besides id, messages, metadata).
        for key in result:
            if key not in ["id", "messages", "metadata"]:
                ordered_result[key] = result[key]
        result = ordered_result
        return result


# ============================================================
# Recipe definition
# ============================================================


class SFTRecipe(BaseRecipe):
    """SFT recipe."""

    config_class = SFTConfig

    def __init__(self, config: SFTConfig):
        super().__init__(config)

    def stages(self) -> list[Stage]:
        """Return stage list (execution order)."""
        return [
            DataConverterStage(self.config),
            ParseStage(self.config),
            SamplerStage(self.config),
            VerifierStage(self.config),
            FormatterStage(self.config),
        ]


if __name__ == "__main__":
    config = SFTConfig()

    data_converter = DataConverterStage(config)

# do quick unit test here
