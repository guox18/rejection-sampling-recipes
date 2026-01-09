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
import base64
from io import BytesIO
import logging
import os

import aiohttp
from PIL import Image

from src.base import BaseRecipe, Stage
from src.utils import get_nested_value
from src.utils.qwen3_vl_util import smart_resize, SPATIAL_MERGE_SIZE, IMAGE_MAX_TOKEN_NUM

from .config import SFTConfig
from .tools import AsyncOpenAIClient, SyncOpenAIClient, DEFAULT_JUDGE_TEMPLATE, clip_thinking, EXTRACT_ANSWER_TEMPLATE

# 配置专门的 Recipe 日志 - 简洁格式，只输出到文件
def _setup_recipe_logger():
    """配置简洁的 Recipe 日志"""
    logger = logging.getLogger("recipe")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 不传播到 root logger
    
    # 如果已经有 handler，不重复添加
    if logger.handlers:
        return logger
    
    # 从环境变量读取日志文件路径
    log_file = os.environ.get("RECIPE_LOG_FILE", "recipe_run.log")
    
    # 创建文件 handler
    handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    handler.setLevel(logging.INFO)
    
    # 简洁格式：只有时间和消息
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

logger = _setup_recipe_logger()


def resize_messages_images(messages: list[dict], max_pixels: int = None) -> list[dict]:
    """
    对 messages 中的所有图片进行 resize 操作.
    
    Args:
        messages: 对话消息列表
        max_pixels: 最大像素数（默认使用 IMAGE_MAX_TOKEN_NUM * factor^2）
        
    Returns:
        处理后的 messages（原地修改并返回）
    """
    # 计算默认 max_pixels
    patch_factor = int(14 * SPATIAL_MERGE_SIZE)
    if max_pixels is None:
        max_pixels = IMAGE_MAX_TOKEN_NUM * patch_factor ** 2
    import copy
    
    def to_rgb(image: Image.Image) -> Image.Image:
        """转换图片为 RGB 格式"""
        if image.mode == "RGB":
            return image
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background.convert("RGB")
    
    def process_base64_image(base64_str: str, max_pixels: int, patch_factor: int, max_base64_kb: int = 800) -> str:
        """
        处理 base64 编码的图片，保证最终 base64 大小不超过指定值.
        
        Args:
            base64_str: 原始 base64 字符串
            max_pixels: 最大像素数（用于初始 resize）
            patch_factor: patch 因子（必须是倍数）
            max_base64_kb: 最大 base64 大小（KB），默认 800KB
        """
        try:
            # 解析 base64 字符串
            if "base64," in base64_str:
                header, base64_data = base64_str.split("base64,", 1)
            else:
                return base64_str  # 不是 base64 格式，跳过
            
            # 解码图片
            data = base64.b64decode(base64_data)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
            
            # 转换为 RGB
            image = to_rgb(image_obj)
            width, height = image.size
            original_size_mb = len(data) / (1024 * 1024)
            
            # 计算新的尺寸
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=patch_factor,
                min_pixels=4 * patch_factor ** 2,
                max_pixels=max_pixels,
            )
            
            # Resize 图片
            resized_image = image.resize((resized_width, resized_height))
            
            # 动态调整 JPEG 质量，确保 base64 大小不超限
            max_base64_bytes = max_base64_kb * 1024
            quality = 85
            attempts = 0
            
            while quality >= 20 and attempts < 10:
                buffer = BytesIO()
                resized_image.save(buffer, format="JPEG", quality=quality)
                jpeg_bytes = buffer.getvalue()
                
                # Base64 编码后大小约为原始大小的 4/3
                estimated_base64_size = len(jpeg_bytes) * 4 / 3
                
                if estimated_base64_size <= max_base64_bytes:
                    # 满足大小要求
                    jpeg_size_kb = len(jpeg_bytes) / 1024
                    new_base64_data = base64.b64encode(jpeg_bytes).decode()
                    base64_size_kb = len(new_base64_data) / 1024
                    
                    logger.info(
                        f"[resize_image] {width}x{height} ({original_size_mb:.2f}MB) → "
                        f"{resized_width}x{resized_height} quality={quality} "
                        f"JPEG={jpeg_size_kb:.1f}KB base64={base64_size_kb:.1f}KB"
                    )
                    
                    return f"data:image/jpeg;base64,{new_base64_data}"
                
                # 如果太大，降低质量重试
                quality -= 10
                attempts += 1
            
            # 如果质量降到很低仍然太大，最后一次尝试用 quality=20
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
    
    # 遍历 messages 并处理图片
    for message in messages:
        if not isinstance(message, dict):
            continue
            
        content = message.get("content")
        if not content:
            continue
        
        # 如果 content 是列表（多模态消息）
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                
                # 检查是否是图片
                if item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                        if url.startswith("data:image") and "base64," in url:
                            # 处理 base64 图片
                            new_url = process_base64_image(url, max_pixels, patch_factor)
                            image_url["url"] = new_url
    
    return messages


def _restore_image_urls(messages: list, original_urls: list) -> list:
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
                            "image_url": {
                                "url": original_urls[url_index]
                            }
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

class DataConverterStage(Stage):
    """
    数据格式转换阶段：将原始数据转换为 SFT 训练格式.
    
    支持的输入格式：
    1. 多模态数据（带图像）
    2. 纯文本数据（不带图像）
    
    输入格式示例 1 - 多模态数据:
    {
        "id": -1,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "image.jpg", "image_wh": [1000,1000]}},
                    {"type": "text", "text": "<IMG_CONTEXT>\n问题文本"}
                ]
            },
            {"role": "assistant", "content": "答案"}
        ],
        "doc_loc": "s3://.../P~xxx~1.0.0~0.0/jsonl/part-001.jsonl"
    }
    
    输入格式示例 2 - 纯文本数据:
    {
        "id": 123,
        "messages": [
            {"role": "user", "content": "问题文本"},
            {"role": "assistant", "content": "答案"}
        ],
        "doc_loc": "s3://..."
    }
    
    转换规则:
    1. 对于多模态数据：读取本地图片文件并编码为 base64（保持 OpenAI 格式以兼容 vLLM API）
    2. 对于纯文本数据：将字符串格式转换为标准的 content 列表格式
    3. 移除 <IMG_CONTEXT> 标记
    4. 提取 assistant 的回答到 metadata.answer
    5. 只保留 user 消息
    6. 保留原始的 image_wh 信息（如果存在）
    
    输出格式:
    {
        "id": "xxx",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "image_wh": [w, h]}},  # 如果有图像
                    {"type": "text", "text": "问题文本"}
                ]
            }
        ],
        "metadata": {"answer": "答案"}
    }
    """
    
    def __init__(self, config: SFTConfig):
        """
        初始化数据转换器.
        
        Args:
            config: SFT 配置对象，包含图片路径等配置
        
        图片路径配置优先级:
            1. image_base_path: 完整路径，所有数据集共用
            2. image_base_dir: 基础目录，为每个数据集动态推断完整路径
        """
        self.config = config
        self.abs_image_path_field = config.abs_image_path_field

    def _convert_image_url_to_base64(self, content_item: dict, image_base_path: str) -> tuple[dict, str, str]:
        """
        将 image_url 格式转换为 base64 编码格式.
        
        读取本地图片文件并编码为 base64，兼容 vLLM API endpoint.
        同时保留原始的 image_wh 信息（图片宽高）。
        
        Args:
            content_item: {"type": "image_url", "image_url": {"url": "...", "image_wh": [w, h]}}
            image_base_path: 图片基础路径（绝对路径）
        
        Returns:
            tuple: (converted_content_item, full_image_path, relative_path)
                - converted_content_item: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...", "image_wh": [w, h]}}
                - full_image_path: 图像文件的完整路径（用于调试）
                - relative_path: 原始的相对路径（用于最终输出恢复）
        
        Raises:
            ValueError: 如果图片路径未提供或图像文件过大
            FileNotFoundError: 如果图片文件不存在
        """
        import base64
        import mimetypes
        
        if not image_base_path:
            raise ValueError(
                "遇到图像数据但未提供图片绝对路径。\n"
                "请先运行 preprocess_images.py 脚本为数据添加绝对路径信息。"
            )
        
        image_url_data = content_item.get("image_url", {})
        relative_path = image_url_data.get("url", "")
        full_path = os.path.join(image_base_path, relative_path)
        
        # 检查文件是否存在
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Image file not found: {full_path}")

        # 检查图像文件大小（避免过大的图像导致输出过长）
        file_size = os.path.getsize(full_path)
        max_image_size = getattr(self.config, 'max_image_size_mb', 10) * 1024 * 1024  
        if file_size > max_image_size:
            size_mb = file_size / (1024 * 1024)
            max_size_mb = max_image_size / (1024 * 1024)
            raise ValueError(
                f"Image file too large: {size_mb:.2f}MB (max: {max_size_mb:.2f}MB)\n"
                f"File: {full_path}\n"
                f"Large images may cause output to be too long or exceed token limits."
            )
        
        # 读取图像文件
        with open(full_path, 'rb') as f:
            image_data = f.read()
        
        # 编码为 base64
        base64_encoded = base64.b64encode(image_data).decode('utf-8')
        
        # 推断 MIME 类型
        mime_type, _ = mimetypes.guess_type(full_path)
        if mime_type is None:
            # 默认使用 jpeg
            mime_type = "image/jpeg"
        
        # 保留原始的 image_wh 信息（如果有）
        image_wh = image_url_data.get("image_wh")
        
        # 返回 base64 编码的 data URL、完整路径和相对路径
        converted_item = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_encoded}"
            }
        }
        
        # 如果原始数据包含尺寸信息，保留它
        if image_wh:
            converted_item["image_url"]["image_wh"] = image_wh
        
        return converted_item, full_path, relative_path
    
    def _normalize_text_content(self, content_item: dict) -> dict:
        """
        标准化文本内容，移除特殊标记.
        
        Args:
            content_item: {"type": "text", "text": "..."}
        
        Returns:
            处理后的文本 content
        """
        text = content_item.get("text", "")
        # 移除 <IMG_CONTEXT> 标记
        text = text.replace("<IMG_CONTEXT>\n", "").strip()
        
        return {
            "type": "text",
            "text": text
        }
    
    def _process_user_content(self, content, image_base_path: str = None) -> tuple[list, list, list]:
        """
        处理 user 消息的 content 字段.
        
        Args:
            content: 可能是字符串或列表
            image_base_path: 图片基础路径（如果包含图像）
        
        Returns:
            tuple: (content_list, image_paths, relative_paths)
                - content_list: 标准化的 content 列表
                - image_paths: 图像文件完整路径列表（用于调试）
                - relative_paths: 原始的相对路径列表（用于最终输出恢复）
        """
        # 情况1：纯文本格式（字符串）
        if isinstance(content, str):
            return [{
                "type": "text",
                "text": content
            }], [], []
        
        # 情况2：结构化格式（列表）
        new_content = []
        image_paths = []
        relative_paths = []
        for content_item in content:
            if not isinstance(content_item, dict):
                continue  # 跳过非字典元素
            
            content_type = content_item.get("type")
            
            if content_type == "image_url":
                converted_item, full_path, relative_path = self._convert_image_url_to_base64(content_item, image_base_path)
                new_content.append(converted_item)
                image_paths.append(full_path)
                relative_paths.append(relative_path)
            elif content_type == "text":
                new_content.append(self._normalize_text_content(content_item))
            else:
                # 保留其他类型
                new_content.append(content_item)
        
        return new_content, image_paths, relative_paths
    
    def process(self, batch: list[dict]) -> list[dict]:
        """覆盖标准的 process 方法, 因为最后一个阶段需要进行失败处理"""
        results = []
        for item in batch:
            # (1) bugfix: 之前的 formatter 忘记恢复 img_url
            # 导致, 第二轮 roll 数据时, 无法找到第一轮出错项的 img_url
            original_urls = (item.get("metadata") or {}).get("original_image_urls")
            if original_urls:
                item["messages"] = _restore_image_urls(item["messages"], original_urls)
            
            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                logger.error(
                    f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed: {e}\n{error_trace}"
                )
                results.append(
                    {
                        **item,
                        "_failed": True,
                        "_error": f"{type(self).__name__}: {e}",
                        "_traceback": error_trace,
                    }
                )
        return results
    
    def process_item(self, item: dict) -> dict:
        """
        处理单个数据项，转换为 SFT 训练格式.
        
        Args:
            item: 原始数据项（预处理后应包含绝对路径信息）
        
        Returns:
            转换后的数据项
        
        Raises:
            ValueError: 如果包含图像但绝对路径字段不存在
        """
        # 如果未使用 ground_truth, 则说明已经产生有效的输出.
        if item.get("metadata") is not None and item.get("metadata", {}).get("used_ground_truth") is False:
            return item
        # 1. 获取图片绝对路径（从配置的字段读取）
        image_base_path = get_nested_value(item, self.abs_image_path_field)

        result = {
            "id": item.get("id", "unknown"),
            "messages": [],
            "metadata": {},
        }
        if item.get("abs_path") is not None:
            result["abs_path"] = item.get("abs_path")
        
        # 3. 提取 assistant 的回答到 metadata
        # 如果 metadata 中已经有 answer，优先使用已有的
        existing_answer = (item.get("metadata") or {}).get("answer")
        if existing_answer:
            result["metadata"]["answer"] = existing_answer
        else:
            # 否则从 messages 中提取
            for msg in item.get("messages", []):
                if msg.get("role") == "assistant":
                    answer = msg.get("content", "")
                    if answer:
                        result["metadata"]["answer"] = answer
                    break
        
        # 4. 处理 user 消息，收集图像路径和相对路径
        all_image_paths = []
        all_relative_paths = []
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", [])
                new_content, image_paths, relative_paths = self._process_user_content(content, image_base_path)
                all_image_paths.extend(image_paths)
                all_relative_paths.extend(relative_paths)
                
                result["messages"].append({
                    "role": "user",
                    "content": new_content
                })
        
        # 5. 将图像路径和相对路径保存到 metadata
        if all_image_paths:
            result["metadata"]["image_paths"] = all_image_paths  # 用于调试
        if all_relative_paths:
            result["metadata"]["original_image_urls"] = all_relative_paths  # 用于最终输出恢复
        
        return result

# ============================================================
# 示例 2: 多线程模式 - 使用装饰器 
# ============================================================

@Stage.threaded_mode
class ParseStage(Stage):
    """
    解析阶段: 解析答案. 
    
    通过条件: 只要 metadata.short_answer 字段存在, 就跳过
    
    输出: 
        对 answer 进行解析, 得到 short_answer. 仅改变 verify 的行为, 不影响失败处理 (保留原始的 answer, 相对长一些)
    
    与后续管线: verify 使用 metadata.short_answer 来进行验证, 而非 metadata.answer
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.client: SyncOpenAIClient = None
    
    def initialize(self):
        """
        Actor 创建时调用一次, 设置线程池大小并初始化客户端.
        
        通过设置 self._thread_pool_size 指定线程池大小.
        """
        self._thread_pool_size = self.config.verifier_max_workers # 直接和 verifier 共用一组模型.
        
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
        - 失败的 item 会被框架自动跳过, 不会进入 process_item 方法
        - 无需手动 try-catch
        """
        # 如果未使用 ground_truth, 则说明已经产生有效的输出.
        if (item.get("metadata") or {}).get("used_ground_truth") is False:
            return item
        
        # 如果已经有 short answer, 跳过
        if (item.get("metadata") or {}).get("short_answer") is not None:
            return item
        
        messages = item.get("messages", [])

        # 提取问题 (支持多模态数据)
        question = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    # 如果 content 是 list (多模态), 提取 text 部分
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                question = content_item.get("text", "")
                                break
                    else:
                        # 如果 content 是 string (纯文本)
                        question = content
                    break
        
        answer = item.get("metadata", {}).get("answer")
        
        prompt = EXTRACT_ANSWER_TEMPLATE.format(
            question_text=question,
            answer_text=answer
        )
        short_answer = self._call_parse(prompt)
        item["metadata"]["short_answer"] = short_answer

        # logger.info(f"[ParseStage] Item {item.get('id', 'unknown')}: Short answer: {short_answer}")
        
        return item

    def _call_parse(self, prompt: str) -> str:
        """调用 parse model."""
        return self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.config.parse_model,
            max_tokens=self.config.parse_max_tokens,
            temperature=self.config.parse_temperature,
        )


# 自行实现
class SamplerStage(Stage):
    """
    采样阶段: 为每个 item 生成 n 个 responses.
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
        timeout = aiohttp.ClientTimeout(total=1800)  # 30分钟总超时
        async with aiohttp.ClientSession(timeout=timeout) as session:
            semaphore = asyncio.Semaphore(self.client.semaphore_size)
            
            # 并发处理 batch 内的所有 items
            async def process_one(item: dict) -> dict:
                # 如果已经失败，直接返回
                if item.get("_failed") is True:
                    return item
                
                # 如果已经产生有效输出 (ground truth 为 False)，直接返回
                if item.get("metadata", {}).get("used_ground_truth") is False:
                    return item
                
                messages = item.get("messages", [])
                item_id = item.get("id", "unknown")
                
                try:
                    # 第一次尝试：正常调用 API
                    # 注意：tools.py 会自动处理除 413 外的所有错误重试
                    responses = await self.client.chat_completion(
                        session=session,
                        semaphore=semaphore,
                        messages=messages,
                        model=self.config.model,
                        n=self.config.n_samples,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    logger.info(f"[SamplerStage] Item {item_id}: Generated {len(responses)} responses")
                    return {**item, "responses": responses}
                
                except Exception as e:
                    import traceback
                    error_str = str(e)
                    
                    # 只处理 413 错误（请求太长）：resize 图片后重试
                    if "413" in error_str:
                        logger.warning(f"[SamplerStage] Item {item_id}: 413, resizing images and retrying...")
                        try:
                            # 计算更激进的 resize 参数
                            patch_factor = int(14 * SPATIAL_MERGE_SIZE)
                            aggressive_max_pixels = (IMAGE_MAX_TOKEN_NUM // 2) * patch_factor ** 2
                            logger.info(f"[SamplerStage] Item {item_id}: Using aggressive resize with max_pixels={aggressive_max_pixels}")
                            
                            # Resize 图片
                            resized_messages = resize_messages_images(messages, max_pixels=aggressive_max_pixels)
                            
                            # 重试一次（tools.py 仍会处理其他错误的重试）
                            responses = await self.client.chat_completion(
                                session=session,
                                semaphore=semaphore,
                                messages=resized_messages,
                                model=self.config.model,
                                n=self.config.n_samples,
                                temperature=self.config.temperature,
                                max_tokens=self.config.max_tokens,
                            )
                            logger.info(f"[SamplerStage] Item {item_id}: Retry successful after resizing, generated {len(responses)} responses")
                            
                            # 更新 item 中的 messages 为 resized 版本
                            return {**item, "messages": resized_messages, "responses": responses}
                        
                        except Exception as retry_e:
                            # Resize 后仍然失败
                            error_trace = traceback.format_exc()
                            logger.error(f"[SamplerStage] ❌ Item {item_id} failed after resizing: {retry_e}\n{error_trace}")
                            return {**item, "_failed": True, "_error": f"SamplerStage (after resize): {retry_e}", "_traceback": error_trace}
                    
                    else:
                        # 其他错误（500, 104 等）已经在 tools.py 中重试过了，直接标记失败
                        error_trace = traceback.format_exc()
                        logger.error(f"[SamplerStage] ❌ Item {item_id} failed (already retried in client): {e}\n{error_trace}")
                        return {**item, "_failed": True, "_error": f"SamplerStage: {e}", "_traceback": error_trace}
            
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
        - 失败的 item 会被框架自动跳过, 不会进入 process_item 方法
        - 无需手动 try-catch
        """
        # 如果未使用 ground_truth, 则说明已经产生有效的输出.
        if item.get("metadata", {}).get("used_ground_truth") is False:
            return item
        
        responses = item.get("responses", [])
        metadata = item.get("metadata", {})
        messages = item.get("messages", [])
        item_id = item.get("id", "unknown")
        
        logger.info(f"[VerifierStage] Item {item_id}: Verifying {len(responses)} responses")
        
        # 提取问题 (支持多模态数据)
        question = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    # 如果 content 是 list (多模态), 提取 text 部分
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and content_item.get("type") == "text":
                                question = content_item.get("text", "")
                                break
                    else:
                        # 如果 content 是 string (纯文本)
                        question = content
                    break
        
        # 验证所有 responses, 并返回第一条 judge 提示词和输出用于调试
        rollouts, first_judge_prompt, first_judge_output = self._verify_llm_judge(responses, metadata, question)
        
        # 返回结果（排除 responses 字段）
        result = {k: v for k, v in item.items() if k != "responses"}
        result["rollouts"] = rollouts
       
        # 确保 metadata 是字典而不是 None（防止下游 Stage 出错）
        if result.get("metadata") is None:
            logger.warning(f"[VerifierStage][BUG] Item {item_id}: metadata is None, setting to empty dictionary")
            logger.warning(f"item: {item}")
            result["metadata"] = {}
        
        # 保存第一条 judge 提示词和输出到 metadata (用于调试)
        if first_judge_prompt is not None:
            result["metadata"]["judge_prompt_sample"] = first_judge_prompt
            result["metadata"]["judge_output_sample"] = first_judge_output
        
        return result
    
    def _verify_llm_judge(self, responses: list[str], metadata: dict, question: str) -> tuple[list[dict], str, str]:
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
        
        if "short_answer" not in metadata:
            # logger.warning(f"[VerifierStage] ⚠️  Question: {question}, short_answer not in metadata, using answer as gold_target")
            # gold_target = metadata.get("answer", "")
            raise ValueError(f"[VerifierStage] ⚠️  Question: {question}, short_answer not in metadata")
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
            
            try:
                judge_output = self._call_judge(prompt)
                is_correct = self._parse_judge_output(judge_output)
                score = 1.0 if is_correct else 0.0
                
                # 保存第一条提示词和输出用于调试
                if idx == 0:
                    first_judge_prompt = prompt
                    first_judge_output = judge_output
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                logger.warning(f"[VerifierStage] ⚠️  Judge error on response {idx}: {e}\n{error_trace}")
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
    
    模式: 同步
    - 只实现 process_item()
    - 框架自动处理异常, 顺序执行
    - 适合: 轻量计算, 无需并发
    """
    
    def __init__(self, config: SFTConfig):
        self.config = config

    def process(self, batch: list[dict]) -> list[dict]:
        """覆盖标准的 process 方法, 因为最后一个阶段需要进行失败处理"""
        results = []
        for item in batch:
            # 无论成功失败, 都需要恢复 image url 字段, 因为可能还会再 roll 一轮数据
            # 恢复原始图像 URL（从 base64 恢复为相对路径）
            original_urls = (item.get("metadata") or {}).get("original_image_urls")
            if original_urls:
                item["messages"] = _restore_image_urls(item["messages"], original_urls)
            
            
            if item.get("_failed") is True:
                results.append(item)
                continue

            try:
                result = self.process_item(item)
                results.append(result)
            except Exception as e:
                import traceback

                error_trace = traceback.format_exc()
                logger.error(
                    f"[{type(self).__name__}] ❌ Item {item.get('id', 'unknown')} failed: {e}\n{error_trace}"
                )
                results.append(
                    {
                        **item,
                        "_failed": True,
                        "_error": f"{type(self).__name__}: {e}",
                        "_traceback": error_trace,
                    }
                )
        return results
    
    def process_item(self, item: dict) -> dict:
        """
        处理单个 item, 格式化为 SFT 训练数据.
        """
        if (item.get("metadata") or {}).get("used_ground_truth") is False:
            logger.info(f"[FormatterStage] Item {item.get('id', 'unknown')}: Already has valid output, skipping")
            item["metadata"]["skipped"] = True
            return item
        
        messages = item.get("messages", [])
        rollouts = item.get("rollouts") or []
        metadata = item.get("metadata", {})
        item_id = item.get("id", "unknown")
        
        # 选择通过验证的 responses
        passed = [r for r in rollouts if r.get("score", 0) >= self.config.pass_threshold]
        logger.info(f"[FormatterStage] Item {item_id}: {len(passed)}/{len(rollouts)} passed")
        
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
                logger.info(f"[FormatterStage] Item {item_id}: Using ground truth")
            else:
                # 主动标记失败
                logger.warning(f"[FormatterStage] Item {item_id}: No valid response")
                return {
                    **item,
                    "_failed": True,
                    "_error": "No response passed and no ground truth",
                }
        
        # 构建 SFT 格式
        sft_messages = messages + [{"role": "assistant", "content": best_response}]
        
        # 清理 metadata（移除内部调试信息）
        clean_metadata = {k: v for k, v in metadata.items() 
                         if k not in ["image_paths", "original_image_urls", "judge_prompt_sample", "judge_output_sample"]}
        clean_metadata.update({
            "n_passed": len(passed),
            "n_total": len(rollouts),
            "used_ground_truth": used_gt,
        })
        
        result = {}
        result = {
            **item,
            "messages": sft_messages,
            "metadata": clean_metadata,
        }

        # 输出前确保 id / messages / metadata 按顺序排列
        ordered_result = {
            "id": result["id"],
            "messages": result["messages"],
            "metadata": result["metadata"],
        }
        # 添加其他字段（除了已经包含的 id, messages, metadata）
        for key in result:
            if key not in ["id", "messages", "metadata"]:
                ordered_result[key] = result[key]
        result = ordered_result
        return result

# ============================================================
# Recipe 定义
# ============================================================

class SFTRecipe(BaseRecipe):
    """
    """
    
    config_class = SFTConfig
    
    def __init__(self, config: SFTConfig):
        super().__init__(config)
    
    def stages(self) -> list[Stage]:
        """返回 Stage 列表 (按执行顺序)."""
        return [
            DataConverterStage(self.config),
            ParseStage(self.config),
            SamplerStage(self.config),
            VerifierStage(self.config),
            FormatterStage(self.config),
        ]


if __name__ == "__main__":
    config = SFTConfig()

    import json
    
    data_converter = DataConverterStage(config)
   
    test_file = "/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/tests/_testsvl_cot_sft_plus_parse.jsonl"
    batch = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                batch.append(json.loads(line))

    results = data_converter.process(batch)
    for result in results:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 80)
    
    import sys
    res = sys.modules['recipes.vl_cot_sft_plus_parse.recipe'] is sys.modules['__main__']
    print(res)
    from recipes.vl_cot_sft_plus_parse.recipe import DataConverterStage
    res = isinstance(data_converter, DataConverterStage)
    print(res)
    print(type(data_converter).__name__)
    print(DataConverterStage.__name__)
    print('-------------------')
    print(sys.modules)