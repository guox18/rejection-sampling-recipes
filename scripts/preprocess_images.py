#!/usr/bin/env python3
"""
图像路径预处理脚本.

功能:
1. 读取 JSONL 文件
2. 为每个包含图像的数据项计算图片的绝对路径
3. 将绝对路径存储到 meta_info.abs_image_path 字段
4. 检查图像文件是否存在，不存在的发出告警
5. 生成新的 JSONL 文件 (xxx_abs.jsonl)

Usage:
    # 方式1：直接指定完整图片路径（不需要 doc_loc）
    python scripts/preprocess_images.py \
        --input data/train.jsonl data/test.jsonl \
        --image-base-path /mnt/.../internxx/P~xxx~1.0.0~0.0/multimodal_elements \
        [--abs-image-path-field abs_path]
    
    # 方式2：使用基础目录从 doc_loc 推断（需要数据有 doc_loc 字段）
    python scripts/preprocess_images.py \
        --input data/train.jsonl data/test.jsonl \
        --image-base-dir /mnt/.../internxx \
        [--abs-image-path-field abs_path]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional


def infer_image_base_path_from_doc_loc(doc_loc: str, image_base_dir: str) -> Optional[str]:
    """
    从 doc_loc 自动推断图片基础路径.

    Args:
        doc_loc: S3 路径，格式如 "s3://.../P~xxx~1.0.0~0.0_suffix/jsonl/part-001.jsonl"
        image_base_dir: 图片文件的基础目录

    Returns:
        图片完整路径，格式如 "{image_base_dir}/P~xxx~1.0.0~0.0/multimodal_elements"
        如果无法推断则返回 None

    示例:
        输入: "s3://.../P~Document_QA~unknown~xxx~1.0.0~0.0_Bo1f7/jsonl/part-001.jsonl"
        配置: image_base_dir = "/mnt/.../internxx"
        输出: "/mnt/.../internxx/P~Document_QA~unknown~xxx~1.0.0~0.0/multimodal_elements"

    注意:
        doc_loc 中的数据集名称可能包含随机后缀（如 _Bo1f7-xxx），
        实际目录名只到版本号为止（~1.0.0~0.0），需要去掉后缀。
    """
    if not image_base_dir:
        return None

    # 正则匹配：提取数据集名称（P~ 开头到版本号为止）
    # 格式: P~xxx~xxx~xxx~x.x.x~x.x
    pattern = r"(P~[^/]+?~\d+\.\d+\.\d+~\d+\.\d+)(?:_[^/]+)?/jsonl/"
    match = re.search(pattern, doc_loc)

    if not match:
        return None

    dataset_name = match.group(1)
    image_path = os.path.join(image_base_dir, dataset_name, "multimodal_elements")

    return image_path


def has_image_content(item: dict) -> bool:
    """
    检查数据项是否包含图像内容.

    Args:
        item: 数据项

    Returns:
        True 表示包含图像，False 表示纯文本
    """
    for msg in item.get("messages", []):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])

        # content 可能是字符串（纯文本格式）或列表（结构化格式）
        if isinstance(content, str):
            continue  # 字符串格式不包含图像

        # 列表格式：检查是否有 image_url 类型的元素
        for content_item in content:
            if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                return True

    return False


def extract_image_relative_paths(item: dict) -> list[str]:
    """
    提取数据项中的所有图像相对路径.

    Args:
        item: 数据项

    Returns:
        图像相对路径列表
    """
    image_paths = []

    for msg in item.get("messages", []):
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])

        # content 必须是列表格式才可能包含图像
        if not isinstance(content, list):
            continue

        for content_item in content:
            if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                image_url_data = content_item.get("image_url", {})
                relative_path = image_url_data.get("url", "")
                if relative_path:
                    image_paths.append(relative_path)

    return image_paths


def set_nested_field(item: dict, field_path: str, value) -> None:
    """
    设置嵌套字段的值.

    Args:
        item: 数据字典
        field_path: 字段路径，如 "meta_info.abs_image_path" 或 "abs_path"
        value: 要设置的值
    """
    parts = field_path.split(".")
    current = item

    # 遍历到倒数第二层，确保中间字典存在
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        current = current[part]

    # 设置最后一层的值
    current[parts[-1]] = value


def preprocess_file(
    input_path: str,
    output_path: str,
    image_base_dir: str,
    abs_image_path_field: str,
    image_base_path: str = None,
) -> dict:
    """
    预处理单个 JSONL 文件.

    Args:
        input_path: 输入 JSONL 文件路径
        output_path: 输出 JSONL 文件路径
        image_base_dir: 图片文件的基础目录（用于从 doc_loc 推断）
        abs_image_path_field: 存储绝对路径的字段名（支持嵌套）
        image_base_path: 图片的完整路径（可选，如果指定则不从 doc_loc 推断）

    Returns:
        统计信息字典
    """
    stats = {
        "total_items": 0,
        "items_with_images": 0,
        "items_without_images": 0,
        "total_image_files": 0,
        "image_files_exist": 0,
        "image_files_missing": 0,
        "missing_image_details": [],  # 存储缺失图像的详细信息
    }

    with (
        open(input_path, "r", encoding="utf-8") as fin,
        open(output_path, "w", encoding="utf-8") as fout,
    ):
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                stats["total_items"] += 1

                # 检查是否包含图像
                if not has_image_content(item):
                    stats["items_without_images"] += 1
                    # 纯文本数据：直接写入，不处理
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    continue

                stats["items_with_images"] += 1

                # 获取图片基础路径
                if image_base_path:
                    # 用户直接指定了完整路径
                    item_image_base_path = image_base_path
                else:
                    # 从 doc_loc 推断
                    doc_loc = item.get("doc_loc", "")
                    item_image_base_path = infer_image_base_path_from_doc_loc(
                        doc_loc, image_base_dir
                    )

                    if not item_image_base_path:
                        print(f"⚠️  行 {line_num}: 无法推断图片路径，跳过")
                        print(f"   doc_loc: {doc_loc}")
                        print(
                            f"   提示: 如果数据没有 doc_loc 字段，请使用 --image-base-path 直接指定完整路径"
                        )
                        continue

                # 提取所有图像相对路径
                relative_paths = extract_image_relative_paths(item)

                if not relative_paths:
                    stats["items_without_images"] += 1
                    stats["items_with_images"] -= 1
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                    continue

                # 检查图像文件是否存在
                all_images_exist = True
                for relative_path in relative_paths:
                    full_path = os.path.join(item_image_base_path, relative_path)
                    stats["total_image_files"] += 1

                    if os.path.exists(full_path):
                        stats["image_files_exist"] += 1
                    else:
                        stats["image_files_missing"] += 1
                        stats["missing_image_details"].append(
                            {
                                "line": line_num,
                                "item_id": item.get("id", "unknown"),
                                "relative_path": relative_path,
                                "full_path": full_path,
                            }
                        )
                        all_images_exist = False
                        print(f"❌ 行 {line_num}: 图像文件不存在")
                        print(f"   ID: {item.get('id', 'unknown')}")
                        print(f"   相对路径: {relative_path}")
                        print(f"   完整路径: {full_path}")

                # 将绝对路径存储到指定字段
                set_nested_field(item, abs_image_path_field, item_image_base_path)

                # 写入输出文件
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"⚠️  行 {line_num}: JSON 解析错误 - {e}")
            except Exception as e:
                print(f"⚠️  行 {line_num}: 处理错误 - {e}")

    return stats


def main():
    """主函数."""
    parser = argparse.ArgumentParser(
        description="图像路径预处理脚本：为 JSONL 文件中的数据项添加图片绝对路径",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="输入 JSONL 文件路径（支持多个文件）",
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default=None,
        help="图片文件的基础目录（用于从 doc_loc 推断完整路径），例如 /mnt/.../internxx",
    )
    parser.add_argument(
        "--image-base-path",
        type=str,
        default=None,
        help="图片的完整路径（如果指定则不从 doc_loc 推断），例如 /mnt/.../internxx/P~xxx~1.0.0~0.0/multimodal_elements",
    )
    parser.add_argument(
        "--abs-image-path-field",
        type=str,
        default="abs_path",
        help="存储绝对路径的字段名（支持嵌套，如 meta_info.abs_image_path 或 abs_path）",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_abs",
        help="输出文件名后缀，例如 train.jsonl -> train_abs.jsonl",
    )

    args = parser.parse_args()

    # 检查输入文件
    input_files = args.input
    for input_file in input_files:
        if not os.path.exists(input_file):
            print(f"❌ 输入文件不存在: {input_file}")
            return

    # 检查图片路径配置
    if not args.image_base_dir and not args.image_base_path:
        print(f"❌ 必须指定 --image-base-dir 或 --image-base-path 之一")
        return

    if args.image_base_dir and not os.path.exists(args.image_base_dir):
        print(f"❌ 图片基础目录不存在: {args.image_base_dir}")
        return

    if args.image_base_path and not os.path.exists(args.image_base_path):
        print(f"❌ 图片完整路径不存在: {args.image_base_path}")
        return

    print("=" * 80)
    print("图像路径预处理脚本")
    print("=" * 80)
    print(f"输入文件数量: {len(input_files)}")
    if args.image_base_path:
        print(f"图片完整路径: {args.image_base_path} (直接指定)")
    else:
        print(f"图片基础目录: {args.image_base_dir} (从 doc_loc 推断)")
    print(f"绝对路径字段: {args.abs_image_path_field}")
    print(f"输出文件后缀: {args.output_suffix}")
    print("=" * 80)
    print()

    # 处理每个文件
    total_stats = {
        "total_items": 0,
        "items_with_images": 0,
        "items_without_images": 0,
        "total_image_files": 0,
        "image_files_exist": 0,
        "image_files_missing": 0,
        "missing_image_details": [],
    }

    for input_file in input_files:
        print(f"\n{'=' * 80}")
        print(f"处理文件: {input_file}")
        print(f"{'=' * 80}")

        # 生成输出文件路径
        input_path = Path(input_file)
        output_filename = f"{input_path.stem}{args.output_suffix}{input_path.suffix}"
        output_path = input_path.parent / output_filename

        print(f"输出文件: {output_path}")

        # 预处理文件
        stats = preprocess_file(
            input_path=str(input_path),
            output_path=str(output_path),
            image_base_dir=args.image_base_dir,
            abs_image_path_field=args.abs_image_path_field,
            image_base_path=args.image_base_path,
        )

        # 打印统计信息
        print(f"\n文件统计:")
        print(f"  总数据项:     {stats['total_items']}")
        print(f"  包含图像:     {stats['items_with_images']}")
        print(f"  不含图像:     {stats['items_without_images']}")
        print(f"  图像文件总数: {stats['total_image_files']}")
        print(f"  图像存在:     {stats['image_files_exist']}")
        print(f"  图像缺失:     {stats['image_files_missing']}")

        if stats["image_files_missing"] > 0:
            print(f"\n⚠️  警告: 发现 {stats['image_files_missing']} 个缺失的图像文件")

        # 累计统计
        for key in total_stats:
            if key == "missing_image_details":
                total_stats[key].extend(stats[key])
            else:
                total_stats[key] += stats[key]

    # 打印总体统计
    print(f"\n\n{'=' * 80}")
    print("总体统计")
    print(f"{'=' * 80}")
    print(f"总数据项:     {total_stats['total_items']}")
    print(f"包含图像:     {total_stats['items_with_images']}")
    print(f"不含图像:     {total_stats['items_without_images']}")
    print(f"图像文件总数: {total_stats['total_image_files']}")
    print(f"图像存在:     {total_stats['image_files_exist']}")
    print(f"图像缺失:     {total_stats['image_files_missing']}")

    if total_stats["total_image_files"] > 0:
        exist_rate = 100 * total_stats["image_files_exist"] / total_stats["total_image_files"]
        print(f"图像存在率:   {exist_rate:.2f}%")

    # 打印缺失图像详情
    if total_stats["missing_image_details"]:
        print(f"\n\n{'=' * 80}")
        print("缺失图像详情（前 20 条）")
        print(f"{'=' * 80}")
        for detail in total_stats["missing_image_details"][:20]:
            print(f"  行 {detail['line']}, ID {detail['item_id']}")
            print(f"    相对路径: {detail['relative_path']}")
            print(f"    完整路径: {detail['full_path']}")
            print()

        if len(total_stats["missing_image_details"]) > 20:
            print(f"  ... 还有 {len(total_stats['missing_image_details']) - 20} 条缺失记录")

    print(f"\n{'=' * 80}")
    print("✅ 预处理完成")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
