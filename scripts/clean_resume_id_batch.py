#!/usr/bin/env python3
"""
批量清理目录下所有 JSONL 文件中的 _resume_id 字段

用途：
  批量处理整个目录中的所有 JSONL 文件，移除 _resume_id 字段

用法：
  python scripts/clean_resume_id_batch.py -i <input_dir> -o <output_dir>
  
  示例：
  python scripts/clean_resume_id_batch.py -i data/raw -o data/cleaned

注意：
  - 会递归处理输入目录下的所有 .jsonl 文件
  - 保持原有的目录结构
  - 自动创建输出目录
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple


def clean_resume_id_from_file(input_path: str, output_path: str, verbose: bool = True) -> Tuple[int, int, int]:
    """
    清理单个 JSONL 文件中的 _resume_id 字段。
    
    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        verbose: 是否显示详细信息
        
    Returns:
        (total_lines, cleaned_lines, error_lines)
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    total_lines = 0
    cleaned_lines = 0
    error_lines = 0
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            total_lines += 1
            
            try:
                item = json.loads(line)
                
                # 移除 _resume_id 字段
                if "_resume_id" in item:
                    del item["_resume_id"]
                    cleaned_lines += 1
                
                # 写入清理后的数据
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                
            except json.JSONDecodeError as e:
                error_lines += 1
                if verbose:
                    print(f"  [Warning] Line {line_num}: JSON decode error - {e}")
                continue
    
    return total_lines, cleaned_lines, error_lines


def clean_directory(input_dir: str, output_dir: str, verbose: bool = True):
    """
    批量清理目录下所有 JSONL 文件中的 _resume_id 字段。
    
    Args:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        verbose: 是否显示详细信息
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    
    # 查找所有 .jsonl 文件
    jsonl_files = list(input_path.rglob("*.jsonl"))
    
    if not jsonl_files:
        print(f"[Warning] No .jsonl files found in: {input_dir}")
        return
    
    print(f"[Info] Found {len(jsonl_files)} JSONL file(s) in: {input_dir}")
    print(f"[Info] Output directory: {output_dir}")
    print()
    
    total_files_processed = 0
    total_files_with_resume_id = 0
    total_lines_all = 0
    total_cleaned_all = 0
    total_errors_all = 0
    
    for jsonl_file in sorted(jsonl_files):
        # 计算相对路径
        rel_path = jsonl_file.relative_to(input_path)
        output_file = output_path / rel_path
        
        if verbose:
            print(f"[Processing] {rel_path}")
        
        try:
            total_lines, cleaned_lines, error_lines = clean_resume_id_from_file(
                str(jsonl_file), 
                str(output_file), 
                verbose=verbose
            )
            
            total_files_processed += 1
            total_lines_all += total_lines
            total_cleaned_all += cleaned_lines
            total_errors_all += error_lines
            
            if cleaned_lines > 0:
                total_files_with_resume_id += 1
            
            if verbose:
                print(f"  ✓ Lines: {total_lines}, Cleaned: {cleaned_lines}, Errors: {error_lines}")
                print()
        
        except Exception as e:
            print(f"  ✗ Error processing {rel_path}: {e}")
            print()
    
    # 打印总结
    print("=" * 60)
    print("[Summary]")
    print(f"  Total files processed:     {total_files_processed}")
    print(f"  Files with _resume_id:     {total_files_with_resume_id}")
    print(f"  Total lines:               {total_lines_all}")
    print(f"  Total cleaned lines:       {total_cleaned_all}")
    print(f"  Total error lines:         {total_errors_all}")
    print(f"  Output directory:          {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="批量清理目录下所有 JSONL 文件中的 _resume_id 字段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 批量清理整个目录
  python scripts/clean_resume_id_batch.py -i data/raw -o data/cleaned
  
  # 安静模式
  python scripts/clean_resume_id_batch.py -i data/raw -o data/cleaned --quiet
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        dest="input_dir",
        help="输入目录路径"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        dest="output_dir",
        help="输出目录路径"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安静模式，只显示摘要信息"
    )
    
    args = parser.parse_args()
    
    # 检查输入输出目录是否相同
    input_abs = os.path.abspath(args.input_dir)
    output_abs = os.path.abspath(args.output_dir)
    
    if input_abs == output_abs:
        parser.error("输出目录不能与输入目录相同")
    
    # 检查输出目录是否在输入目录内
    if output_abs.startswith(input_abs + os.sep):
        print("[Warning] 输出目录在输入目录内，这可能导致递归处理问题")
        response = input("是否继续？(y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            return
    
    clean_directory(args.input_dir, args.output_dir, verbose=not args.quiet)
    print("\n[Done] 所有文件处理完成！")


if __name__ == "__main__":
    main()

