import os
import pandas as pd
import json
import argparse
from tqdm import tqdm
from typing import Dict

def write_jsonl(data, datapath):
    os.makedirs( os.path.dirname(datapath) ,exist_ok=True)
    print(f'saving file at {datapath}')
    with open(datapath, "w", encoding='utf-8') as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")

def write_json(data, datapath):
    os.makedirs(os.path.dirname(datapath),exist_ok=True)
    print(f'saving file at {datapath}')
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(datapath, "w", encoding='utf-8') as json_file:
        json_file.write(json_str)

def read_jsonl(datapath):
    res = []
    print(f'reading file at {datapath}')
    with open(datapath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res

def read_json(datapath):
    print(f'reading file at {datapath}')
    with open(datapath, "r", encoding='utf-8') as f:
        res = json.load(f)
    return res

def write_parquet(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    print(f'saving parquet file at {datapath}')
    df = pd.DataFrame(data)
    df.to_parquet(datapath, index=False)

### Here! 修改处理函数
def process_fn(example: Dict, file_with_abs_path_dict: Dict) -> Dict:
    """
    处理单个样本的函数
    
    Args:
        example: 原始数据样本
        file_with_abs_path_dict: 包含绝对路径的样本
    
    Returns:
        处理后的样本
    """
    assert example['id'] == file_with_abs_path_dict['id']
    if file_with_abs_path_dict.get('abs_path') is not None:
        example['abs_path'] = file_with_abs_path_dict['abs_path']
    
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据处理脚本')
    # Here! 填写输入输出
    parser.add_argument('--input_path', type=str, default='', help='输入文件路径')
    parser.add_argument('--file_with_abs_path', type=str, default='', help='输入文件路径')
    parser.add_argument('--output_path', type=str, default='', help='输出文件路径')
    
    args = parser.parse_args()
    
    # 读取数据
    data = read_jsonl(args.input_path)
    file_with_abs_path = read_jsonl(args.file_with_abs_path)
    assert len(data) == len(file_with_abs_path)
    print(f'读取到 {len(data)} 条数据')
    
    # 处理数据
    data = [process_fn(item, file_with_abs_path[idx]) for idx, item in tqdm(enumerate(data))]
    
    # 保存数据
    write_jsonl(data, args.output_path)
    print(f'处理完成，共 {len(data)} 条数据')
