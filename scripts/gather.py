import os
import pandas as pd
import json
import argparse
from tqdm import tqdm
from typing import Dict

def write_jsonl(data, datapath):
    os.makedirs( os.path.dirname(datapath) ,exist_ok=True)
    # print(f'saving file at {datapath}')
    with open(datapath, "w", encoding='utf-8') as f:
        for item in data:
            json_item = json.dumps(item, ensure_ascii=False)
            f.write(json_item + "\n")

def write_json(data, datapath):
    os.makedirs(os.path.dirname(datapath),exist_ok=True)
    # print(f'saving file at {datapath}')
    json_str = json.dumps(data, indent=4, ensure_ascii=False)
    with open(datapath, "w", encoding='utf-8') as json_file:
        json_file.write(json_str)

def read_jsonl(datapath):
    res = []
    # print(f'reading file at {datapath}')
    with open(datapath, "r", encoding='utf-8') as f:
        for line in f.readlines():
            res.append(json.loads(line))
    return res

def read_json(datapath):
    # print(f'reading file at {datapath}')
    with open(datapath, "r", encoding='utf-8') as f:
        res = json.load(f)
    return res

def write_parquet(data, datapath):
    os.makedirs(os.path.dirname(datapath), exist_ok=True)
    # print(f'saving parquet file at {datapath}')
    df = pd.DataFrame(data)
    df.to_parquet(datapath, index=False)

### Here! 修改处理函数
def process_fn(example: Dict, qwen30ba3bfile_dict: Dict, qwen235ba22bfile_dict: Dict) -> Dict:
    """
    处理单个样本的函数
    
    Args:
        example: 原始数据样本
        qwen30ba3bfile: 30b 文件
        qwen235ba22bfile: 235b 文件
    
    Returns:
        处理后的样本
    """
    assert example['id'] == qwen30ba3bfile_dict['id']
    assert example['id'] == qwen235ba22bfile_dict['id']

    def get_assistant_content(example):
        for msg in example.get('messages', []):
            if msg.get('role') == 'assistant':
                return msg.get('content')
        return None
    def set_assistant_content(example, content):
      for msg in example.get('messages', []):
        if msg.get('role') == 'assistant':
          msg['content'] = content
          break
      return example
    
    # 初始化 workload 字段
    workload = {
        'original_assistant_content': get_assistant_content(example),
        'source': 'origin'  # 默认标记为原始数据
    }
    
    qwen235b_metadata = qwen235ba22bfile_dict.get('metadata') or {}
    if qwen235b_metadata.get('used_ground_truth') is False: # 对默认值要格外小心. 这里其实有四种分支, is True, is False, 以及, "值" is None and 不存在. 尽量写 is, 而不是 is not. is not 要考虑的情况太多了
      workload['source'] = 'qwen3vl_235b_a22b_thinking'
      workload['n_passed'] = qwen235ba22bfile_dict['metadata']['n_passed']
      workload['n_total'] = qwen235ba22bfile_dict['metadata']['n_total']
      set_assistant_content(example, get_assistant_content(qwen235ba22bfile_dict))

    qwen30b_metadata = qwen30ba3bfile_dict.get('metadata') or {}
    if qwen30b_metadata.get('used_ground_truth') is False:
      workload['source'] = 'qwen3vl_30b_a3b_thinking'
      workload['n_passed'] = qwen30ba3bfile_dict['metadata']['n_passed']
      workload['n_total'] = qwen30ba3bfile_dict['metadata']['n_total']
      set_assistant_content(example, get_assistant_content(qwen30ba3bfile_dict))
    
    # 两个模型都没有成功生成新数据，保持原始数据
    example['workload'] = workload
    
    return example

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据处理脚本')
    # Here! 填写输入输出
    parser.add_argument('--input_path', type=str, default='', help='输入文件路径')
    parser.add_argument('--qwen3vl30ba3bthinking_path', type=str, default='', help='输入文件路径')
    parser.add_argument('--qwen3vl235ba22bthinking_path', type=str, default='', help='输入文件路径')
    parser.add_argument('--output_path', type=str, default='', help='输出文件路径') # 不填, 默认 _train.jsonl
    
    args = parser.parse_args()
    
    # 读取数据
    data = read_jsonl(args.input_path)
    qwen30ba3bfile = read_jsonl(args.qwen3vl30ba3bthinking_path)
    qwen235ba22bfile = read_jsonl(args.qwen3vl235ba22bthinking_path)
    assert len(data) == len(qwen30ba3bfile)
    assert len(data) == len(qwen235ba22bfile)
    # print(f'读取到 {len(data)} 条数据')
    
    # 处理数据
    data = [process_fn(item, qwen30ba3bfile[idx], qwen235ba22bfile[idx]) for idx, item in tqdm(enumerate(data))]
    
    # 统计成功有多少不是 origin 的
    n_not_origin = sum(1 for item in data if item['workload']['source'] != 'origin')
    n_total = len(data)
    print(f'替换率: {n_not_origin / n_total * 100}%')
    # 保存数据
    write_jsonl(data, args.output_path)
    # print(f'处理完成，共 {len(data)} 条数据')
