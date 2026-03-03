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
def process_fn_ddm_to_messages(example: Dict, idx: int) -> Dict:
    """
    处理单个样本的函数
    
    Args:
        example: 原始数据样本
        idx: 样本索引
    
    Returns:
        处理后的样本
    """
    new_dict = {}
    new_dict['id'] = example['id_ddm']
    new_dict['messages'] = example['dialogs']
    new_dict['doc_loc'] = ''
    new_dict['track_loc'] = ['']

    return new_dict

def process_fn_messages_to_ddm(example: Dict, idx: int) -> Dict:
    """
    处理单个样本的函数
    
    Args:
        example: 原始数据样本
        idx: 样本索引
    
    Returns:
        处理后的样本
    """
    new_dict = {}
    new_dict['id_ddm'] = example.get('id') or example.get('_id')
    new_dict['dialogs'] = example['messages']
    return new_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='数据处理脚本')
    # Here! 填写输入输出
    parser.add_argument('--input_path', type=str, default='/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/delivery/instruction_following_ifbench/collect/feasible_and_pass/feasible_and_pass.jsonl', help='输入文件路径')
    parser.add_argument('--output_path', type=str, default='/mnt/shared-storage-user/songdemin/user/guoxu/public/rejection-sampling-recipes/delivery/instruction_following_ifbench/collect/feasible_and_pass_messages/feasible_and_pass.jsonl', help='输出文件路径')
    parser.add_argument('--process_fn', type=str, default='ddm_to_messages', choices=['ddm_to_messages', 'messages_to_ddm'], help='处理函数')
    
    args = parser.parse_args()
    
    # 读取数据
    data = read_jsonl(args.input_path)
    print(f'读取到 {len(data)} 条数据')
    
    # 处理数据
    if args.process_fn == 'ddm_to_messages':
        data = [process_fn_ddm_to_messages(item, idx) for idx, item in tqdm(enumerate(data))]
    elif args.process_fn == 'messages_to_ddm':
        data = [process_fn_messages_to_ddm(item, idx) for idx, item in tqdm(enumerate(data))]
    else:
        raise ValueError(f'处理函数 {args.process_fn} 不存在')
    
    # 保存数据
    write_jsonl(data, args.output_path)
    print(f'处理完成，共 {len(data)} 条数据')
