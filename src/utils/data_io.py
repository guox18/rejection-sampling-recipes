"""数据读写工具."""

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def iter_jsonl(path: str) -> Iterator[dict]:
    """
    流式读取 jsonl 文件.

    Args:
        path: 文件路径

    Yields:
        每行解析后的 dict
    """
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, items: list[dict], append: bool = False):
    """
    写入 jsonl 文件.

    Args:
        path: 文件路径
        items: 要写入的数据列表
        append: 是否追加模式
    """
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_python_types(obj: Any) -> Any:
    """
    递归转换 numpy/pandas 类型为 Python 原生类型.

    用于解决 Ray Data 使用 pandas 格式时, 某些值会变成 numpy 类型,
    导致 JSON 序列化失败的问题.

    Args:
        obj: 要转换的对象(可以是任意类型)

    Returns:
        转换后的 Python 原生类型对象

    Examples:
        >>> import numpy as np
        >>> convert_to_python_types(np.int64(42))
        42
        >>> convert_to_python_types({'a': np.array([1, 2, 3])})
        {'a': [1, 2, 3]}
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj


def convert_scalar_to_python(obj: Any) -> Any:
    """
    转换单个标量值为 Python 原生类型.

    用于比较操作(如集合成员检查)时, 确保类型一致.
    比 convert_to_python_types 更轻量, 不递归处理容器类型.

    Args:
        obj: 要转换的标量值

    Returns:
        转换后的 Python 原生类型

    Examples:
        >>> import numpy as np
        >>> convert_scalar_to_python(np.int64(42))
        42
        >>> import pandas as pd
        >>> convert_scalar_to_python(pd.NA)
        None
    """
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj
