"""Data I/O utilities."""

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd


def iter_jsonl(path: str) -> Iterator[dict]:
    """
    Stream-read a JSONL file.

    Args:
        path: file path

    Yields:
        dict parsed from each line
    """
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, items: list[dict], append: bool = False):
    """
    Write a JSONL file.

    Args:
        path: file path
        items: list of items to write
        append: whether to append
    """
    mode = "a" if append else "w"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode) as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def convert_to_python_types(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas types to native Python types.

    This avoids JSON serialization failures when Ray Data materializes pandas
    objects that carry numpy scalar types.

    Args:
        obj: object to convert (any type)

    Returns:
        converted native Python object

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


def get_nested_value(item: dict, field_path: str):
    """
    Get a nested field value.

    Args:
        item: data dict
        field_path: dotted path, e.g. "metadata._uid" or "id"

    Returns:
        field value, or None if missing

    Examples:
        >>> get_nested_value({"a": {"b": 1}}, "a.b")
        1
        >>> get_nested_value({"a": {"b": 1}}, "a.c")
        None
    """
    parts = field_path.split(".")
    value = item
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None
    return value


def convert_scalar_to_python(obj: Any) -> Any:
    """
    Convert a scalar value to a native Python type.

    Useful for comparisons (e.g., set membership). This is lighter than
    convert_to_python_types and does not recurse into containers.

    Args:
        obj: scalar value to convert

    Returns:
        converted native Python value

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
