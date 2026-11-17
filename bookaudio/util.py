import json
import math
from typing import Any


def read_jsonl(filename):
    with open(filename) as jsonl:
        lines: list[str] = jsonl.readlines()
        objects = []
        for (i, line) in enumerate(lines):
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Error parsing line {i+1}: {e.msg}")

        return objects


def load_jsonl_from_string(string: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for line in string.strip().splitlines():
        line = line.strip()
        data = json.loads(line)
        result.append(data)
    return result


def batch_indices(items_length: int, i: int, batch_size: int = 5) -> range:
    """Safe batch index context calculation."""
    if items_length <= batch_size:
        return range(items_length)
    if i < batch_size / 2:
        return range(batch_size)
    elif i >= items_length - batch_size / 2:
        return range(items_length - batch_size, items_length)
    else:
        start = math.ceil(i - batch_size / 2)
        return range(start, start + batch_size)
