import json


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
