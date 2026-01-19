from openai import OpenAI
from dotenv import load_dotenv
import json
import re
import threading


def load_jsonl(file_path: str) -> list:
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def write_jsonl_with_lock(file_path: str, data: dict):
    lock = threading.Lock()
    with lock:
        with open(file_path, "a+", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False))
            f.write("\n")
            f.flush()


def write_jsonl(file_path: str, data: list):
    with open(file_path, "a+", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False))
            f.write("\n")


def is_valid_json(s: str) -> bool:
    """
    Checks if a string is valid JSON.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(s)
        return True
    except json.JSONDecodeError:
        return False


def parse_json(s: str) -> dict:
    """
    Parses a JSON string into a dictionary.

    Args:
        s (str): The JSON string to parse.

    Returns:
        dict: The parsed dictionary.
    """
    # 移除对象末尾的多余逗号: ,}
    s = re.sub(r",\s*}", "}", s)

    # 移除数组末尾的多余逗号: ,]
    s = re.sub(r",\s*]", "]", s)
    if is_valid_json(s):
        return json.loads(s)
    else:
        # Try to extract JSON from markdown code block first
        markdown_pattern = r"```(?:json)?\s*(\{.*\})\s*```"
        markdown_match = re.search(markdown_pattern, s, re.DOTALL)
        if markdown_match:
            json_str = markdown_match.group(1)
            if is_valid_json(json_str):
                return json.loads(json_str)

        # Fallback to original pattern matching
        pattern = r"\{.*\}"
        match = re.search(pattern, s, re.DOTALL)
        if match:
            json_str = match.group(0)
            if is_valid_json(json_str):
                return json.loads(json_str)
        else:
            answer_match = re.search(r'\{[^{}]*"answer"[^{}]*\}', s)
            if answer_match:
                json_str = answer_match.group(0)
                if is_valid_json(json_str):
                    return json.loads(json_str)
        return s
