import json
import os
from tqdm import tqdm
from utils import parse_json
import re


def evaluate_multi_options(output_file: str):
    """
    Evaluate the model's performance on multi-option questions.
    """
    f = open(output_file, "r", encoding="utf-8")
    results = f.readlines()
    correct_count = 0
    for i, item in enumerate(tqdm(results)):
        item = json.loads(item.strip())
        response = item["response"]
        response = parse_json(response)
        answer = item["answer_idx"]
        if isinstance(answer, str):
            gt_answer = answer.strip()
        if isinstance(answer, list):
            gt_answer = "\n".join([str(a).strip() for a in answer])
        if isinstance(response, str):
            match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            thinking_match = re.search(r"</think>\s*(.*)", response)
            if match:
                answer = match.group(1)
                if gt_answer == answer:
                    correct_count += 1
            if thinking_match:
                answer = thinking_match.group(1).strip()
                if gt_answer == answer:
                    correct_count += 1
            else:
                if gt_answer == response.strip():
                    correct_count += 1
        elif isinstance(response, dict):
            if gt_answer == response.get("answer"):
                correct_count += 1
    accuracy = correct_count / len(results) * 100
    return accuracy


if __name__ == "__main__":
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    dataset = "MMedBench"
    subset = "Spanish"
    split = "Test"
    output_file = os.path.join(
        f"benchmark/{dataset}/results",
        f"run_round_1",
        f"{subset}",
        f"{model_name}_base_{split}.jsonl",
    )
    print(f"Evaluating {output_file} ...")
    accuracy = evaluate_multi_options(output_file)
    print(f"Accuracy: {accuracy:.2f}%")
