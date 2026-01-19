import os
import json
import ast
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from utils import parse_json


def load_medagents(subset: str, split: str = "test") -> list[dict]:
    """
    Load the MedAgents benchmark.

    Args:
        - subset (str): The subset of the MedAgents dataset to load.
         options: AfrimedQA, MMLU-Pro, MMLU, MedBullets, MedExQA, MedMCQA, MedQA, MedXpertQA-R, MedXpertQA-U, PudMedQA, PubMedQA
        - split (str): The split of the dataset to load
         options: test, test_hard

    Returns:
        Dataset: The loaded dataset. (keys: question, options, answer_idx, reason, answer)
    """
    root_path = f"benchmark/medagents/data"
    file_path = os.path.join(root_path, subset, f"{split}.jsonl")
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            dataset.append(
                {
                    "question": data["question"],
                    "options": data["options"],
                    "answer_idx": data["answer_idx"],
                    "reason": data.get("reason", ""),
                    "answer": data["answer"],
                }
            )
    return dataset


def load_mmedbench(subset: str, split: str = "") -> list[dict]:
    """
    Load the MMEDBench dataset.

    Args:
        - subset (str): The subset of the MMEDBench dataset to load.
        - split (str): The split of the dataset to load

    Returns:
        Dataset: The loaded dataset. (keys: question, options, answer_idx, reason, answer)
    """
    root_path = f"benchmark/MMedBench/{split}"
    file_path = os.path.join(root_path, f"{subset}.jsonl")

    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            dataset.append(
                {
                    "question": data["question"],
                    "options": data["options"],
                    "answer_idx": data["answer_idx"],
                    "rationale": data.get("rationale", ""),
                    "answer": data.get("answer", ""),
                }
            )
    return dataset


def load_translated_mmedbench(subset: str, split: str = "") -> list[dict]:
    rootpath = f"benchmark/MMed-Translated/{split}"
    file_path = os.path.join(rootpath, f"{subset}.jsonl")
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            dataset.append(
                {
                    "question": data["question"],
                    "options": data["options"],
                    "answer_idx": data["answer_idx"],
                    "answer": data.get("answer", ""),
                }
            )
    return dataset


def load_medcasereasoning(subset: str = "", split: str = "test") -> list[dict]:
    """
    Load the MedCaseReasoning dataset.

    Args:
        - subset (str): The subset of the MedCaseReasoning dataset to load.
        - split (str): The split of the dataset to load

    Returns:
        Dataset: The loaded dataset. (keys: question, options, answer_idx, reason, answer)
    """
    root_path = "benchmark/medcasereasoning/data"
    file_path = os.path.join(root_path, f"{split}.jsonl")

    # ds = pd.read_parquet(path=os.path.join(root_path, "test-00000-of-00001.parquet"))
    # ds.to_json(file_path, orient="records", lines=True, force_ascii=False)
    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            dataset.append(
                {
                    "question": data["case_prompt"],
                    "reasoning": data.get("diagnostic_reasoning", ""),
                    "answer": data["final_diagnosis"],
                }
            )
    return dataset


def load_curebench(subset: str, split: str = "test") -> list[dict]:
    """
    Load the CureBench dataset.

    Args:
        - subset (str): The subset of the CureBench dataset to load.
        - split (str): The split of the dataset to load

    Returns:
        Dataset: The loaded dataset. (keys: question, options, answer_idx, reason, answer)
    """
    root_path = "benchmark/cure-bench"
    file_path = os.path.join(root_path, f"curebench_{subset}set.jsonl")

    dataset = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            dataset.append(
                {
                    "question": data["question"],
                    "options": data["options"],
                    "answer_idx": data["correct_answer"],
                    "answer": data["options"][data["correct_answer"]],
                }
            )
    return dataset


def load_global_mmlu(
    subset: str, split: str = "test", subject_category: list = ["Medical"], is_lite=True
) -> list[dict]:
    """
    Load the GlobalMMLU Medical dataset.

    Args:
        - subset (str): The subset of the GlobalMMLU Medical dataset to load, language code like 'en', 'zh', etc.
        - split (str): The split of the dataset to load, 'test' or 'dev'

    Returns:
        Dataset: The loaded dataset. (keys: question, options, answer_idx, reason, answer)
    """
    if is_lite:
        dataset_name = "CohereLabs/Global-MMLU-lite"
    else:
        dataset_name = "CohereLabs/Global-MMLU"
    global_mmlu = load_dataset(dataset_name, subset, split=split)
    global_medical = global_mmlu.filter(
        lambda x: x["subject_category"] in subject_category
    )
    dataset = []
    for data in global_medical:
        options = {
            "A": data["option_a"],
            "B": data["option_b"],
            "C": data["option_c"],
            "D": data["option_d"],
        }

        dataset.append(
            {
                "id": data["sample_id"],
                "question": data["question"],
                "options": options,
                "answer_idx": data["answer"],
                "answer": options[data["answer"]],
            }
        )
    return dataset


def load_mmlu_prox(
    subset: str, split: str = "test", category: str = "health", is_lite=True
) -> list[dict]:
    """
    Load the MMLU-Prox dataset.

    Args:
        - subset (str): The subset of the MMLU-Prox dataset to load.
        - split (str): The split of the dataset to load
        - category (str): The category of the dataset to load

    Returns:
        Dataset: The loaded dataset. (keys: question, options, answer_idx, reason, answer)
    """
    if is_lite:
        dataset_name = "li-lab/MMLU-ProX-Lite"
    else:
        dataset_name = "li-lab/MMLU-ProX"
    mmlu_prox = load_dataset(dataset_name, subset, split=split)
    mmlu_prox_category = mmlu_prox.filter(lambda x: x["category"] == category)
    dataset = []
    for data in mmlu_prox_category:
        options = {
            "A": data["option_0"],
            "B": data["option_1"],
            "C": data["option_2"],
            "D": data["option_3"],
            "E": data["option_4"],
            "F": data["option_5"],
            "G": data["option_6"],
            "H": data["option_7"],
            "I": data["option_8"],
            "J": data["option_9"],
        }

        dataset.append(
            {
                "id": data["question_id"],
                "question": data["question"],
                "options": options,
                "answer_idx": data["answer"],
                "answer": options[data["answer"]],
            }
        )
    return dataset


def load_globmedx_mcqa(subset: str, split: str) -> list[dict]:
    """
    Load the GlobalMed dataset.
    Args:
        - subset (str): language subset of the GlobalMed dataset to load.
        - split (str): headqa, medexqa, medqa, mmlu-pro
    """
    dataset = []
    root_path = Path(f"benchmark/gm20251108/{subset}/MCQA/{split} (75)")
    file_name = f"{split}_{subset}_75.xlsx"
    file_path = root_path / file_name
    df = pd.read_excel(file_path, engine="openpyxl")
    idx_map = {
        "0": "A",
        "1": "B",
        "2": "C",
        "3": "D",
        "4": "E",
        "5": "F",
        "6": "G",
        "7": "H",
        "8": "I",
        "9": "J",
    }

    # Process each row
    for idx, row in df.iterrows():
        question_item = row[subset]
        gold_label = row["gold_label"]

        if not isinstance(question_item, dict):
            try:
                if isinstance(question_item, str):
                    # Normalize all Unicode quotes to ASCII quotes using Unicode escape sequences
                    question_item = question_item.replace("\u2018", "'").replace(
                        "\u2019", "'"
                    )
                    # Replace left/right double quotes: " (U+201C) and " (U+201D)
                    question_item = question_item.replace("\u201c", '"').replace(
                        "\u201d", '"'
                    )
                    # Replace other quote-like characters
                    question_item = question_item.replace("\u201a", "'").replace(
                        "\u201e", '"'
                    )
                    question_item = question_item.replace(
                        "`", "'"
                    )  # backtick to single quote

                # Parse as Python dict
                question_item = ast.literal_eval(question_item)
            except (ValueError, SyntaxError) as e:
                print(f"Row {idx} - Failed to parse: {e}")
                continue

        question, options = question_item["question"], question_item["options"]
        answer_key = str(gold_label)
        answer_idx = idx_map[answer_key]
        answer = options[answer_idx]
        dataset.append(
            {
                "question": question,
                "options": options,
                "answer_idx": answer_idx,
                "answer": answer,
            }
        )
    return dataset


def load_globmedx_liveqa(subset: str, split: str) -> list[dict]:
    """
    Load the GlobalMed LiveQA dataset.
    Args:
        - subset (str): language subset of the GlobalMed dataset to load.
        - split (str): liveqa
    """
    makers = {
        "en": {"question_marker": "Question:\n", "answer_marker": "Answer:\n"},
        "zh": {"question_marker": "问题:\n", "answer_marker": "答案:\n"},
        "jp": {"question_marker": "質問:\n", "answer_marker": "回答:\n"},
        "ko": {"question_marker": "질문:\n", "answer_marker": "답변:\n"},
        "th": {"question_marker": "Question:\n", "answer_marker": "Answer:\n"},
        "sw": {"question_marker": "Swali:\n", "answer_marker": "Jibu:\n"},
        "yo": {"question_marker": "Ìbéèrè:\n", "answer_marker": "IÌdáhùn:\n"},
        "zu": {"question_marker": "Umbuzo:\n", "answer_marker": "Impendulo:\n"},
    }
    dataset = []
    root_path = f"benchmark/liveqa_data/liveqa_{subset}_200.xlsx"
    df = pd.read_excel(root_path, engine="openpyxl")
    question_marker = makers[subset]["question_marker"]
    answer_marker = makers[subset]["answer_marker"]

    for idx, row in df.iterrows():
        data_item = row[subset]
        question_part, answer_part = data_item.split(answer_marker, 1)
        question = question_part.replace(question_marker, "").strip()
        answer = answer_part.strip()
        dataset.append(
            {
                "question": question,
                "answer": answer,
            }
        )
    return dataset


def load_globmedx_bionli(subset: str, split: str) -> list[dict]:
    """
    Load the GlobalMed BioNLI dataset.
    Args:
        - subset (str): language subset of the GlobalMed dataset to load.
        - split (str): bionli
    """
    dataset = []
    root_path = f"benchmark/gm20251108/{subset}/"
    if subset != "yo" or subset != "yo":
        file_path = os.path.join(root_path, "NLI/BioNLI", f"bionli_{subset}_150.xlsx")
    else:
        file_path = os.path.join(root_path, f"bionli_{subset}_150.xlsx")
    df = pd.read_excel(file_path, engine="openpyxl")
    for idx, row in df.itterrows():
        if subset == "en":
            collumn_names = "EN"
        else:
            collumn_names = subset
        data_item = row[collumn_names]
        label = row["gold_label"]
        dataset.append(
            {
                "question": data_item,
                "answer": label,
            }
        )
    return dataset


if __name__ == "__main__":
    dataset = load_globmedx_mcqa(subset="de", split="headqa")
    print(len(dataset))
