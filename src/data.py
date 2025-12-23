from typing import Literal, Dict, Any, Tuple, List
from datasets import load_dataset, DatasetDict, Dataset

from .label_maps import (
    CLARITY_TO_ID,
    EVASION_TO_ID,
)


def load_qevasion_hf() -> DatasetDict:
    """
    Load the original QEvasion DatasetDict from Hugging Face.
    """
    return load_dataset("ailsntua/QEvasion")


def build_text(question: str, answer: str) -> str:
    """
    Build the unified text field = Question + Answer.
    Mirrors what you do in the baselines and transformer notebooks.
    """
    return f"Question: {question}\nAnswer: {answer}"


def _encode_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add integer-encoded labels for clarity and evasion on TRAIN examples.
    For TEST, evasion_label can be empty; we still encode clarity.
    """
    clarity_label = example.get("clarity_label")
    evasion_label = example.get("evasion_label")

    example["clarity_id"] = CLARITY_TO_ID.get(clarity_label, -1)

    if evasion_label is not None and evasion_label != "":
        example["evasion_id"] = EVASION_TO_ID.get(evasion_label, -1)
    else:
        example["evasion_id"] = -1

    return example


def prepare_split(ds: Dataset, split_name: str) -> Dataset:
    """
    For a given split:
    - Create 'text' = Question + Answer
    - Encode labels to ids
    """
    def _map_fn(example):
        text = build_text(
            question=example["interview_question"],
            answer=example["interview_answer"],
        )
        example["text"] = text
        example = _encode_labels(example)
        return example

    return ds.map(_map_fn)


def load_qevasion_prepared() -> DatasetDict:
    """
    Return a DatasetDict with:
    - train and test
    - 'text', 'clarity_id', 'evasion_id' added
    """
    raw = load_qevasion_hf()
    train_prep = prepare_split(raw["train"], "train")
    test_prep = prepare_split(raw["test"], "test")
    return DatasetDict({"train": train_prep, "test": test_prep})


def get_text_and_labels(
    split: Literal["train", "test"],
    task: Literal["clarity", "evasion"],
) -> Tuple[List[str], List[int]]:
    """
    Convenience helper to get (texts, label_ids) for a task/split.
    For Task 2 on test, 'evasion_id' is -1 (no single gold); you will
    use annotator columns for evaluation instead.
    """
    ds_dict = load_qevasion_prepared()
    ds = ds_dict[split]

    texts = ds["text"]
    if task == "clarity":
        labels = ds["clarity_id"]
    else:
        labels = ds["evasion_id"]

    return texts, labels
