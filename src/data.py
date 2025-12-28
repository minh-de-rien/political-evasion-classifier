"""
Data loading and preprocessing utilities for QEvasion dataset.
"""
from typing import Tuple
from datasets import load_dataset, DatasetDict, Dataset
import pandas as pd
import numpy as np

# Label mappings
CLARITY_LABELS = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
EVASION_LABELS = [
    'Claims ignorance',
    'Clarification', 
    'Declining to answer',
    'Deflection',
    'Dodging',
    'Explicit',
    'General',
    'Implicit',
    'Partial/half-answer'
]

CLARITY_TO_ID = {label: idx for idx, label in enumerate(CLARITY_LABELS)}
ID_TO_CLARITY = {idx: label for label, idx in CLARITY_TO_ID.items()}

EVASION_TO_ID = {label: idx for idx, label in enumerate(EVASION_LABELS)}
ID_TO_EVASION = {idx: label for label, idx in EVASION_TO_ID.items()}


def load_qevasion_hf() -> DatasetDict:
    """Load the original QEvasion dataset from Hugging Face."""
    return load_dataset("ailsntua/QEvasion")


def _ensure_expected_columns(ds: Dataset) -> None:
    """Validate that dataset has expected columns."""
    expected = {
        "interview_question",
        "interview_answer",
        "clarity_label",
        "evasion_label",
    }
    missing = expected.difference(ds.column_names)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def build_text_column(df: pd.DataFrame, sep_token: str = "[SEP]") -> pd.DataFrame:
    """
    Build unified text input from question and answer.
    
    Args:
        df: DataFrame with 'interview_question' and 'interview_answer' columns
        sep_token: Separator token between question and answer
        
    Returns:
        DataFrame with added 'text' column
    """
    df = df.copy()
    q = df["interview_question"].fillna("")
    a = df["interview_answer"].fillna("")
    df["text"] = f"Question: " + q + f" {sep_token} Answer: " + a
    return df


def add_label_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add integer label ID columns for clarity and evasion.
    
    Args:
        df: DataFrame with 'clarity_label' and optionally 'evasion_label'
        
    Returns:
        DataFrame with added 'clarity_id' and 'evasion_id' columns
    """
    df = df.copy()
    
    # Add clarity IDs
    df["clarity_id"] = df["clarity_label"].map(CLARITY_TO_ID)
    
    # Add evasion IDs (handle missing values)
    mask_valid = df["evasion_label"].notna() & (df["evasion_label"] != "")
    df["evasion_id"] = np.where(
        mask_valid,
        df["evasion_label"].map(EVASION_TO_ID),
        -1  # Invalid label marker
    )
    
    return df


def add_text_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add text length statistics.
    
    Args:
        df: DataFrame with 'interview_question' and 'interview_answer'
        
    Returns:
        DataFrame with added length columns
    """
    df = df.copy()
    df['q_len'] = df['interview_question'].fillna('').str.split().str.len()
    df['a_len'] = df['interview_answer'].fillna('').str.split().str.len()
    df['qa_len'] = df['q_len'] + df['a_len']
    return df


def get_annotator_labels(row: pd.Series) -> set:
    """
    Extract set of annotator labels from test set row.
    
    Args:
        row: DataFrame row with annotator1/2/3 columns
        
    Returns:
        Set of valid annotator labels (excluding empty/NaN)
    """
    labels = []
    for col in ["annotator1", "annotator2", "annotator3"]:
        val = row.get(col)
        if pd.notna(val) and val != "":
            labels.append(val)
    return set(labels)


def prepare_task1_data(
    dataset: DatasetDict,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train/val/test splits for Task 1 (Clarity).
    
    Args:
        dataset: Original QEvasion dataset
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    # Convert to pandas
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Filter valid clarity labels
    train_df = train_df[
        train_df["clarity_label"].notna() & 
        (train_df["clarity_label"] != "")
    ].reset_index(drop=True)
    
    test_df = test_df[
        test_df["clarity_label"].notna() & 
        (test_df["clarity_label"] != "")
    ].reset_index(drop=True)
    
    # Add preprocessing
    train_df = build_text_column(train_df)
    train_df = add_label_ids(train_df)
    train_df = add_text_statistics(train_df)
    
    test_df = build_text_column(test_df)
    test_df = add_label_ids(test_df)
    test_df = add_text_statistics(test_df)
    
    # Split train into train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=val_size,
        stratify=train_df["clarity_id"].values,
        random_state=random_state
    )
    
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_df = train_df.iloc[train_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df


def prepare_task2_data(
    dataset: DatasetDict,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare train/val/test splits for Task 2 (Evasion).
    
    Note: Test split has multiple annotators per example.
    
    Args:
        dataset: Original QEvasion dataset
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    from sklearn.model_selection import train_test_split
    
    # Convert to pandas
    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()
    
    # Filter valid evasion labels (only in train)
    train_df = train_df[
        train_df["evasion_label"].notna() & 
        (train_df["evasion_label"] != "")
    ].reset_index(drop=True)
    
    # Add preprocessing
    train_df = build_text_column(train_df)
    train_df = add_label_ids(train_df)
    train_df = add_text_statistics(train_df)
    
    # Test set: add text and extract annotator labels
    test_df = build_text_column(test_df)
    test_df["annotator_labels"] = test_df.apply(get_annotator_labels, axis=1)
    
    # Filter test rows with at least one annotator label
    test_df = test_df[
        test_df["annotator_labels"].apply(len) > 0
    ].reset_index(drop=True)
    
    # Split train into train/val
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=val_size,
        stratify=train_df["evasion_id"].values,
        random_state=random_state
    )
    
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    train_df = train_df.iloc[train_idx].reset_index(drop=True)
    
    return train_df, val_df, test_df


def load_qevasion_prepared() -> DatasetDict:
    """
    Load QEvasion with basic validation.
    
    Returns:
        Original DatasetDict from Hugging Face
    """
    dataset = load_qevasion_hf()
    _ensure_expected_columns(dataset["train"])
    _ensure_expected_columns(dataset["test"])
    return dataset