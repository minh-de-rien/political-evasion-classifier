"""
Evaluation metrics for QEvasion tasks.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import map_evasion_to_clarity, CLARITY_TO_ID


def evaluate_task1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate Task 1 (Clarity Classification).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of label names for report
        verbose: Whether to print detailed report
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'macro_precision': precision_score(y_true, y_pred, average='macro'),
        'macro_recall': recall_score(y_true, y_pred, average='macro')
    }
    
    if verbose:
        print("=" * 60)
        print("TASK 1: CLARITY CLASSIFICATION")
        print("=" * 60)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred, 
            target_names=label_names,
            zero_division=0
        ))
    
    return metrics


def evaluate_task2_standard(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate Task 2 (Evasion Classification) on validation set with single labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional list of label names for report
        verbose: Whether to print detailed report
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0)
    }
    
    if verbose:
        print("=" * 60)
        print("TASK 2: EVASION CLASSIFICATION (Single Label)")
        print("=" * 60)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=label_names,
            zero_division=0
        ))
    
    return metrics


def evaluate_task2_multi_annotator(
    y_pred: List[str],
    gold_sets: List[set],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate Task 2 on test set with multiple annotators.
    
    A prediction is correct if it matches ANY of the annotator labels.
    
    Args:
        y_pred: Predicted labels (strings)
        gold_sets: List of sets, each containing valid annotator labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with accuracy metric
    """
    if len(y_pred) != len(gold_sets):
        raise ValueError("y_pred and gold_sets must have same length")
    
    correct = sum(1 for pred, gold_set in zip(y_pred, gold_sets) if pred in gold_set)
    accuracy = correct / len(y_pred)
    
    metrics = {'accuracy_any_annotator': accuracy}
    
    if verbose:
        print("=" * 60)
        print("TASK 2: EVASION CLASSIFICATION (Multi-Annotator)")
        print("=" * 60)
        print(f"Total examples:    {len(y_pred)}")
        print(f"Correct:           {correct}")
        print(f"Accuracy:          {accuracy:.4f}")
        print("\nNote: Prediction counted as correct if it matches ANY annotator.")
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    title: str = "Confusion Matrix",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> pd.DataFrame:
    """
    Compute per-class precision, recall, F1.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: List of label names
        
    Returns:
        DataFrame with per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, 
        average=None,
        zero_division=0
    )
    
    df = pd.DataFrame({
        'Label': label_names,
        'Support': support,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    return df


def majority_baseline_accuracy(y_train: np.ndarray, y_test: np.ndarray) -> float:
    """
    Calculate accuracy of majority class baseline.
    
    Args:
        y_train: Training labels (to find majority class)
        y_test: Test labels (to evaluate)
        
    Returns:
        Accuracy of always predicting majority class
    """
    from collections import Counter
    majority_class = Counter(y_train).most_common(1)[0][0]
    y_pred = np.full_like(y_test, majority_class)
    return accuracy_score(y_test, y_pred)


def convert_evasion_preds_to_clarity(
    evasion_preds: np.ndarray,
    evasion_label_names: List[str]
) -> np.ndarray:
    """
    Convert evasion predictions (IDs) to clarity predictions (IDs).
    
    This is for the two-step "evasion-based clarity" classification strategy.
    
    Args:
        evasion_preds: Array of predicted evasion label IDs
        evasion_label_names: List mapping evasion IDs to label names
        
    Returns:
        Array of corresponding clarity label IDs
    """
    
    clarity_preds = []
    for evasion_id in evasion_preds:
        evasion_label = evasion_label_names[evasion_id]
        clarity_label = map_evasion_to_clarity(evasion_label)
        clarity_id = CLARITY_TO_ID[clarity_label]
        clarity_preds.append(clarity_id)
    
    return np.array(clarity_preds)


def evaluate_evasion_based_clarity(
    evasion_preds: np.ndarray,
    clarity_true: np.ndarray,
    evasion_label_names: List[str],
    clarity_label_names: List[str],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Evaluate clarity classification using evasion predictions.
    
    Two-step strategy: predict evasion → map to clarity → evaluate.
    
    Args:
        evasion_preds: Predicted evasion label IDs
        clarity_true: True clarity label IDs
        evasion_label_names: List of evasion label names
        clarity_label_names: List of clarity label names
        verbose: Whether to print results
        
    Returns:
        Dictionary of metrics
    """
    # Map evasion predictions to clarity
    clarity_preds = convert_evasion_preds_to_clarity(evasion_preds, evasion_label_names)
    
    # Evaluate clarity
    metrics = {
        'accuracy': accuracy_score(clarity_true, clarity_preds),
        'macro_f1': f1_score(clarity_true, clarity_preds, average='macro'),
        'weighted_f1': f1_score(clarity_true, clarity_preds, average='weighted'),
        'macro_precision': precision_score(clarity_true, clarity_preds, average='macro'),
        'macro_recall': recall_score(clarity_true, clarity_preds, average='macro')
    }
    
    if verbose:
        print("=" * 60)
        print("EVASION-BASED CLARITY CLASSIFICATION")
        print("=" * 60)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Macro F1:          {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
        print(f"Macro Precision:   {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:      {metrics['macro_recall']:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            clarity_true, clarity_preds,
            target_names=clarity_label_names,
            zero_division=0
        ))
    
    return metrics
