"""
Training and evaluation utilities for QEvasion models.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score


class EarlyStopping:
    """
    Early stopping to halt training when validation metric stops improving.
    
    Args:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        mode: 'max' for metrics to maximize (accuracy, F1), 'min' for loss
        verbose: Whether to print messages
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, epoch: int, value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            epoch: Current epoch number
            value: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            return False
        
        if self._is_improvement(value):
            if self.verbose:
                print(f"Metric improved from {self.best_value:.4f} to {value:.4f}")
            self.best_value = value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter} epoch(s)")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        
        return False
    
    def _is_improvement(self, value: float) -> bool:
        """Check if value represents an improvement."""
        if self.mode == 'max':
            return value > self.best_value + self.min_delta
        else:
            return value < self.best_value - self.min_delta
    
    def get_best(self, values: list, mode: Optional[str] = None) -> Tuple[float, int]:
        """
        Get best value and its index from a list.
        
        Args:
            values: List of metric values
            mode: Override mode ('max' or 'min'), uses self.mode if None
            
        Returns:
            Tuple of (best_value, best_index)
        """
        mode = mode or self.mode
        if mode == 'max':
            best_idx = np.argmax(values)
        else:
            best_idx = np.argmin(values)
        return values[best_idx], best_idx


def train_epoch_improved(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    clarity_loss_fn: nn.Module,
    evasion_loss_fn: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> Dict[str, float]:
    """
    Train for one epoch with gradient clipping and advanced losses.
    
    Args:
        model: Model to train
        loader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        clarity_loss_fn: Loss function for clarity task
        evasion_loss_fn: Loss function for evasion task
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    total_loss = 0
    total_clarity_loss = 0
    total_evasion_loss = 0
    num_batches = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        clarity_labels = batch.get("clarity_labels", batch["labels"]).to(device)
        
        # Forward pass
        if "evasion_labels" in batch:
            # Multi-task model
            clarity_logits, evasion_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            loss_clarity = clarity_loss_fn(clarity_logits, clarity_labels)
            
            evasion_labels = batch["evasion_labels"].to(device)
            evasion_mask = batch["evasion_mask"].to(device).bool()
            
            if evasion_mask.any():
                loss_evasion = evasion_loss_fn(
                    evasion_logits[evasion_mask],
                    evasion_labels[evasion_mask]
                )
                loss = loss_clarity + loss_evasion
                total_evasion_loss += loss_evasion.item()
            else:
                loss = loss_clarity
                
            total_clarity_loss += loss_clarity.item()
        else:
            # Single-task model
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=clarity_labels
            )
            loss = outputs.loss
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    metrics = {'loss': total_loss / num_batches}
    if total_clarity_loss > 0:
        metrics['clarity_loss'] = total_clarity_loss / num_batches
    if total_evasion_loss > 0:
        metrics['evasion_loss'] = total_evasion_loss / num_batches
    
    return metrics


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_predictions: bool = False
) -> Tuple[Dict[str, float], ...]:
    """
    Comprehensive evaluation with optional prediction return.
    
    Args:
        model: Model to evaluate
        loader: Evaluation data loader
        device: Device to evaluate on
        return_predictions: Whether to return predictions and logits
        
    Returns:
        If return_predictions=False: Dictionary of metrics
        If return_predictions=True: (metrics, predictions, labels, logits)
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("clarity_labels", batch["labels"]).to(device)
            
            # Handle both single-task and multi-task models
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]  # clarity logits for multi-task
            else:
                logits = outputs.logits
            
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if return_predictions:
                all_logits.extend(logits.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }
    
    if return_predictions:
        return metrics, all_preds, all_labels, np.array(all_logits)
    return metrics


def evaluate_multitask(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate multi-task model on both tasks.
    
    Args:
        model: Multi-task model to evaluate
        loader: Evaluation data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary with metrics for both tasks
    """
    model.eval()
    
    all_cl_preds = []
    all_cl_labels = []
    all_ev_preds = []
    all_ev_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            clarity_labels = batch["clarity_labels"].to(device)
            evasion_labels = batch["evasion_labels"].to(device)
            evasion_mask = batch["evasion_mask"].to(device).bool()
            
            clarity_logits, evasion_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Clarity predictions
            cl_preds = torch.argmax(clarity_logits, dim=-1)
            all_cl_preds.extend(cl_preds.cpu().numpy())
            all_cl_labels.extend(clarity_labels.cpu().numpy())
            
            # Evasion predictions (only where labels exist)
            if evasion_mask.any():
                ev_preds = torch.argmax(evasion_logits[evasion_mask], dim=-1)
                all_ev_preds.extend(ev_preds.cpu().numpy())
                all_ev_labels.extend(evasion_labels[evasion_mask].cpu().numpy())
    
    # Clarity metrics
    cl_acc = accuracy_score(all_cl_labels, all_cl_preds)
    cl_macro_f1 = f1_score(all_cl_labels, all_cl_preds, average='macro')
    cl_weighted_f1 = f1_score(all_cl_labels, all_cl_preds, average='weighted')
    
    metrics = {
        'clarity_accuracy': cl_acc,
        'clarity_macro_f1': cl_macro_f1,
        'clarity_weighted_f1': cl_weighted_f1
    }
    
    # Evasion metrics (if we have labels)
    if len(all_ev_labels) > 0:
        ev_acc = accuracy_score(all_ev_labels, all_ev_preds)
        ev_macro_f1 = f1_score(all_ev_labels, all_ev_preds, average='macro')
        ev_weighted_f1 = f1_score(all_ev_labels, all_ev_preds, average='weighted')
        
        metrics.update({
            'evasion_accuracy': ev_acc,
            'evasion_macro_f1': ev_macro_f1,
            'evasion_weighted_f1': ev_weighted_f1
        })
    
    return metrics


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    clarity_loss_fn: nn.Module,
    evasion_loss_fn: nn.Module,
    device: torch.device,
    num_epochs: int,
    early_stopping: Optional[EarlyStopping] = None,
    verbose: bool = True,
    is_multitask: bool = False
) -> Dict[str, list]:
    """
    Complete training loop with early stopping.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        clarity_loss_fn: Loss function for clarity task
        evasion_loss_fn: Loss function for evasion task
        device: Device to train on
        num_epochs: Maximum number of epochs
        early_stopping: EarlyStopping instance (optional)
        verbose: Whether to print progress
        is_multitask: Whether model is multi-task
        
    Returns:
        Dictionary of training history
    """
    history = {
        'train_loss': [],
        'val_accuracy': [],
        'val_macro_f1': []
    }
    
    for epoch in range(num_epochs):
        if verbose:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
        
        # Training
        train_metrics = train_epoch_improved(
            model, train_loader, optimizer, scheduler,
            clarity_loss_fn, evasion_loss_fn, device
        )
        
        # Validation
        if is_multitask:
            val_metrics = evaluate_multitask(model, val_loader, device)
            val_acc = val_metrics['clarity_accuracy']
            val_f1 = val_metrics['clarity_macro_f1']
        else:
            val_metrics = evaluate(model, val_loader, device)
            val_acc = val_metrics['accuracy']
            val_f1 = val_metrics['macro_f1']
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['val_accuracy'].append(val_acc)
        history['val_macro_f1'].append(val_f1)
        
        if verbose:
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_acc:.4f}")
            print(f"Val Macro F1: {val_f1:.4f}")
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(epoch, val_f1):
                if verbose:
                    print(f"\nStopping early at epoch {epoch + 1}")
                break
    
    return history
