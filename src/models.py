"""
Model architectures and custom loss functions for QEvasion tasks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Tuple, Optional


class MultiTaskTransformer(nn.Module):
    """
    Multi-task transformer for joint clarity and evasion classification.
    
    Architecture:
    - Shared transformer encoder
    - Separate classification heads for clarity (3-way) and evasion (9-way)
    
    Args:
        model_name: Pretrained transformer model name
        num_clarity_labels: Number of clarity classes (default: 3)
        num_evasion_labels: Number of evasion classes (default: 9)
        dropout_rate: Dropout rate for classification heads
    """
    
    def __init__(
        self,
        model_name: str,
        num_clarity_labels: int = 3,
        num_evasion_labels: int = 9,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Clarity classification head
        self.clarity_dropout = nn.Dropout(dropout_rate)
        self.clarity_classifier = nn.Linear(hidden_size, num_clarity_labels)
        
        # Evasion classification head
        self.evasion_dropout = nn.Dropout(dropout_rate)
        self.evasion_classifier = nn.Linear(hidden_size, num_evasion_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (clarity_logits, evasion_logits)
        """
        # Shared encoding
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Clarity prediction
        clarity_hidden = self.clarity_dropout(pooled_output)
        clarity_logits = self.clarity_classifier(clarity_hidden)
        
        # Evasion prediction
        evasion_hidden = self.evasion_dropout(pooled_output)
        evasion_logits = self.evasion_classifier(evasion_hidden)
        
        return clarity_logits, evasion_logits


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal Loss down-weights easy examples and focuses on hard examples.
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    Args:
        alpha: Weighting factor for each class [num_classes]
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', or 'none')
        
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Class labels [batch_size]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# class LabelSmoothingCrossEntropy(nn.Module):
#     """
#     Cross Entropy Loss with Label Smoothing.
    
#     Label smoothing prevents overconfident predictions by distributing
#     a small amount of probability mass to incorrect classes.
    
#     Args:
#         smoothing: Smoothing factor (default: 0.1)
#         reduction: Reduction method ('mean', 'sum', or 'none')
        
#     Reference:
#         Szegedy et al. "Rethinking the Inception Architecture" (2016)
#     """
    
#     def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
#         super().__init__()
#         self.smoothing = smoothing
#         self.reduction = reduction
    
#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             inputs: Logits [batch_size, num_classes]
#             targets: Class labels [batch_size]
            
#         Returns:
#             Label-smoothed cross entropy loss
#         """
#         num_classes = inputs.size(-1)
#         log_probs = F.log_softmax(inputs, dim=-1)
        
#         # Create smoothed target distribution
#         targets_one_hot = torch.zeros_like(log_probs).scatter_(
#             1, targets.unsqueeze(1), 1
#         )
#         targets_smooth = targets_one_hot * (1 - self.smoothing) + \
#                         self.smoothing / num_classes
        
#         loss = (-targets_smooth * log_probs).sum(dim=-1)
        
#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss


# def create_class_weights(
#     labels: torch.Tensor,
#     num_classes: int,
#     device: torch.device
# ) -> torch.Tensor:
#     """
#     Compute inverse frequency class weights.
    
#     Args:
#         labels: Training labels
#         num_classes: Number of classes
#         device: Device to place weights on
        
#     Returns:
#         Class weights tensor [num_classes]
#     """
#     class_counts = torch.bincount(labels, minlength=num_classes)
#     class_weights = 1.0 / (class_counts.float() + 1e-6)
#     class_weights = class_weights / class_weights.sum() * num_classes
#     return class_weights.to(device)
