"""
Custom metrics and loss functions for H-MTL
Includes EMD-based ordinal loss and Adjacent Confusion Rate (ACR)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix


def emd_loss(y_pred, y_true, num_classes=3):
    """
    Earth Mover's Distance (EMD) based ordinal loss
    Preserves ordinal relationships among severity levels
    
    Args:
        y_pred: (batch_size, num_classes) - predicted logits
        y_true: (batch_size,) - ground truth labels
        num_classes: number of ordinal classes
    
    Returns:
        loss: scalar EMD loss
    """
    # Convert to probabilities
    p = F.softmax(y_pred, dim=1)  # (batch_size, num_classes)
    
    # Convert ground truth to one-hot
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()
    
    # Compute cumulative distributions
    p_cdf = torch.cumsum(p, dim=1)
    y_cdf = torch.cumsum(y_true_onehot, dim=1)
    
    # EMD loss = L1 distance between CDFs
    emd = torch.sum(torch.abs(p_cdf - y_cdf), dim=1)
    
    return emd.mean()


def hierarchical_loss(main_logits, tension_logits, wear_logits, 
                      y_main, y_tension, y_wear, 
                      y_tension_severity, y_wear_severity,
                      lambda_task=1.0, lambda_struct=0.7, lambda_aux=0.3):
    """
    Combined loss for hierarchical multi-task learning
    
    Args:
        main_logits: (batch_size, 3) - main task predictions
        tension_logits: (batch_size, 3) - tension severity predictions
        wear_logits: (batch_size, 3) - wear severity predictions
        y_main: (batch_size,) - ground truth main class
        y_tension: (batch_size,) - ground truth tension severity
        y_wear: (batch_size,) - ground truth wear severity
        y_tension_severity: (batch_size,) - tension severity score (0-2)
        y_wear_severity: (batch_size,) - wear severity score (0-2)
        lambda_task, lambda_struct, lambda_aux: loss weights
    
    Returns:
        total_loss: weighted combination of all losses
    """
    device = main_logits.device
    
    # Task Loss (L_task)
    main_loss = F.cross_entropy(main_logits, y_main)
    
    tension_mask = (y_main == 1) & (y_tension >= 0)
    wear_mask = (y_main == 2) & (y_wear >= 0)
    
    tension_loss = 0.0
    wear_loss = 0.0
    
    if tension_mask.any():
        tension_loss = F.cross_entropy(tension_logits[tension_mask], y_tension[tension_mask])
    
    if wear_mask.any():
        wear_loss = F.cross_entropy(wear_logits[wear_mask], y_wear[wear_mask])
    
    task_loss = main_loss + tension_loss + wear_loss
    
    # Structural Loss (L_struct): EMD-based ordinal loss
    tension_emd_loss = 0.0
    wear_emd_loss = 0.0
    
    if tension_mask.any():
        tension_emd_loss = emd_loss(tension_logits[tension_mask], 
                                    y_tension[tension_mask], num_classes=3)
    
    if wear_mask.any():
        wear_emd_loss = emd_loss(wear_logits[wear_mask], 
                                 y_wear[wear_mask], num_classes=3)
    
    struct_loss = tension_emd_loss + wear_emd_loss
    
    # Auxiliary Loss (L_aux): Continuous severity prediction
    aux_loss = 0.0
    if tension_mask.any():
        tension_severity_normalized = y_tension_severity[tension_mask].float() / 2.0
        aux_loss += F.mse_loss(
            torch.softmax(tension_logits[tension_mask], dim=1).max(dim=1)[0],
            tension_severity_normalized
        )
    
    if wear_mask.any():
        wear_severity_normalized = y_wear_severity[wear_mask].float() / 2.0
        aux_loss += F.mse_loss(
            torch.softmax(wear_logits[wear_mask], dim=1).max(dim=1)[0],
            wear_severity_normalized
        )
    
    # Weighted combination
    total_loss = (lambda_task * task_loss + 
                  lambda_struct * struct_loss + 
                  lambda_aux * aux_loss)
    
    return total_loss


def calculate_acr(y_true, y_pred, num_main_classes=3, num_severity_levels=3):
    """
    Adjacent Confusion Rate (ACR)
    Measures misclassification while reflecting ordinal distance
    
    Args:
        y_true: ground truth labels (0-6: 0=Normal, 1-3=Tension, 4-6=Wear)
        y_pred: predicted labels
        num_main_classes: number of main fault types
        num_severity_levels: number of severity levels per fault type
    
    Returns:
        acr: ACR percentage
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(7))
    
    # Calculate weighted confusion
    total_weighted_confusion = 0.0
    total_samples = 0.0
    
    for i in range(7):
        for j in range(7):
            if i != j:  # Only misclassifications
                # Ordinal distance
                ordinal_dist = abs(i - j)
                
                # Weight for inter-task confusion
                if (i // 3) != (j // 3):  # Different main tasks
                    weight = num_severity_levels  # Higher penalty
                else:
                    weight = ordinal_dist
                
                total_weighted_confusion += weight * cm[i, j]
            
            total_samples += cm[i, j]
    
    if total_samples == 0:
        return 0.0
    
    acr = (total_weighted_confusion / total_samples) * 100
    return acr


def calculate_metrics(y_true, y_pred):
    """
    Calculate standard classification metrics
    
    Args:
        y_true: ground truth labels
        y_pred: predicted labels
    
    Returns:
        dict with accuracy, precision, recall, F1-score
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
