from collections import defaultdict
from typing import Optional
import warnings
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import roc_auc_score, average_precision_score

import wandb

from failure_prob.data.utils import Rollout
from .vis import plot_roc_curves, plot_prc_curves, plot_scores_by_splits
from .conformal.split_cp import split_conformal_binary
from .conformal.functional_predictor import (
    RegressionType,
    ModulationType,
    FunctionalPredictor
)

EVAL_TIMES = [
    "at earliest stop",
    "by earliest stop",
    "by final end",
]

# Doing failure detection, 1 means failure, 0 means success
def compute_roc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc


def compute_prc(success_scores, fail_scores):
    y_true = [1] * len(fail_scores) + [0] * len(success_scores)
    y_score = fail_scores + success_scores
    
    pre, rec, thresholds = precision_recall_curve(y_true, y_score)
    prc_auc = auc(rec, pre)

    return pre, rec, prc_auc



def get_metrics_curve(rollouts, key) -> list[np.ndarray]:
    return [r.logs[key].values for r in rollouts]


def eval_scores_roc_prc(
    rollouts_by_split_name: dict[str, list[Rollout]],
    scores_by_split_name: dict[str, list[np.ndarray]],
    method_name: str,
    time_quantiles: list[float] = None,
    plot_auc_curves: bool = False,
    plot_score_curves: bool = True,
):
    """
    Simplified evaluation for SafetyGym:
    Only log failure score curves. No ROC/PRC, no task_min_step.
    """
    to_be_logged = {}

    if plot_score_curves:
        fig, _ = plot_scores_by_splits(
            scores_by_split_name,
            rollouts_by_split_name,
            individual=True
        )
        fig.suptitle(method_name)
        to_be_logged[f"failure_scores/{method_name}_indiv"] = fig
        plt.close(fig)

        fig, _ = plot_scores_by_splits(
            scores_by_split_name,
            rollouts_by_split_name,
            individual=False
        )
        fig.suptitle(method_name)
        to_be_logged[f"failure_scores/{method_name}_agg"] = wandb.Image(fig)
        plt.close(fig)

    return to_be_logged



def eval_binary_classification(
    scores: np.ndarray | list,
    labels: np.ndarray | list,
    threshold: float,
) -> dict[str, float]:
    '''
    Compute the metrics for a binary classification task.
    Compute TPR, TNR, Accuracy, F1 Score based on the given threshold.
    Also compute the ROC AUC and PRC AUC, which are agnostic to the threshold.
    Properly handle the case where there is only one class in the labels.
    
    Args:
        scores: classifier scores, shape (n_samples,), higher score means more likely to be positive.
        labels: GT labels, shape (n_samples,), 1 means positive, 0 means negative.
        threshold: The threshold for the binary classification.
    
    Returns:
        dict: A dictionary of the computed metrics, with keys {tpr, tnr, accuracy, f1, roc_auc, prc_auc}.
    '''
    if isinstance(scores, list):
        scores = np.array(scores)
    if isinstance(labels, list):
        labels = np.array(labels)
        
    pos_freq = np.sum(labels) / len(labels)
    neg_freq = 1 - pos_freq

    # Generate binary predictions using the threshold.
    preds = (scores >= threshold).astype(int)
    
    # Calculate confusion matrix components.
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    TN = np.sum((preds == 0) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))
    
    # Compute TPR (Recall) and TNR.
    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    # Compute Accuracy.
    acc = (TP + TN) / len(labels) if len(labels) > 0 else 0.0
    bal_acc = (tpr + tnr) / 2
    weighted_acc = (tpr * neg_freq + tnr * pos_freq) # Weighted by the inverse class frequency
    
    # Compute Precision.
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Compute F1 Score.
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    
    # Compute ROC AUC and PRC AUC, handling the case of a single class.
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        roc_auc = float('nan')
        prc_auc = float('nan')
    else:
        roc_auc = roc_auc_score(labels, scores)
        prc_auc = average_precision_score(labels, scores)
    
    # Return the computed metrics.
    return {
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "acc": acc,
        "bal_acc": bal_acc,
        "f1": f1,
        "weighted-acc": weighted_acc,
        "roc_auc": roc_auc,
        "prc_auc": prc_auc,
    }

    
def eval_detection_time(
    scores: list[np.ndarray],
    labels: np.ndarray,
    threshold: float,
) -> float:
    '''
    Evaluate the earliest detection time, which is the earliest timestep that a score exceeds the threshold.
    Each time series in scores is labelled 1 or 0. A time series is classified as positive if its score at any time
    exceeds the threshold. This function returns the average detection time for the true positive time series.

    Args:
        scores: List of classifier scores (length n_samples), each a numpy array of shape (n_timesteps,).
        labels: Ground truth labels, numpy array of shape (n_samples,), where 1 indicates positive.
        threshold: The threshold above which a time point is considered a detection.

    Returns:
        A dictionary with a single key "avg_det_time" whose value is the average detection time
        (i.e., the earliest timestep index at which the score exceeds the threshold) for all
        time series that are labeled positive and for which a detection occurs.
        If no positive series are detected, returns NaN.
    '''
    detection_times = []

    # Loop over each time series and its corresponding ground truth label.
    for score, label in zip(scores, labels):
        if label == 1:
            # Find indices where score exceeds (or equals) the threshold.
            detection_indices = np.where(score >= threshold)[0]
            if detection_indices.size > 0:
                # Record the first occurrence (earliest detection time).
                detection_times.append(detection_indices[0] / len(score))
            else:
                # If not detected, record the maximum time (1.0).
                detection_times.append(1.0)

    # Calculate average detection time if there are any detections.
    if detection_times:
        avg_det_time = sum(detection_times) / len(detection_times)
    else:
        avg_det_time = float('nan')  # No detections for positive samples.

    return avg_det_time
    