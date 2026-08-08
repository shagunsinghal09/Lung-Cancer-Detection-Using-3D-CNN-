from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    metrics = {"accuracy": float(accuracy), "precision": float(precision), "recall": float(recall), "f1": float(f1)}

    unique_labels = np.unique(y_true)
    if unique_labels.size < 2:
        roc_auc = 0.0
    else:
        order = np.argsort(-y_score)
        y_true_sorted = y_true[order]
        pos = np.sum(y_true_sorted == 1)
        neg = np.sum(y_true_sorted == 0)
        if pos == 0 or neg == 0:
            roc_auc = 0.0
        else:
            tpr = np.cumsum(y_true_sorted == 1) / pos
            fpr = np.cumsum(y_true_sorted == 0) / neg
            roc_auc = float(np.trapz(tpr, fpr))
    metrics["roc_auc"] = roc_auc
    metrics["confusion_matrix"] = [[tn, fp], [fn, tp]]
    return metrics
