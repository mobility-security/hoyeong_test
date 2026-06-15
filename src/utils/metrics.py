from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


def compute_binary_metrics(y_true, y_pred, y_prob) -> dict:
    """S1용: accuracy, f1_binary, fpr, fnr, auc_roc"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float('nan')
    return {
        'accuracy': acc,
        'f1_binary': f1,
        'fpr': fpr,
        'fnr': fnr,
        'auc_roc': auc,
    }


def compute_multiclass_metrics(y_true, y_pred) -> dict:
    """S2/S3용: macro_f1, weighted_f1, per_class_recall, accuracy, confusion_matrix"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes,
                               average='macro', zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes,
                                  average='weighted', zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1
    per_class_recall = (cm.diagonal() / row_sums).tolist()
    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_recall': per_class_recall,
        'confusion_matrix': cm.tolist(),
    }


def compute_loao_metrics(y_true, y_pred, mse, tau) -> dict:
    """LOAO용: cae_anomaly_recall, unknown_rate, normal_fpr

    Args:
        y_true: ground-truth labels — attack samples of the excluded class (label > 0)
                and Normal samples (label 0). Pass the combined array.
        y_pred: pipeline final output per sample (0..5=known class, 6=Unknown).
        mse:    CAE per-sample reconstruction error.
        tau:    CAE anomaly threshold.
    """
    UNKNOWN = 6

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    mse = np.asarray(mse, dtype=np.float64)
    tau = float(tau)

    excluded_mask = y_true > 0
    normal_mask = y_true == 0

    cae_anomaly_recall = (
        float((mse[excluded_mask] > tau).mean()) if excluded_mask.any() else float('nan')
    )
    unknown_rate = (
        float(((mse[excluded_mask] > tau)
               & (y_pred[excluded_mask] == UNKNOWN)).mean())
        if excluded_mask.any() else float('nan')
    )
    normal_fpr = (
        float((mse[normal_mask] > tau).mean()) if normal_mask.any() else float('nan')
    )
    return {
        'cae_anomaly_recall': cae_anomaly_recall,
        'unknown_rate': unknown_rate,
        'normal_fpr': normal_fpr,
    }
