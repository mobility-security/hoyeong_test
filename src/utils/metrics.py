from typing import List
import numpy as np


def compute_binary_metrics(y_true, y_pred, y_prob) -> dict:
    """S1용: accuracy, f1_binary, fpr, fnr, auc_roc"""
    ...


def compute_multiclass_metrics(y_true, y_pred) -> dict:
    """S2/S3용: macro_f1, weighted_f1, per_class_recall, accuracy, confusion_matrix"""
    ...


def compute_loao_metrics(y_true, y_pred, mse, tau) -> dict:
    """LOAO용: cae_anomaly_recall, unknown_rate, normal_fpr"""
    ...
