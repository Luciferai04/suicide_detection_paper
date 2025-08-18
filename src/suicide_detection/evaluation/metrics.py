from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class EvalResults:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion: List[List[int]]
    cost_weighted_accuracy: Optional[float]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5, fn_cost: float = 5.0, fp_cost: float = 1.0) -> EvalResults:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = None
    try:
        pr = average_precision_score(y_true, y_prob)
    except Exception:
        pr = None
    cm = confusion_matrix(y_true, y_pred).tolist()

    # Cost-weighted accuracy: penalize FN more heavily
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tn + fp + fn + tp
    # Reward correct decisions as 1, penalize FP/FN by costs
    cost = tp + tn - (fp_cost * fp + fn_cost * fn)
    cwa = cost / total if total else None

    return EvalResults(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc,
        pr_auc=pr,
        confusion=cm,
        cost_weighted_accuracy=cwa,
    )

