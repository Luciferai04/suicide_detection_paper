from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
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


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> EvalResults:
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
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


def fairness_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, groups: Optional[np.ndarray], threshold: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """Compute simple fairness metrics per group.

    Returns a dict keyed by group with metrics:
      - positive_rate (Demographic Parity proxy)
      - tpr (Equal Opportunity / recall for positive class)
      - fpr (Fall-out)
      - support (number of samples)
    If groups is None or empty, returns an empty dict.
    """
    if groups is None:
        return {}
    groups = np.asarray(groups)
    if groups.size == 0:
        return {}
    y_pred = (y_prob >= threshold).astype(int)
    out: Dict[str, Dict[str, float]] = {}
    for g in np.unique(groups):
        idx = groups == g
        if not np.any(idx):
            continue
        yt, yp = y_true[idx], y_pred[idx]
        support = int(idx.sum())
        pos_rate = float(yp.mean()) if support else 0.0
        # TPR and FPR
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        out[str(g)] = {
            "support": support,
            "positive_rate": pos_rate,
            "tpr": tpr,
            "fpr": fpr,
        }
    return out
