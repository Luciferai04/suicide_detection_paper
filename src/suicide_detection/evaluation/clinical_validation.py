from __future__ import annotations

from typing import Dict, List

import numpy as np


def threshold_for_target_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = 0.90
) -> float:
    # Sweep thresholds to find the smallest threshold achieving at least target sensitivity (recall)
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    tp = 0
    fn = int((y_true == 1).sum())
    best_thr = 0.5
    if fn == 0:
        return 0.5
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
            fn -= 1
        # Threshold just below current prob
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        thr = (p_sorted[i] + (p_sorted[i + 1] if i + 1 < len(p_sorted) else 0.0)) / 2.0
        if sens >= target_sensitivity:
            best_thr = max(min(thr, 1.0), 0.0)
            break
    return float(best_thr)


def error_groups(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, List[int]]:
    y_pred = (y_prob >= threshold).astype(int)
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0].tolist()
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0].tolist()
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0].tolist()
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0].tolist()
    return {"FN": fn_idx, "FP": fp_idx, "TP": tp_idx, "TN": tn_idx}
