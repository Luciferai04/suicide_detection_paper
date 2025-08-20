from typing import Dict

import numpy as np


def demographic_group_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Compute simple subgroup metrics if demographic groups are available.

    Safe no-op pattern: if groups is None or empty, return empty dict.
    """
    if groups is None or len(groups) == 0:
        return {}
    res: Dict[str, Dict[str, float]] = {}
    y_pred = (y_prob >= 0.5).astype(int)
    for g in np.unique(groups):
        idx = groups == g
        if idx.sum() == 0:
            continue
        yt, yp = y_true[idx], y_pred[idx]
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        res[str(g)] = {"support": int(idx.sum()), "sensitivity": sens, "specificity": spec}
    return res
