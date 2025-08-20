from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
) -> Tuple[float, float]:
    """Compute McNemar's test between two classifiers on the same examples.

    Returns (statistic, p_value).
    """
    assert np.array_equal(y_true_a, y_true_b), "Ground truth labels must match"
    y_true = y_true_a
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    b01 = int(np.sum(a_correct & (~b_correct)))
    b10 = int(np.sum((~a_correct) & b_correct))
    table = [[0, b01], [b10, 0]]
    res = mcnemar(table, exact=False, correction=True)
    return float(res.statistic), float(res.pvalue)


def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Compute bootstrap confidence interval for a metric function of (y_true, y_prob).

    Robust to undefined metrics on some bootstrap samples (e.g., ROC AUC when a
    sample contains only one class). Such samples are skipped until we collect
    up to ``n_boot`` valid values or we hit a maximum number of attempts.

    Returns (point_estimate, (low, high)). If too few valid samples are
    available, falls back to a degenerate interval (point, point).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    indices = np.arange(n)

    values = []
    max_attempts = max(n_boot * 10, n_boot + 10)
    attempts = 0
    while len(values) < n_boot and attempts < max_attempts:
        attempts += 1
        sample_idx = rng.choice(indices, size=n, replace=True)
        # Skip samples with a single class to avoid undefined metrics (e.g., AUC)
        if len(np.unique(y_true[sample_idx])) < 2:
            continue
        try:
            v = metric_fn(y_true[sample_idx], y_prob[sample_idx])
        except Exception:
            continue
        # Skip non-finite results
        if not np.isfinite(v):
            continue
        values.append(float(v))

    values = np.array(values, dtype=float)
    # Point estimate on full data (may still be undefined for some metrics)
    try:
        point = float(metric_fn(y_true, y_prob))
    except Exception:
        point = float("nan")

    if values.size >= 2:
        low = float(np.percentile(values, 100 * (alpha / 2)))
        high = float(np.percentile(values, 100 * (1 - alpha / 2)))
    elif values.size == 1:
        low = high = float(values[0])
    else:
        # If we couldn't obtain any valid bootstrap value, fall back to point
        low = high = point
    return point, (low, high)
