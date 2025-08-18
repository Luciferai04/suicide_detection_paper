from typing import Callable, Tuple
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def mcnemar_test(y_true_a: np.ndarray, y_pred_a: np.ndarray, y_true_b: np.ndarray, y_pred_b: np.ndarray) -> Tuple[float, float]:
    """Compute McNemar's test between two classifiers on the same examples.

    Returns (statistic, p_value).
    """
    assert np.array_equal(y_true_a, y_true_b), "Ground truth labels must match"
    y_true = y_true_a
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    b01 = np.sum((a_correct == True) & (b_correct == False))
    b10 = np.sum((a_correct == False) & (b_correct == True))
    table = [[0, b01], [b10, 0]]
    res = mcnemar(table, exact=False, correction=True)
    return float(res.statistic), float(res.pvalue)


def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, random_state: int = 42) -> Tuple[float, Tuple[float, float]]:
    """Compute bootstrap confidence interval for a metric function of (y_true, y_prob).

    metric_fn should be deterministic given inputs (e.g., AUC or F1 at fixed threshold).
    Returns (point_estimate, (low, high)).
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    indices = np.arange(n)
    values = []
    for _ in range(n_boot):
        sample_idx = rng.choice(indices, size=n, replace=True)
        values.append(metric_fn(y_true[sample_idx], y_prob[sample_idx]))
    values = np.array(values)
    point = metric_fn(y_true, y_prob)
    low = np.percentile(values, 100 * (alpha / 2))
    high = np.percentile(values, 100 * (1 - alpha / 2))
    return float(point), (float(low), float(high))
