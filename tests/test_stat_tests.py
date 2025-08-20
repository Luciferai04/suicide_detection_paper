import numpy as np

from suicide_detection.evaluation.stat_tests import bootstrap_ci, mcnemar_test


def test_mcnemar_shapes_and_values():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_pred_a = np.array([0, 1, 0, 0, 1, 0, 1, 0])
    y_pred_b = np.array([0, 0, 1, 0, 1, 0, 0, 0])
    stat, p = mcnemar_test(y_true, y_pred_a, y_true, y_pred_b)
    assert isinstance(stat, float) and isinstance(p, float)


def test_bootstrap_ci_returns_interval():
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.4])
    from sklearn.metrics import roc_auc_score

    point, (low, high) = bootstrap_ci(roc_auc_score, y_true, y_prob, n_boot=200, random_state=0)
    assert low <= point <= high
