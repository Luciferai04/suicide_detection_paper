from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


def save_curves(y_true: np.ndarray, y_prob: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {prefix}")
    plt.legend()
    plt.savefig(out_dir / f"{prefix}_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    # PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {prefix}")
    plt.legend()
    plt.savefig(out_dir / f"{prefix}_pr.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {prefix}")
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, int(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(out_dir / f"{prefix}_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()
