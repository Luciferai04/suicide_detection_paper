from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from ..features.tfidf_features import TfidfFeatures


@dataclass
class SVMBaseline:
    """TF-IDF + SVM baseline with optional grid search.

    Uses RBF kernel SVC and combines word + char TF-IDF features.
    """

    grid_search: bool = True
    cv: int = 5

    def build(self) -> Pipeline:
        # Note: SMOTE is not applied because TF-IDF features are sparse; SMOTE requires dense arrays.
        # We instead use class_weight='balanced' in SVC to handle class imbalance.
        features = TfidfFeatures().build()
        svm = SVC(kernel="rbf", probability=True, class_weight="balanced")
        steps = [("features", features), ("clf", svm)]
        pipe = Pipeline(steps)
        return pipe

    def tune(self, X: np.ndarray, y: np.ndarray) -> GridSearchCV:
        pipe = self.build()
        param_grid: Dict[str, Any] = {
            "clf__C": [0.5, 1.0, 2.0, 4.0],
            "clf__gamma": ["scale", "auto"],
        }
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=self.cv, n_jobs=-1, verbose=1)
        gs.fit(X, y)
        return gs

    @staticmethod
    def report(y_true: np.ndarray, y_pred: np.ndarray, target_names=None) -> str:
        return classification_report(y_true, y_pred, target_names=target_names)
