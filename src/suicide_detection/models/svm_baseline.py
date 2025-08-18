from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from ..features.tfidf_features import TfidfFeatures


@dataclass
class SVMBaseline:
    """TF-IDF + SVM baseline with optional grid search.

    Uses RBF kernel SVC and combines word + char TF-IDF features.
    """

    grid_search: bool = True
    cv: int = 5

    def build(self) -> Pipeline:
        features = TfidfFeatures().build()
        svm = SVC(kernel="rbf", probability=True)
        # SMOTE inside the pipeline ensures resampling is done only on training folds during CV
        pipe = Pipeline([("features", features), ("smote", SMOTE(random_state=42)), ("clf", svm)])
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

