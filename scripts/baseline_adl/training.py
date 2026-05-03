"""Cross-validation, metrics, best-model selection (weighted F1 like Colab)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from baseline_adl.config import CV_FOLDS, RANDOM_STATE


def train_eval_multiclass(
    name: str,
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    """5-fold CV (accuracy) + test metrics; returns metrics dict and test predictions."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    cv_scores = cross_val_score(
        estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
    )
    fit_start = time.perf_counter()
    estimator.fit(X_train, y_train)
    train_time = time.perf_counter() - fit_start
    y_pred = estimator.predict(X_test)
    total_time = time.perf_counter() - t0

    metrics: dict[str, Any] = {
        "Model": name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "CV_Mean": float(cv_scores.mean()),
        "CV_Std": float(cv_scores.std()),
        "Train_Time_s": float(train_time),
        "Total_Time_s": float(total_time),
    }
    return metrics, y_pred


def pick_best_by_f1(results: list[dict[str, Any]]) -> dict[str, Any]:
    return max(results, key=lambda r: r["F1-Score"])
