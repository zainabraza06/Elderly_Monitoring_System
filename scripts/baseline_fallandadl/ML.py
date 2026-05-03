"""Classical ML baselines: enhanced hand-crafted features, scaling, resampling, CV."""

from __future__ import annotations

import os
import pickle
import time
import warnings
from typing import Any, Literal

import numpy as np
import psutil
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .enhanced_features import extract_enhanced_features

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _model_size_mb(estimator: Any) -> float:
    return len(pickle.dumps(estimator)) / (1024 * 1024)


def _infer_binary_proba(model: Any, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    base = getattr(model, "estimator", None) or getattr(model, "base_estimator", None)
    if base is not None and hasattr(base, "predict_proba"):
        return base.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _build_models(random_state: int = 42) -> dict[str, Any]:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


def _train_eval_one(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    task_type: Literal["binary", "multi"],
) -> tuple[dict[str, Any], np.ndarray]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = "f1" if task_type == "binary" else "accuracy"
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    start = time.time()
    fit_model = clone(model)
    if task_type == "binary" and model_name == "Logistic Regression":
        fit_model = CalibratedClassifierCV(fit_model, cv=3, method="sigmoid")
    fit_model.fit(X_train, y_train)
    train_time = time.time() - start

    mem_after = process.memory_info().rss / (1024 * 1024)
    memory_delta = max(0.0, mem_after - mem_before)

    y_pred = fit_model.predict(X_test)

    inference_times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        fit_model.predict(X_test)
        inference_times.append((time.perf_counter() - t0) * 1000 / max(len(X_test), 1))
    inference_ms = float(np.mean(inference_times))

    size_mb = _model_size_mb(fit_model)

    if task_type == "binary":
        try:
            auc = float(roc_auc_score(y_test, _infer_binary_proba(fit_model, X_test)))
        except ValueError:
            auc = float("nan")

        metrics = {
            "Model": model_name,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "F1": float(f1_score(y_test, y_pred, zero_division=0)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "AUC_ROC": auc,
            "CV_Mean": float(cv_scores.mean()),
            "CV_Std": float(cv_scores.std()),
            "Train_Time_s": train_time,
            "Inference_Time_ms": inference_ms,
            "Model_Size_MB": size_mb,
            "Memory_Usage_MB": memory_delta,
            "Time_Complexity": "O(n_estimators * n_features * log(n))",
            "Space_Complexity": f"O({size_mb:.2f} MB serialized)",
            "y_pred": y_pred,
            "y_true": y_test,
        }
    else:
        metrics = {
            "Model": model_name,
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "CV_Mean": float(cv_scores.mean()),
            "CV_Std": float(cv_scores.std()),
            "Train_Time_s": train_time,
            "Inference_Time_ms": inference_ms,
            "Model_Size_MB": size_mb,
            "Memory_Usage_MB": memory_delta,
            "Time_Complexity": "O(n_estimators * n_features * log(n))",
            "Space_Complexity": f"O({size_mb:.2f} MB serialized)",
            "y_pred": y_pred,
            "y_true": y_test,
        }

    return metrics, y_pred


def _balance_fall(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        smt = SMOTETomek(random_state=42, sampling_strategy=0.5)
        return smt.fit_resample(X, y)
    except Exception:
        smote = SMOTE(random_state=42, sampling_strategy=0.5)
        return smote.fit_resample(X, y)


def _balance_adl(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        ada = ADASYN(random_state=42, sampling_strategy="auto")
        return ada.fit_resample(X, y)
    except Exception:
        smote = SMOTE(random_state=42, sampling_strategy="auto")
        return smote.fit_resample(X, y)


def run_ml_experiments(
    X_train_raw: np.ndarray,
    y_fall_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_fall_test: np.ndarray,
    X_train_adl: np.ndarray,
    y_train_adl: np.ndarray,
    X_test_adl: np.ndarray,
    y_test_adl: np.ndarray,
    X_gyro_train: np.ndarray | None = None,
    X_gyro_test: np.ndarray | None = None,
    X_ori_train: np.ndarray | None = None,
    X_ori_test: np.ndarray | None = None,
    X_gyro_train_adl: np.ndarray | None = None,
    X_gyro_test_adl: np.ndarray | None = None,
    X_ori_train_adl: np.ndarray | None = None,
    X_ori_test_adl: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Train Task 1 (fall vs non-fall) and Task 2 (ADL multiclass) on engineered features.
    Raw inputs are windows (N, T, C); optional gyro/ori must align with acc windows.
    """
    print("\n" + "=" * 70)
    print("ENHANCED FEATURE EXTRACTION (multi-sensor fusion where available)")
    print("=" * 70)

    g_tr = X_gyro_train if X_gyro_train is not None else np.zeros_like(X_train_raw)
    g_te = X_gyro_test if X_gyro_test is not None else np.zeros_like(X_test_raw)
    o_tr = X_ori_train if X_ori_train is not None else np.zeros_like(X_train_raw)
    o_te = X_ori_test if X_ori_test is not None else np.zeros_like(X_test_raw)

    Xf_train = extract_enhanced_features(X_train_raw, g_tr, o_tr)
    Xf_test = extract_enhanced_features(X_test_raw, g_te, o_te)

    ga_tr = X_gyro_train_adl if X_gyro_train_adl is not None else np.zeros_like(X_train_adl)
    ga_te = X_gyro_test_adl if X_gyro_test_adl is not None else np.zeros_like(X_test_adl)
    oa_tr = X_ori_train_adl if X_ori_train_adl is not None else np.zeros_like(X_train_adl)
    oa_te = X_ori_test_adl if X_ori_test_adl is not None else np.zeros_like(X_test_adl)

    Xa_train = extract_enhanced_features(X_train_adl, ga_tr, oa_tr)
    Xa_test = extract_enhanced_features(X_test_adl, ga_te, oa_te)

    print("\n" + "=" * 70)
    print("SCALING AND BALANCING")
    print("=" * 70)

    scaler_fall = RobustScaler()
    X_train_fall_s = scaler_fall.fit_transform(Xf_train)
    X_test_fall_s = scaler_fall.transform(Xf_test)

    X_train_fall_bal, y_train_fall_bal = _balance_fall(X_train_fall_s, y_fall_train)
    print(
        f"Fall detection — after balancing: non-fall={np.sum(y_train_fall_bal == 0):,}, "
        f"fall={np.sum(y_train_fall_bal == 1):,}"
    )

    scaler_adl = RobustScaler()
    X_train_adl_s = scaler_adl.fit_transform(Xa_train)
    X_test_adl_s = scaler_adl.transform(Xa_test)

    X_train_adl_bal, y_train_adl_bal = _balance_adl(X_train_adl_s, y_train_adl)
    print(f"ADL classification — balanced train samples: {len(X_train_adl_bal):,}")

    models = _build_models()

    print("\n" + "=" * 60)
    print("TASK 1: FALL DETECTION (ML)")
    print("=" * 60)

    ml_fall_results: list[dict[str, Any]] = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics, _ = _train_eval_one(
            model,
            X_train_fall_bal,
            y_train_fall_bal,
            X_test_fall_s,
            y_fall_test,
            name,
            "binary",
        )
        ml_fall_results.append(metrics)
        print(f"   Accuracy: {metrics['Accuracy']:.4f}  F1: {metrics['F1']:.4f}  "
              f"AUC: {metrics['AUC_ROC']:.4f}  CV: {metrics['CV_Mean']:.4f} (±{metrics['CV_Std']:.4f})")

    print("\n" + "=" * 60)
    print("TASK 2: ADL CLASSIFICATION (ML)")
    print("=" * 60)

    ml_adl_results: list[dict[str, Any]] = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics, _ = _train_eval_one(
            model,
            X_train_adl_bal,
            y_train_adl_bal,
            X_test_adl_s,
            y_test_adl,
            name,
            "multi",
        )
        ml_adl_results.append(metrics)
        print(
            f"   Accuracy: {metrics['Accuracy']:.4f}  F1: {metrics['F1']:.4f}  "
            f"CV: {metrics['CV_Mean']:.4f} (±{metrics['CV_Std']:.4f})"
        )

    ml_fall_results.sort(key=lambda r: r["F1"], reverse=True)
    ml_adl_results.sort(key=lambda r: r["F1"], reverse=True)

    best_f = ml_fall_results[0]
    best_a = ml_adl_results[0]
    print("\n" + "=" * 70)
    print("BEST ML MODELS")
    print("=" * 70)
    print(f"Fall: {best_f['Model']}  Acc={best_f['Accuracy']:.4f}  F1={best_f['F1']:.4f}  AUC={best_f['AUC_ROC']:.4f}")
    print(f"ADL:  {best_a['Model']}  Acc={best_a['Accuracy']:.4f}  F1={best_a['F1']:.4f}")

    return ml_fall_results, ml_adl_results


def get_ml_results() -> tuple[list, list]:
    """Backward-compatible stub: prefer ``run_ml_experiments`` from the runner."""
    return [], []
