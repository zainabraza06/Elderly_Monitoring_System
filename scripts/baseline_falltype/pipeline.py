"""Multi-sensor feature matrix, MI selection, training, saving."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import (
    FALL_NAMES,
    GB_PARAMS,
    LGBM_PARAMS,
    RANDOM_STATE,
    RF_PARAMS,
    TEST_SIZE,
    TOP_N_FEATURES,
    XGB_PARAMS,
)
from .feature_extractors import CompleteFallFeatureExtractor


def build_feature_matrix(
    acc_windows: np.ndarray,
    gyro_windows: np.ndarray,
    ori_windows: np.ndarray,
    cache_dir: Path,
    force_recompute: bool,
    fs_hz: float,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "multisensor_features.pkl"

    if cache_file.exists() and not force_recompute:
        return joblib.load(cache_file)

    extractor = CompleteFallFeatureExtractor(fs=fs_hz)
    X = extractor.extract_batch(acc_windows, gyro_windows, ori_windows)
    joblib.dump(X, cache_file)
    return X


def select_and_split(
    X_features: np.ndarray,
    y_encoded: np.ndarray,
    top_n: int = TOP_N_FEATURES,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray, int]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    mi_scores = mutual_info_classif(X_scaled, y_encoded, random_state=random_state)
    k = min(top_n, X_scaled.shape[1])
    top_features = np.argsort(mi_scores)[-k:]
    X_selected = X_scaled[:, top_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=random_state,
        stratify=y_encoded,
    )
    return X_train, X_test, y_train, y_test, scaler, top_features, k


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], np.ndarray]:
    results: dict[str, dict[str, Any]] = {}
    models: dict[str, Any] = {}

    lgbm = LGBMClassifier(**LGBM_PARAMS)
    t0 = time.perf_counter()
    lgbm.fit(X_train, y_train)
    results["LightGBM"] = {
        "accuracy": float(accuracy_score(y_test, lgbm.predict(X_test))),
        "f1": float(f1_score(y_test, lgbm.predict(X_test), average="weighted")),
        "time": float(time.perf_counter() - t0),
    }
    models["LightGBM"] = lgbm

    xgb = XGBClassifier(**XGB_PARAMS)
    t0 = time.perf_counter()
    xgb.fit(X_train, y_train)
    results["XGBoost"] = {
        "accuracy": float(accuracy_score(y_test, xgb.predict(X_test))),
        "f1": float(f1_score(y_test, xgb.predict(X_test), average="weighted")),
        "time": float(time.perf_counter() - t0),
    }
    models["XGBoost"] = xgb

    rf = RandomForestClassifier(**RF_PARAMS)
    t0 = time.perf_counter()
    rf.fit(X_train, y_train)
    results["Random Forest"] = {
        "accuracy": float(accuracy_score(y_test, rf.predict(X_test))),
        "f1": float(f1_score(y_test, rf.predict(X_test), average="weighted")),
        "time": float(time.perf_counter() - t0),
    }
    models["Random Forest"] = rf

    gb = GradientBoostingClassifier(**GB_PARAMS)
    t0 = time.perf_counter()
    gb.fit(X_train, y_train)
    results["Gradient Boosting"] = {
        "accuracy": float(accuracy_score(y_test, gb.predict(X_test))),
        "f1": float(f1_score(y_test, gb.predict(X_test), average="weighted")),
        "time": float(time.perf_counter() - t0),
    }
    models["Gradient Boosting"] = gb

    voting = VotingClassifier(
        estimators=[("lgbm", lgbm), ("xgb", xgb), ("rf", rf)],
        voting="soft",
    )
    t0 = time.perf_counter()
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)
    results["Voting Ensemble"] = {
        "accuracy": float(accuracy_score(y_test, voting_pred)),
        "f1": float(f1_score(y_test, voting_pred, average="weighted")),
        "time": float(time.perf_counter() - t0),
    }
    models["Voting Ensemble"] = voting

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = models[best_name]
    best_pred = best_model.predict(X_test)

    return results, models, best_pred


def save_metadata(
    path: Path,
    best_model_name: str,
    results: dict[str, dict[str, Any]],
    class_names: list[str],
    top_n: int,
    n_samples: int,
    n_features_raw: int,
    pipeline: str = "multisensor",
) -> None:
    payload = {
        "dataset": "MobiAct",
        "task": "4-class fall type classification",
        "pipeline": pipeline,
        "classes": class_names,
        "n_samples": n_samples,
        "n_features_extracted": n_features_raw,
        "top_n_features_mi": top_n,
        "best_model": best_model_name,
        "metrics": results,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_reports(
    reports_dir: Path,
    results: dict[str, dict[str, Any]],
    label_encoder: LabelEncoder,
    y_test: np.ndarray,
    best_pred: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    short_names = list(label_encoder.classes_)

    results_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Accuracy (%)": results[name]["accuracy"] * 100,
                "Weighted F1 (%)": results[name]["f1"] * 100,
                "Training Time (s)": results[name]["time"],
            }
            for name in results
        ]
    ).sort_values("Accuracy (%)", ascending=False)
    results_df.to_csv(reports_dir / "results_summary.csv", index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, best_pred, average=None, zero_division=0
    )
    per_class_df = pd.DataFrame(
        {
            "Fall Type": [f"{code} ({FALL_NAMES.get(code, code)})" for code in short_names],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Support": support,
        }
    )
    per_class_df.to_csv(reports_dir / "per_class_metrics.csv", index=False)

    report_dict = classification_report(
        y_test, best_pred, target_names=short_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report_dict).transpose().to_csv(reports_dir / "classification_report.csv")

    return results_df, per_class_df
