"""
Train fall vs non-fall (116-D RobustScaler, subject split, balanced training).

Used by `train_fall_detection_mobiact.py` and optionally by the combined baseline script.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler

from baseline_fall.config import (
    BEST_FALL_MODEL_NAME,
    CV_FOLDS,
    LEGACY_FALL_MODEL_NAME,
    RANDOM_STATE,
    SCALER_FALL_NAME,
)
from baseline_fall.enhanced_features import ENHANCED_FEATURE_DIM, extract_enhanced_features
from baseline_fall.fall_detection_models import build_models_fall_detection, build_xgb_fall_baseline
from baseline_fall.fall_detection_visualization import (
    save_binary_per_class_bars,
    save_confusion_matrix_binary,
    save_model_comparison_bars,
)
from baseline_fall.sampling import balance_fall_train
from baseline_fall.subject_split import subject_masks

PickMetric = Literal["f1", "accuracy"]


def train_fall_binary_pipeline(
    X_feat: np.ndarray,
    y_fall: np.ndarray,
    subject_ids: np.ndarray,
    *,
    models_dir: Path,
    results_dir: Path | None,
    random_state: int = RANDOM_STATE,
    pick_metric: PickMetric = "f1",
    xgb_only: bool = False,
    skip_plots: bool = False,
    skip_cv: bool = False,
) -> dict[str, Any]:
    """
    Subject-wise split → RobustScaler → balance train → train classifier(s).

    Saves `best_fall_model.pkl`, `scaler_fall.pkl`, `fall_detection_training_summary.json`.
    If `xgb_only`, also saves `fall_detection_xgboost.pkl` (legacy filename).

    If `results_dir` is set, writes CSV reports and figures under `reports/` and `figures/`.
    """
    assert X_feat.shape[1] == ENHANCED_FEATURE_DIM, X_feat.shape

    train_m, test_m = subject_masks(subject_ids, y_fall, random_state=random_state)
    X_tr, X_te = X_feat[train_m], X_feat[test_m]
    yf_tr, yf_te = y_fall[train_m], y_fall[test_m]

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_tr_bal, yf_tr_bal = balance_fall_train(X_tr_s, yf_tr, random_state=random_state)

    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=random_state)

    models: dict[str, Any]
    if xgb_only:
        name = "XGBoost (baseline)"
        est = build_xgb_fall_baseline(random_state=random_state)
        models = {name: est}
    else:
        models = build_models_fall_detection(random_state=random_state)

    results: dict[str, dict[str, float]] = {}
    cv_means: dict[str, float] = {}
    cv_stds: dict[str, float] = {}

    for name, est in models.items():
        if not skip_cv:
            cv_scores = cross_val_score(est, X_tr_bal, yf_tr_bal, cv=cv, scoring="f1", n_jobs=-1)
            cv_means[name] = float(cv_scores.mean())
            cv_stds[name] = float(cv_scores.std())
        else:
            cv_means[name] = float("nan")
            cv_stds[name] = float("nan")

        t0 = time.perf_counter()
        est.fit(X_tr_bal, yf_tr_bal)
        fit_time = time.perf_counter() - t0
        pred = est.predict(X_te_s)
        results[name] = {
            "accuracy": float(accuracy_score(yf_te, pred)),
            "f1": float(f1_score(yf_te, pred, average="binary", zero_division=0)),
            "time": float(fit_time),
        }

    def score_key(name: str) -> float:
        if pick_metric == "accuracy":
            return results[name]["accuracy"]
        return results[name]["f1"]

    best_name = max(results, key=lambda k: score_key(k))
    best_model = models[best_name]
    best_pred = best_model.predict(X_te_s)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / BEST_FALL_MODEL_NAME)
    joblib.dump(scaler, models_dir / SCALER_FALL_NAME)

    if xgb_only:
        joblib.dump(best_model, models_dir / LEGACY_FALL_MODEL_NAME)

    precision, recall, f1_vec, support = precision_recall_fscore_support(
        yf_te, best_pred, average=None, labels=[0, 1], zero_division=0
    )

    summary: dict[str, Any] = {
        "best_model": best_name,
        "pick_metric": pick_metric,
        "feature_dim": ENHANCED_FEATURE_DIM,
        "random_state": random_state,
        "xgb_only": xgb_only,
        "cv_folds": CV_FOLDS,
        "results_test": results,
        "cv_f1_mean": cv_means,
        "cv_f1_std": cv_stds,
        "best_test_accuracy": results[best_name]["accuracy"],
        "best_test_f1_binary": results[best_name]["f1"],
        "classes_binary": ["non_fall", "fall"],
        "per_class": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "f1": f1_vec.tolist(),
            "support": support.tolist(),
        },
    }
    (models_dir / "fall_detection_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Legacy metrics file (used by older notebooks / docs)
    (models_dir / "fall_metrics.json").write_text(
        json.dumps(
            {
                "feature_dim": ENHANCED_FEATURE_DIM,
                "best_model": best_name,
                "cv_f1_mean": float(cv_means.get(best_name, float("nan"))),
                "cv_f1_std": float(cv_stds.get(best_name, float("nan"))),
                "test_accuracy": results[best_name]["accuracy"],
                "test_f1": results[best_name]["f1"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Human-readable model_info.json (compatible with existing tooling)
    mi = {
        "model_name": best_name,
        "training_date": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "accuracy": results[best_name]["accuracy"],
        "f1_score": results[best_name]["f1"],
        "precision": float(precision[1]) if len(precision) > 1 else float(precision[0]),
        "recall": float(recall[1]) if len(recall) > 1 else float(recall[0]),
        "cv_mean": float(cv_means.get(best_name, float("nan"))),
        "cv_std": float(cv_stds.get(best_name, float("nan"))),
        "train_time": results[best_name]["time"],
    }
    (models_dir / "model_info.json").write_text(json.dumps(mi, indent=2), encoding="utf-8")

    print(
        f"\nFall detection — {best_name}: "
        f"CV F1 mean={cv_means.get(best_name, float('nan')):.4f} ± {cv_stds.get(best_name, float('nan')):.4f}, "
        f"test acc={results[best_name]['accuracy']:.4f}, "
        f"test F1(fall)={results[best_name]['f1']:.4f}"
    )

    if results_dir is not None:
        rep = results_dir / "reports"
        fig = results_dir / "figures"
        rep.mkdir(parents=True, exist_ok=True)
        fig.mkdir(parents=True, exist_ok=True)

        rows = [
            {
                "Model": n,
                "Accuracy (%)": results[n]["accuracy"] * 100,
                "F1 fall (%)": results[n]["f1"] * 100,
                "Train time (s)": results[n]["time"],
                "CV F1 mean": cv_means.get(n, float("nan")),
                "CV F1 std": cv_stds.get(n, float("nan")),
            }
            for n in results
        ]
        try:
            import pandas as pd

            pd.DataFrame(rows).sort_values("F1 fall (%)", ascending=False).to_csv(
                rep / "results_summary.csv", index=False
            )
            pd.DataFrame(
                {
                    "Class": ["Non-fall (0)", "Fall (1)"],
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1_vec,
                    "Support": support,
                }
            ).to_csv(rep / "per_class_metrics.csv", index=False)
            report = classification_report(
                yf_te, best_pred, labels=[0, 1], target_names=["non_fall", "fall"], output_dict=True, zero_division=0
            )
            pd.DataFrame(report).transpose().to_csv(rep / "classification_report.csv")
        except ImportError:
            pass

        if not skip_plots:
            save_confusion_matrix_binary(
                yf_te,
                best_pred,
                fig / "confusion_matrix.png",
                title=f"Fall detection — {best_name}",
            )
            save_model_comparison_bars(results, fig / "model_comparison.png")
            save_binary_per_class_bars(
                precision,
                recall,
                f1_vec,
                ["Non-fall", "Fall"],
                fig / "per_class_performance.png",
            )

    return {
        "summary": summary,
        "best_name": best_name,
        "best_model": best_model,
        "scaler": scaler,
        "y_test": yf_te,
        "y_pred": best_pred,
    }


def load_raw_windows_extract_features(raw: dict[str, Any]) -> np.ndarray:
    """116-D features from sliding-window bundle (`mobiact_dataset.load_sliding_windows`)."""
    return extract_enhanced_features(raw["X_acc"], raw["X_gyro"], raw["X_ori"])
