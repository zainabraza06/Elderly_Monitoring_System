"""
Full 4-class fall-type pipeline (MobiAct multi-sensor) — modular rerun of Colab notebook.

Steps: load impact windows → 263-D features → StandardScaler → MI (top 150) →
train LGBM/XGB/RF/GB/Voting → pick best by **accuracy** → save artifacts for inference.

Usage (repo root, PYTHONPATH=scripts):
  py scripts/baseline_falltype/train_fall_type_mobiact.py --data-root "path/to/Annotated Data"

Writes:
  models/baseline_falltype/{best_fall_classifier.pkl, scaler.pkl, selected_features.pkl, label_encoder.pkl}
  results/baseline_falltype/{figures, reports}
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from baseline_falltype.config import FALL_TYPE_RAW_DIM, TOP_MI_FEATURES_DEFAULT, WINDOW_SAMPLES
from baseline_falltype.feature_extractors import CompleteFallFeatureExtractor
from baseline_falltype.feature_selection_mi import select_top_features_mi
from baseline_falltype.fall_type_models import build_models_colab
from baseline_falltype.fall_type_visualization import (
    save_confusion_matrix_heatmap,
    save_model_comparison_bars,
    save_per_class_bars,
    save_sensor_contribution_pie,
)
from baseline_falltype.fall_window_dataset import load_fall_windows_from_annotated_dir

RANDOM_STATE = 42
TOP_MI_DEFAULT = TOP_MI_FEATURES_DEFAULT


def _ensure_pandas():
    try:
        import pandas as pd

        return pd
    except ImportError as e:
        raise RuntimeError("Install pandas for CSV/Excel reports: pip install pandas openpyxl") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 4-class fall-type models (Colab pipeline).")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to MobiAct .../Annotated Data",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_REPO / "models" / "baseline_falltype",
        help="joblib output (inference paths)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO / "results" / "baseline_falltype",
        help="figures + CSV reports",
    )
    parser.add_argument(
        "--cache-features",
        type=Path,
        default=None,
        help="Optional joblib cache path for raw feature matrix (N, 263)",
    )
    parser.add_argument("--top-mi", type=int, default=TOP_MI_DEFAULT, help="MI-selected columns")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    pd = _ensure_pandas()

    fig_dir = args.results_dir / "figures"
    rep_dir = args.results_dir / "reports"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    print("Loading fall windows...")
    bundle = load_fall_windows_from_annotated_dir(args.data_root, show_progress=True)
    acc = bundle["acc_windows"]
    gyro = bundle["gyro_windows"]
    ori = bundle["ori_windows"]
    y_enc = bundle["y_encoded"]
    label_encoder = bundle["label_encoder"]

    print(f"Windows: {acc.shape}, labels: {np.bincount(y_enc)}")

    cache_path = args.cache_features
    if cache_path and cache_path.is_file():
        print(f"Loading cached features from {cache_path}")
        X = joblib.load(cache_path)
    else:
        ext = CompleteFallFeatureExtractor(fs=50.0)
        print("Extracting 263-D features...")
        t0 = time.perf_counter()
        X = ext.extract_batch(acc, gyro, ori, desc="263-D features")
        print(f"Done {X.shape} in {time.perf_counter() - t0:.1f}s")
        assert X.shape[1] == FALL_TYPE_RAW_DIM, X.shape
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(X, cache_path)
            print(f"Cached → {cache_path}")

    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)

    top_n = min(args.top_mi, X_scaled.shape[1])
    top_indices, mi_scores = select_top_features_mi(
        X_scaled, y_enc, top_n=top_n, random_state=args.random_state
    )
    X_sel = X_scaled[:, top_indices]
    print(f"MI selected {top_n} / {X_scaled.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel,
        y_enc,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y_enc,
    )

    models = build_models_colab(random_state=args.random_state)
    results: dict[str, dict[str, float]] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, est in models.items():
        t0 = time.perf_counter()
        est.fit(X_train, y_train)
        dt = time.perf_counter() - t0
        pred = est.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
            "time": float(dt),
        }
        predictions[name] = pred
        print(f"{name}: acc={results[name]['accuracy']:.4f} F1w={results[name]['f1']:.4f} t={dt:.2f}s")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = models[best_name]
    best_pred = predictions[best_name]
    print(f"\nBest (accuracy): {best_name}")

    precision, recall, f1_vec, support = precision_recall_fscore_support(
        y_test, best_pred, average=None, zero_division=0
    )
    cm_classes = [str(c) for c in label_encoder.classes_]

    args.models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, args.models_dir / "best_fall_classifier.pkl")
    joblib.dump(scaler_full, args.models_dir / "scaler.pkl")
    joblib.dump(top_indices, args.models_dir / "selected_features.pkl")
    joblib.dump(label_encoder, args.models_dir / "label_encoder.pkl")

    summary = {
        "best_model": best_name,
        "top_mi_features": int(top_n),
        "raw_feature_dim": FALL_TYPE_RAW_DIM,
        "window_samples": WINDOW_SAMPLES,
        "random_state": args.random_state,
        "results": results,
        "test_accuracy": results[best_name]["accuracy"],
        "classes": cm_classes,
    }
    (args.models_dir / "fall_type_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    rows = [
        {"Model": n, "Accuracy (%)": results[n]["accuracy"] * 100, "Weighted F1 (%)": results[n]["f1"] * 100}
        for n in results
    ]
    pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False).to_csv(
        rep_dir / "results_summary.csv", index=False
    )

    per_class_df = pd.DataFrame(
        {
            "Fall Type": [f"{c}" for c in cm_classes],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_vec,
            "Support": support,
        }
    )
    per_class_df.to_csv(rep_dir / "per_class_metrics.csv", index=False)

    report = classification_report(
        y_test, best_pred, target_names=cm_classes, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(rep_dir / "classification_report.csv")

    if not args.skip_plots:
        save_confusion_matrix_heatmap(
            y_test,
            best_pred,
            cm_classes,
            fig_dir / "confusion_matrix.png",
            title=f"Confusion matrix — {best_name}",
        )
        save_model_comparison_bars(results, fig_dir / "model_comparison.png")
        save_per_class_bars(precision, recall, f1_vec, cm_classes, fig_dir / "per_class_performance.png")
        save_sensor_contribution_pie(fig_dir / "sensor_contribution.png")

    print(f"\nSaved models → {args.models_dir}")
    print(f"Reports → {rep_dir}")
    print("Run from repo root: py scripts/sync_inference_manifest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
