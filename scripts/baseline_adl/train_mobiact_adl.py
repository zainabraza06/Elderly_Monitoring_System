"""
Train ADL classifiers on MobiAct (116-D enhanced fusion), evaluate, plot, save best model.

Reproducibility: fixed seeds in config, stratified splits, StratifiedKFold CV.

Usage (from repo root):
  set PYTHONPATH=scripts  (Windows: set PYTHONPATH=scripts)
  py scripts/baseline_adl/train_mobiact_adl.py --data-root path/to/MobiAct_Dataset_v2.0/Annotated Data

Requires: xgboost, lightgbm, scikit-learn, numpy, imbalanced-learn (optional balance).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from baseline_adl.config import (
    DEFAULT_BALANCE_ADL,
    FEATURE_DIM,
    RANDOM_STATE,
    RESULTS_SUBDIR,
)
from baseline_adl.data_prep import prepare_adl_dataset, stratified_train_test
from baseline_adl.models import build_model_candidates
from baseline_adl.training import pick_best_by_f1, train_eval_multiclass
from baseline_adl.visualization import save_confusion_matrix_png, save_model_comparison_bar
from baseline_fall.enhanced_features import ENHANCED_FEATURE_DIM, extract_enhanced_features
from baseline_fall.mobiact_dataset import discover_data_root, load_sliding_windows


def _maybe_balance_adl(
    X: np.ndarray, y: np.ndarray, enable: bool
) -> tuple[np.ndarray, np.ndarray]:
    if not enable:
        return X, y
    try:
        from imblearn.over_sampling import ADASYN, SMOTE

        try:
            return ADASYN(random_state=RANDOM_STATE, sampling_strategy="auto").fit_resample(X, y)
        except Exception:
            return SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto").fit_resample(X, y)
    except ImportError as e:
        raise RuntimeError("Install imbalanced-learn for --balance-adl: pip install imbalanced-learn") from e


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ADL multiclass (116-D) — Colab Task 2 style.")
    parser.add_argument("--data-root", type=Path, default=None, help="MobiAct Annotated Data folder")
    parser.add_argument("--models-dir", type=Path, default=_REPO / "models", help="Output models root")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO / "results" / RESULTS_SUBDIR,
        help="Figures + summary CSV",
    )
    parser.add_argument(
        "--balance-adl",
        action="store_true",
        help="Apply ADASYN/SMOTE on scaled training set (Colab baseline often trains without this).",
    )
    parser.add_argument(
        "--no-balance-adl",
        action="store_true",
        help="Force no oversampling (default matches recent Colab: train on scaled counts).",
    )
    args = parser.parse_args()
    balance = DEFAULT_BALANCE_ADL
    if args.balance_adl:
        balance = True
    if args.no_balance_adl:
        balance = False

    ann = discover_data_root(args.data_root)
    print(f"Annotated data: {ann}")

    raw = load_sliding_windows(ann)
    print(f"Windows: {raw['X_acc'].shape[0]}")

    print("Extracting 116-D features...")
    t0 = time.perf_counter()
    X_feat = extract_enhanced_features(raw["X_acc"], raw["X_gyro"], raw["X_ori"])
    print(f"Features {X_feat.shape} in {time.perf_counter() - t0:.1f}s")
    assert X_feat.shape[1] == ENHANCED_FEATURE_DIM == FEATURE_DIM

    X_adl, y_adl, label_encoder = prepare_adl_dataset(
        X_feat, raw["y_fall"], raw["y_adl"], min_samples=100
    )
    print(f"ADL samples: {len(X_adl)}, classes: {len(label_encoder.classes_)}")

    X_tr, X_te, y_tr, y_te = stratified_train_test(X_adl, y_adl)

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    X_tr_fit, y_tr_fit = _maybe_balance_adl(X_tr_s, y_tr, balance)
    print(
        f"Train shape after scaling: {X_tr_s.shape}; "
        f"fit set {'(balanced) ' if balance else ''}{X_tr_fit.shape}"
    )

    candidates = build_model_candidates()
    results: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    for name, est in candidates.items():
        print(f"\n--- {name} ---")
        m, pred = train_eval_multiclass(
            name, clone(est), X_tr_fit, y_tr_fit, X_te_s, y_te
        )
        results.append(m)
        predictions[name] = pred
        print(
            f"  acc={m['Accuracy']:.4f} F1w={m['F1-Score']:.4f} "
            f"CVacc={m['CV_Mean']:.4f}±{m['CV_Std']:.4f} time={m['Train_Time_s']:.2f}s"
        )

    best_row = pick_best_by_f1(results)
    best_name = str(best_row["Model"])
    print(f"\nBest (weighted F1): {best_name}")

    best_est = clone(candidates[best_name])
    best_est.fit(X_tr_fit, y_tr_fit)
    best_pred = predictions[best_name]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_model_comparison_bar(results, args.results_dir / "adl_model_comparison.png")
    class_names = [str(c) for c in label_encoder.classes_]
    save_confusion_matrix_png(
        y_te,
        best_pred,
        class_names,
        args.results_dir / f"confusion_matrix_{best_name.replace(' ', '_')}.png",
        title=f"ADL — {best_name}",
    )

    summary_csv = args.results_dir / "results_summary.csv"
    import csv

    results_sorted = sorted(results, key=lambda r: r["F1-Score"], reverse=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader()
        for row in results_sorted:
            w.writerow(row)
    print(f"Saved figures + {summary_csv}")

    report_txt = classification_report(
        y_te, best_pred, target_names=class_names, zero_division=0
    )
    (args.results_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    adl_dir = args.models_dir / "baseline_adl"
    adl_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_est, adl_dir / "best_adl_model.pkl")
    joblib.dump(scaler, adl_dir / "scaler_adl.pkl")
    joblib.dump(label_encoder, adl_dir / "adl_label_encoder.pkl")

    slug = best_name.lower().replace(" ", "_")
    joblib.dump(best_est, adl_dir / f"adl_classification_{slug}.pkl")

    summary = {
        "feature_dim": FEATURE_DIM,
        "random_state": RANDOM_STATE,
        "balanced_training": balance,
        "best_model": best_name,
        "n_classes": int(len(class_names)),
        "classes": class_names,
        "test_accuracy": best_row["Accuracy"],
        "test_f1_weighted": best_row["F1-Score"],
        "cv_accuracy_mean": best_row["CV_Mean"],
        "cv_accuracy_std": best_row["CV_Std"],
        "all_models": results,
    }
    (adl_dir / "adl_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved → {adl_dir / 'best_adl_model.pkl'} (canonical for inference; manifest baseline_adl/model_path)")
    print("Update manifest path if needed, then: py scripts/sync_inference_manifest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
