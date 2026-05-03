"""
Fall detection (binary) on MobiAct sliding windows — modular pipeline.

116-D `extract_enhanced_features` → RobustScaler → SMOTETomek/SMOTE on train →
compare LightGBM / XGBoost / RF / GB / Voting (or `--xgb-only` legacy) →
pick best by F1 (fall) or accuracy → save `best_fall_model.pkl` + `scaler_fall.pkl`.

Usage (repo root, PYTHONPATH=scripts):
  py scripts/baseline_fall/train_fall_detection_mobiact.py --data-root "path/to/MobiAct_or_Annotated"

Writes:
  models/baseline_fall/{best_fall_model.pkl, scaler_fall.pkl, fall_detection_training_summary.json, ...}
  results/baseline_fall/{figures, reports}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from baseline_fall.config import RANDOM_STATE
from baseline_fall.fall_detection_core import load_raw_windows_extract_features, train_fall_binary_pipeline
from baseline_fall.mobiact_dataset import discover_data_root, load_sliding_windows


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fall-vs-ADL binary classifier (116-D MobiAct pipeline).")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Folder containing MobiAct or direct path to Annotated Data (default: repo/data)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=_REPO / "models" / "baseline_fall",
        help="Output directory for joblib models",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO / "results" / "baseline_fall",
        help="Figures and CSV reports",
    )
    parser.add_argument(
        "--cache-features",
        type=Path,
        default=None,
        help="Optional joblib path to cache the (N, 116) feature matrix",
    )
    parser.add_argument(
        "--pick-metric",
        choices=("f1", "accuracy"),
        default="f1",
        help="Metric to select the champion model (default: F1 on fall class)",
    )
    parser.add_argument("--xgb-only", action="store_true", help="Only train legacy single XGBoost (200 trees).")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 5-fold CV (faster smoke runs).")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    ann = discover_data_root(args.data_root)
    print(f"Annotated data: {ann}")

    raw = load_sliding_windows(ann)
    print(f"Windows: {raw['X_acc'].shape[0]}")

    cache_path = args.cache_features
    if cache_path and cache_path.is_file():
        print(f"Loading cached features from {cache_path}")
        X_feat = joblib.load(cache_path)
    else:
        print("Extracting 116-D features...")
        t0 = time.perf_counter()
        X_feat = load_raw_windows_extract_features(raw)
        print(f"Features {X_feat.shape} in {time.perf_counter() - t0:.1f}s")
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(X_feat, cache_path)
            print(f"Cached → {cache_path}")

    y_fall = raw["y_fall"]
    subject_ids = raw["subject_ids"]

    train_fall_binary_pipeline(
        X_feat,
        y_fall,
        subject_ids,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        random_state=args.random_state,
        pick_metric=args.pick_metric,  # type: ignore[arg-type]
        xgb_only=args.xgb_only,
        skip_plots=args.skip_plots,
        skip_cv=args.skip_cv,
    )

    print(f"\nSaved models → {args.models_dir}")
    print(f"Reports → {args.results_dir}")
    print("Run from repo root: py scripts/sync_inference_manifest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
