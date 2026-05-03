"""
Train Task 1 (fall vs ADL) and Task 2 (ADL multiclass) with 116-D enhanced features.

Fall detection is delegated to `fall_detection_core.train_fall_binary_pipeline` (same logic as
`train_fall_detection_mobiact.py` with `--xgb-only`).

For multi-model fall detection + plots, run:
  scripts/baseline_fall/train_fall_detection_mobiact.py

For Colab-style multi-model ADL (LightGBM, comparison plots, `best_adl_model.pkl`), use:
  scripts/baseline_adl/train_mobiact_adl.py

This combined script writes XGBoost-only fall artifacts plus ADL XGBoost as before.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from baseline_fall.enhanced_features import ENHANCED_FEATURE_DIM, extract_enhanced_features
from baseline_fall.fall_detection_core import train_fall_binary_pipeline
from baseline_fall.mobiact_dataset import discover_data_root, load_sliding_windows
from baseline_fall.sampling import balance_adl_train

RANDOM_STATE = 42


def main() -> int:
    parser = argparse.ArgumentParser(description="Train fall + ADL XGBoost (116-D enhanced features).")
    parser.add_argument("--data-root", type=Path, default=None, help="Folder containing MobiAct (default: repo/data)")
    parser.add_argument("--models-dir", type=Path, default=_REPO / "models", help="Output models root")
    args = parser.parse_args()

    ann = discover_data_root(args.data_root)
    print(f"Annotated data: {ann}")

    raw = load_sliding_windows(ann)
    print(f"Windows: {raw['X_acc'].shape[0]}")

    print("Extracting 116-D features...")
    t0 = time.perf_counter()
    X_feat = extract_enhanced_features(raw["X_acc"], raw["X_gyro"], raw["X_ori"])
    print(f"Features {X_feat.shape} in {time.perf_counter() - t0:.1f}s")
    assert X_feat.shape[1] == ENHANCED_FEATURE_DIM

    y_fall = raw["y_fall"]
    y_adl = raw["y_adl"]
    subject_ids = raw["subject_ids"]

    fall_dir = args.models_dir / "baseline_fall"
    train_fall_binary_pipeline(
        X_feat,
        y_fall,
        subject_ids,
        models_dir=fall_dir,
        results_dir=None,
        random_state=RANDOM_STATE,
        pick_metric="f1",
        xgb_only=True,
        skip_plots=True,
        skip_cv=False,
    )
    print(f"Saved fall model → {fall_dir}")

    # --- ADL (non-fall only) ---
    non_fall = y_fall == 0
    X_nf = X_feat[non_fall]
    y_nf = y_adl[non_fall]

    uniq, cnt = np.unique(y_nf, return_counts=True)
    valid = uniq[cnt >= 100]
    keep = np.isin(y_nf, valid)
    X_nf, y_nf = X_nf[keep], y_nf[keep]

    le = LabelEncoder()
    y_clean = le.fit_transform(y_nf)

    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(
        X_nf, y_clean, test_size=0.2, random_state=RANDOM_STATE, stratify=y_clean
    )

    scaler_adl = RobustScaler()
    Xa_tr_s = scaler_adl.fit_transform(Xa_tr)
    Xa_te_s = scaler_adl.transform(Xa_te)

    Xa_bal, ya_bal = balance_adl_train(Xa_tr_s, ya_tr, random_state=RANDOM_STATE)

    xgb_adl = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric="mlogloss",
    )

    print("CV (ADL, accuracy)...")
    cv_a = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_a_scores = cross_val_score(xgb_adl, Xa_bal, ya_bal, cv=cv_a, scoring="accuracy", n_jobs=-1)
    print(f"  CV acc: {cv_a_scores.mean():.4f} ± {cv_a_scores.std():.4f}")

    t0 = time.perf_counter()
    xgb_adl.fit(Xa_bal, ya_bal)
    pred_a = xgb_adl.predict(Xa_te_s)
    print(
        f"ADL test — acc={accuracy_score(ya_te, pred_a):.4f}, "
        f"F1w={f1_score(ya_te, pred_a, average='weighted'):.4f}, time={time.perf_counter()-t0:.1f}s"
    )
    print(classification_report(ya_te, pred_a, zero_division=0))

    adl_dir = args.models_dir / "baseline_adl"
    adl_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_adl, adl_dir / "adl_classification_xgboost.pkl")
    joblib.dump(scaler_adl, adl_dir / "scaler_adl.pkl")
    joblib.dump(le, adl_dir / "adl_label_encoder.pkl")
    (adl_dir / "adl_metrics.json").write_text(
        json.dumps(
            {
                "feature_dim": ENHANCED_FEATURE_DIM,
                "n_classes": int(len(le.classes_)),
                "classes": [str(c) for c in le.classes_],
                "cv_accuracy_mean": float(cv_a_scores.mean()),
                "test_accuracy": float(accuracy_score(ya_te, pred_a)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved ADL model → {adl_dir}")
    print("Run: python scripts/sync_inference_manifest.py  (from repo root, PYTHONPATH=.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
