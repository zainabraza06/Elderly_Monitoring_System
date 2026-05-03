"""
Train Task 1 (fall vs ADL) and Task 2 (ADL multiclass) with 116-D enhanced features.
Matches the Colab notebook: RobustScaler, SMOTETomek, ADASYN, XGBoost.
Saves to models/baseline_fall/ and models/baseline_adl/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from xgboost import XGBClassifier

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

from baseline_fall.enhanced_features import ENHANCED_FEATURE_DIM, extract_enhanced_features
from baseline_fall.mobiact_dataset import discover_data_root, load_sliding_windows

RANDOM_STATE = 42


def _balance_fall(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        smt = SMOTETomek(random_state=RANDOM_STATE, sampling_strategy=0.5)
        return smt.fit_resample(X, y)
    except Exception:
        smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.5)
        return smote.fit_resample(X, y)


def _balance_adl(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return ADASYN(random_state=RANDOM_STATE, sampling_strategy="auto").fit_resample(X, y)
    except Exception:
        return SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto").fit_resample(X, y)


def subject_masks(
    subject_ids: np.ndarray,
    y_fall: np.ndarray,
    frac: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    y_len = len(y_fall)
    subs = list({str(s) for s in subject_ids[:y_len]})
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(subs)
    if len(subs) < 2:
        idx = np.arange(y_len)
        try:
            tr, te = train_test_split(
                idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_fall
            )
        except ValueError:
            tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
        train_m = np.zeros(y_len, dtype=bool)
        test_m = np.zeros(y_len, dtype=bool)
        train_m[tr] = True
        test_m[te] = True
        return train_m, test_m

    n_tr = max(1, int(frac * len(subs)))
    train_s = set(subs[:n_tr])
    test_s = set(subs[n_tr:])
    train_m = np.array([str(s) in train_s for s in subject_ids[:y_len]])
    test_m = np.array([str(s) in test_s for s in subject_ids[:y_len]])
    return train_m, test_m


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

    train_m, test_m = subject_masks(subject_ids, y_fall)
    X_tr, X_te = X_feat[train_m], X_feat[test_m]
    yf_tr, yf_te = y_fall[train_m], y_fall[test_m]

    scaler_fall = RobustScaler()
    X_tr_s = scaler_fall.fit_transform(X_tr)
    X_te_s = scaler_fall.transform(X_te)

    X_tr_bal, yf_tr_bal = _balance_fall(X_tr_s, yf_tr)

    xgb_fall = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )

    print("CV (fall, F1)...")
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(xgb_fall, X_tr_bal, yf_tr_bal, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    t0 = time.perf_counter()
    xgb_fall.fit(X_tr_bal, yf_tr_bal)
    pred_f = xgb_fall.predict(X_te_s)
    print(f"Fall test — acc={accuracy_score(yf_te, pred_f):.4f}, F1={f1_score(yf_te, pred_f):.4f}, time={time.perf_counter()-t0:.1f}s")

    fall_dir = args.models_dir / "baseline_fall"
    fall_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_fall, fall_dir / "fall_detection_xgboost.pkl")
    joblib.dump(scaler_fall, fall_dir / "scaler_fall.pkl")
    (fall_dir / "fall_metrics.json").write_text(
        json.dumps(
            {
                "feature_dim": ENHANCED_FEATURE_DIM,
                "cv_f1_mean": float(cv_scores.mean()),
                "cv_f1_std": float(cv_scores.std()),
                "test_accuracy": float(accuracy_score(yf_te, pred_f)),
                "test_f1": float(f1_score(yf_te, pred_f)),
            },
            indent=2,
        ),
        encoding="utf-8",
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

    Xa_bal, ya_bal = _balance_adl(Xa_tr_s, ya_tr)

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
