from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from data_utils import load_features
from evaluation import (
    classification_metrics,
    regression_metrics,
    save_confusion_matrix,
    save_metrics_report,
    save_roc_curve,
)


try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False


META_COLUMNS = {
    "dataset",
    "subject",
    "age_group",
    "activity_code",
    "fall_label",
    "met_class",
    "frailty_proxy",
    "gait_stability_proxy",
    "movement_disorder_proxy",
}


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in META_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def group_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(df))
    train_idx, test_idx = next(gss.split(idx, groups=df[group_col]))
    return train_idx, test_idx


def build_fall_model(seed: int) -> object:
    """
    Builds a BalancedRandomForestClassifier, which is well-suited for imbalanced
    datasets like fall detection. It undersamples the majority class in each
    bootstrap sample to balance the class distribution.
    """
    return BalancedRandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        sampling_strategy="auto",  # Resamples each bootstrap sample
        replacement=False,  # No replacement in undersampling
        random_state=seed,
        n_jobs=-1,
    )


def save_fall_runtime_bundle(
    model: object,
    feature_cols: List[str],
    out_dir: Path,
    target_fs: int,
    window_sec: float,
    overlap: float,
) -> None:
    window_size_samples = int(round(target_fs * window_sec))
    step_size_samples = max(1, int(round(window_size_samples * (1.0 - overlap))))

    bundle = {
        "model": model,
        "feature_columns": feature_cols,
        "target_fs": target_fs,
        "window_sec": window_sec,
        "overlap": overlap,
        "window_size_samples": window_size_samples,
        "step_size_samples": step_size_samples,
        "model_name": "fall_detector",
    }
    joblib.dump(bundle, out_dir / "fall_detector_bundle.joblib")

    metadata = {
        "feature_columns": feature_cols,
        "target_fs": target_fs,
        "window_sec": window_sec,
        "overlap": overlap,
        "window_size_samples": window_size_samples,
        "step_size_samples": step_size_samples,
        "model_name": "fall_detector",
    }
    with (out_dir / "fall_detector_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def train_fall_detector(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    seed: int,
    target_fs: int,
    window_sec: float,
    overlap: float,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)

    external_mode = (df["dataset"] == "sisfall").any() and (df["dataset"] != "sisfall").any()
    if external_mode:
        train_df = df[df["dataset"] != "sisfall"].copy()
        test_df = df[df["dataset"] == "sisfall"].copy()
        if train_df["fall_label"].nunique() < 2:
            external_mode = False

    if not external_mode:
        train_idx, test_idx = group_train_test_split(df, group_col="subject", test_size=0.2, seed=seed)
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["fall_label"].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df["fall_label"].to_numpy(dtype=int)

    model = build_fall_model(seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = classification_metrics(y_test, y_pred, y_prob)
    metrics["split"] = "cross_dataset_sisfall_external" if external_mode else "group_split"

    joblib.dump(model, out_dir / "fall_detector.joblib")
    save_fall_runtime_bundle(
        model=model,
        feature_cols=feature_cols,
        out_dir=out_dir,
        target_fs=target_fs,
        window_sec=window_sec,
        overlap=overlap,
    )
    save_confusion_matrix(y_test, y_pred, out_dir / "fall_confusion_matrix.png", "Fall vs Non-Fall")
    if y_prob is not None and len(np.unique(y_test)) == 2:
        save_roc_curve(y_test, y_prob, out_dir / "fall_roc_curve.png", "Fall Detector ROC")

    return metrics


def sisfall_loso_fall(df: pd.DataFrame, feature_cols: List[str], out_dir: Path, seed: int) -> Dict[str, float]:
    sis = df[df["dataset"] == "sisfall"].copy()
    if sis.empty or sis["subject"].nunique() < 3:
        return {}

    gkf = GroupKFold(n_splits=min(5, sis["subject"].nunique()))
    X = sis[feature_cols].to_numpy(dtype=np.float32)
    y = sis["fall_label"].to_numpy(dtype=int)
    groups = sis["subject"].to_numpy()

    fold_scores: List[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue
        model = build_fall_model(seed)
        model.fit(X[train_idx], y_train)
        pred = model.predict(X[test_idx])
        fold_scores.append(float((pred == y[test_idx]).mean()))

    if not fold_scores:
        return {}

    return {
        "loso_acc_mean": float(np.mean(fold_scores)),
        "loso_acc_std": float(np.std(fold_scores)),
        "loso_folds": float(len(fold_scores)),
    }


def train_met_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    seed: int,
) -> Dict[str, float]:
    met_df = df[(df["fall_label"] == 0) & (df["met_class"].isin(["light", "moderate", "vigorous"]))].copy()
    if met_df.empty or met_df["met_class"].nunique() < 2:
        return {"status": "skipped"}

    train_idx, test_idx = group_train_test_split(met_df, group_col="subject", test_size=0.2, seed=seed)
    train_df = met_df.iloc[train_idx]
    test_df = met_df.iloc[test_idx]

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(met_df["met_class"])
    y_train = le.transform(train_df["met_class"])
    y_test = le.transform(test_df["met_class"])

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)

    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "f1_macro": float(
            classification_report(y_test, y_pred, output_dict=True, zero_division=0).get("macro avg", {}).get("f1-score", 0.0)
        ),
    }

    joblib.dump(model, out_dir / "met_classifier.joblib")
    joblib.dump(le, out_dir / "met_label_encoder.joblib")
    save_confusion_matrix(y_test, y_pred, out_dir / "met_confusion_matrix.png", "MET Classifier")

    return metrics


def train_proxy_regressor(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    seed: int,
) -> Dict[str, float]:
    reg_df = df[df["fall_label"] == 0].copy()
    if reg_df.shape[0] < 100:
        return {"status": "skipped"}

    y_cols = ["frailty_proxy", "gait_stability_proxy", "movement_disorder_proxy"]

    train_idx, test_idx = group_train_test_split(reg_df, group_col="subject", test_size=0.2, seed=seed)
    train_df = reg_df.iloc[train_idx]
    test_df = reg_df.iloc[test_idx]

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df[y_cols].to_numpy(dtype=np.float32)
    y_test = test_df[y_cols].to_numpy(dtype=np.float32)

    base = GradientBoostingRegressor(random_state=seed)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MultiOutputRegressor(base)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {}
    for i, name in enumerate(y_cols):
        m = regression_metrics(y_test[:, i], y_pred[:, i])
        metrics[f"{name}_mae"] = m["mae"]
        metrics[f"{name}_rmse"] = m["rmse"]
        metrics[f"{name}_r2"] = m["r2"]

    joblib.dump(model, out_dir / "proxy_regressor.joblib")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mobility-health models from feature table.")
    p.add_argument("--features", type=Path, default=Path("results/artifacts/features.pkl"))
    p.add_argument("--output-dir", type=Path, default=Path("results/artifacts"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-fs", type=int, default=50)
    p.add_argument("--window-sec", type=float, default=2.56)
    p.add_argument("--overlap", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = load_features(args.features)
    if df.empty:
        print("Features table is empty.")
        return

    feature_cols = get_feature_columns(df)
    if not feature_cols:
        print("No numeric feature columns found.")
        return

    metrics_all: Dict[str, Dict[str, float]] = {}

    fall_metrics = train_fall_detector(
        df,
        feature_cols,
        args.output_dir,
        args.seed,
        args.target_fs,
        args.window_sec,
        args.overlap,
    )
    metrics_all["fall_detector"] = fall_metrics

    loso_metrics = sisfall_loso_fall(df, feature_cols, args.output_dir, args.seed)
    if loso_metrics:
        metrics_all["fall_loso"] = loso_metrics

    met_metrics = train_met_classifier(df, feature_cols, args.output_dir, args.seed)
    metrics_all["met_classifier"] = met_metrics

    reg_metrics = train_proxy_regressor(df, feature_cols, args.output_dir, args.seed)
    metrics_all["proxy_regressor"] = reg_metrics

    save_metrics_report(metrics_all, args.output_dir / "metrics_summary.csv")
    with (args.output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_all, f, indent=2)

    print("Training complete.")
    print(json.dumps(metrics_all, indent=2))


if __name__ == "__main__":
    main()
