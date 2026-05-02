"""
CLI: 4-class MobiAct fall-type classification (multi-sensor: ACC, gyro, orientation).

From the repository root, with ``scripts`` on the path (or run from ``scripts``)::

    cd scripts
    python -m baseline_falltype.run --mobiact-root "path/to/MobiAct_Dataset_v2.0"

Google Colab::

    python -m baseline_falltype.run --mount-colab-drive --rar /content/drive/MyDrive/mobiact.rar \\
        --extract-to /content/mobiact
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import joblib
import numpy as np

from baseline_falltype.config import (
    DEFAULT_MODELS_DIR,
    DEFAULT_RESULTS_DIR,
    FALL_NAMES,
    RANDOM_STATE,
    SAMPLING_RATE_HZ,
    TOP_N_FEATURES,
)
from baseline_falltype.data_loader import (
    extract_rar,
    find_annotated_data_dir,
    load_multisensor_fall_windows,
)
from baseline_falltype.pipeline import (
    build_feature_matrix,
    save_metadata,
    save_reports,
    select_and_split,
    train_models,
)
from baseline_falltype.visualization import save_all_figures


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MobiAct 4-class fall-type classification (multi-sensor pipeline)."
    )
    p.add_argument(
        "--mobiact-root",
        type=str,
        default=None,
        help="Folder containing MobiAct (parent of 'Annotated Data' or dataset root).",
    )
    p.add_argument(
        "--rar",
        type=str,
        default=None,
        help="Path to mobiact.rar; extract with unrar/7z/patool into --extract-to.",
    )
    p.add_argument(
        "--extract-to",
        type=str,
        default=None,
        help="Extraction directory (default: <repo>/data/mobiact).",
    )
    p.add_argument(
        "--models-dir",
        type=str,
        default=str(DEFAULT_MODELS_DIR),
        help=f"Models output (default: {DEFAULT_MODELS_DIR}).",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Results root (default: {DEFAULT_RESULTS_DIR}).",
    )
    p.add_argument("--top-n", type=int, default=TOP_N_FEATURES, help="Top features (mutual information).")
    p.add_argument(
        "--fs",
        type=float,
        default=SAMPLING_RATE_HZ,
        help="Assumed sampling rate in Hz (default: 50, for filters and spectral features).",
    )
    p.add_argument(
        "--force-features",
        action="store_true",
        help="Recompute cached feature matrix (ignore multisensor_features.pkl).",
    )
    p.add_argument(
        "--mount-colab-drive",
        action="store_true",
        help="Mount Google Drive when running in Colab.",
    )
    return p.parse_args()


def _resolve_extract_to(args: argparse.Namespace) -> Path:
    if args.extract_to:
        return Path(args.extract_to)
    from baseline_falltype.config import REPO_ROOT

    return REPO_ROOT / "data" / "mobiact"


def _maybe_mount_colab() -> None:
    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
    except Exception:
        pass


def main() -> int:
    args = _parse_args()
    if args.mount_colab_drive:
        _maybe_mount_colab()

    models_dir = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    cache_dir = results_dir / "cache"
    reports_dir = results_dir / "reports"
    for d in (models_dir, results_dir, cache_dir, reports_dir, results_dir / "figures"):
        d.mkdir(parents=True, exist_ok=True)

    if args.rar:
        extract_root = _resolve_extract_to(args)
        if not extract_root.exists() or not any(extract_root.iterdir()):
            if not Path(args.rar).is_file():
                print(f"RAR not found: {args.rar}", file=sys.stderr)
                return 1
            if not extract_rar(args.rar, extract_root):
                return 1
        search_root = extract_root
    else:
        search_root = Path(args.mobiact_root) if args.mobiact_root else None

    if search_root is None or not search_root.exists():
        print("Set --mobiact-root or use --rar to extract the dataset first.", file=sys.stderr)
        return 1

    mobiact_path = find_annotated_data_dir(search_root)
    if mobiact_path is None:
        print(f"Could not find 'Annotated Data' under {search_root}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("4-class fall type classification — multi-sensor (MobiAct)")
    print(f"Annotated data: {mobiact_path}")
    print(f"Models: {models_dir}")
    print(f"Results: {results_dir}")
    print("=" * 80)

    acc_w, gyro_w, ori_w, y_enc, le = load_multisensor_fall_windows(mobiact_path)
    n = acc_w.shape[0]
    print(f"Samples: {n}  window: {acc_w.shape[1:]}")
    gyro_nonzero = int(np.sum(~np.all(gyro_w == 0, axis=(1, 2))))
    ori_nonzero = int(np.sum(~np.all(ori_w == 0, axis=(1, 2))))
    print(f"  Gyroscope non-zero windows: {gyro_nonzero}")
    print(f"  Orientation non-zero windows: {ori_nonzero}")

    for i, code in enumerate(le.classes_):
        c = int(np.sum(y_enc == i))
        name = FALL_NAMES.get(code, code)
        print(f"  {code} ({name}): {c}")

    np.save(cache_dir / "acc_windows.npy", acc_w)
    np.save(cache_dir / "gyro_windows.npy", gyro_w)
    np.save(cache_dir / "ori_windows.npy", ori_w)
    np.save(cache_dir / "y_encoded.npy", y_enc)
    joblib.dump(le, models_dir / "label_encoder.pkl")

    X_feat = build_feature_matrix(
        acc_w,
        gyro_w,
        ori_w,
        cache_dir,
        force_recompute=args.force_features,
        fs_hz=args.fs,
    )
    print(f"Feature matrix: {X_feat.shape}")

    X_train, X_test, y_train, y_test, scaler, top_features, k_selected = select_and_split(
        X_feat, y_enc, top_n=args.top_n, random_state=RANDOM_STATE
    )
    print(f"MI selected features: {k_selected}  |  Train: {len(X_train)}  Test: {len(X_test)}")

    results, _models, best_pred = train_models(X_train, y_train, X_test, y_test)
    best_name = max(results, key=lambda r: results[r]["accuracy"])
    best_accuracy = results[best_name]["accuracy"]
    best_model = _models[best_name]

    print(f"Best model: {best_name}  accuracy: {best_accuracy * 100:.2f}%")

    joblib.dump(best_model, models_dir / "best_fall_classifier.pkl")
    joblib.dump(best_model, models_dir / "best_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(top_features, models_dir / "selected_features.pkl")

    save_metadata(
        models_dir / "model_metadata.json",
        best_name,
        results,
        list(le.classes_),
        k_selected,
        n,
        n_features_raw=int(X_feat.shape[1]),
    )
    save_reports(reports_dir, results, le, y_test, best_pred)
    save_all_figures(
        results_dir,
        le,
        y_test,
        best_pred,
        results,
        best_name,
        best_accuracy,
        y_enc,
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
