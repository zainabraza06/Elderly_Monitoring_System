import argparse
import os
import sys
from pathlib import Path

import numpy as np

from baseline_fallandadl.Dl import run_dl_experiments
from baseline_fallandadl.ML import run_ml_experiments
from baseline_fallandadl.mobiact_loader import default_data_root, default_npz_cache_path, load_or_build_npz
from baseline_fallandadl.results import print_final_summary, print_section, save_results_excel
from baseline_fallandadl.visulaition import plot_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline model comparison (DL + ML). "
        "Omit --npz-path to build/load windows from extracted MobiAct under data/.",
    )
    parser.add_argument(
        "--npz-path",
        default=None,
        help=(
            "Path to .npz with arrays: X_train_raw, y_fall_train, X_test_raw, y_fall_test, "
            "X_train_adl, y_train_adl, X_test_adl, y_test_adl. Optional fused sensors (same window shapes as acc): "
            "X_gyro_train, X_gyro_test, X_ori_train, X_ori_test, X_gyro_train_adl, X_gyro_test_adl, "
            "X_ori_train_adl, X_ori_test_adl. If omitted, uses --data-root / MobiAct Annotated Data."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Folder containing extracted MobiAct (default: {default_data_root()}).",
    )
    parser.add_argument(
        "--mobiact-cache",
        type=Path,
        default=None,
        help=f"Where to cache the built .npz (default: {default_npz_cache_path()}).",
    )
    parser.add_argument(
        "--rebuild-mobiact",
        action="store_true",
        help="Rebuild cached npz from CSV even if it already exists.",
    )
    parser.add_argument("--results-dir", default="results", help="Directory for plots and reports.")
    args = parser.parse_args()

    if args.npz_path:
        npz_path = Path(args.npz_path)
    else:
        npz_path = Path(
            load_or_build_npz(
                data_root=args.data_root,
                output_npz=args.mobiact_cache,
                force_rebuild=args.rebuild_mobiact,
            )
        )

    if not npz_path.is_file():
        print(f"Missing data file: {npz_path}")
        return 1

    data = np.load(npz_path, allow_pickle=True)
    X_train_raw = data["X_train_raw"]
    y_fall_train = data["y_fall_train"]
    X_test_raw = data["X_test_raw"]
    y_fall_test = data["y_fall_test"]
    X_train_adl = data["X_train_adl"]
    y_train_adl = data["y_train_adl"]
    X_test_adl = data["X_test_adl"]
    y_test_adl = data["y_test_adl"]

    X_gyro_train = data["X_gyro_train"] if "X_gyro_train" in data.files else None
    X_gyro_test = data["X_gyro_test"] if "X_gyro_test" in data.files else None
    X_ori_train = data["X_ori_train"] if "X_ori_train" in data.files else None
    X_ori_test = data["X_ori_test"] if "X_ori_test" in data.files else None
    X_gyro_train_adl = data["X_gyro_train_adl"] if "X_gyro_train_adl" in data.files else None
    X_gyro_test_adl = data["X_gyro_test_adl"] if "X_gyro_test_adl" in data.files else None
    X_ori_train_adl = data["X_ori_train_adl"] if "X_ori_train_adl" in data.files else None
    X_ori_test_adl = data["X_ori_test_adl"] if "X_ori_test_adl" in data.files else None

    os.makedirs(args.results_dir, exist_ok=True)

    print_section("DEEP LEARNING MODELS (Using ALL Data - NO SAMPLING)")
    dl_fall_results, dl_adl_results = run_dl_experiments(
        X_train_raw,
        y_fall_train,
        X_test_raw,
        y_fall_test,
        X_train_adl,
        y_train_adl,
        X_test_adl,
        y_test_adl,
    )

    print_section("CLASSICAL ML (enhanced features + SMOTETomek / ADASYN)")
    ml_fall_results, ml_adl_results = run_ml_experiments(
        X_train_raw,
        y_fall_train,
        X_test_raw,
        y_fall_test,
        X_train_adl,
        y_train_adl,
        X_test_adl,
        y_test_adl,
        X_gyro_train=X_gyro_train,
        X_gyro_test=X_gyro_test,
        X_ori_train=X_ori_train,
        X_ori_test=X_ori_test,
        X_gyro_train_adl=X_gyro_train_adl,
        X_gyro_test_adl=X_gyro_test_adl,
        X_ori_train_adl=X_ori_train_adl,
        X_ori_test_adl=X_ori_test_adl,
    )

    if not ml_fall_results or not ml_adl_results:
        print("\nML results are empty.")
        return 1

    plot_comparison(ml_fall_results, dl_fall_results, ml_adl_results, dl_adl_results, args.results_dir)
    excel_path = save_results_excel(
        ml_fall_results,
        dl_fall_results,
        ml_adl_results,
        dl_adl_results,
        args.results_dir,
    )

    best_fall_ml = max(ml_fall_results, key=lambda x: x["F1"])
    best_adl_ml = max(ml_adl_results, key=lambda x: x["F1"])
    best_fall_dl = dl_fall_results[3] if dl_fall_results[3]["F1"] > dl_fall_results[4]["F1"] else dl_fall_results[4]
    best_adl_dl = dl_adl_results[4] if dl_adl_results[4]["F1"] > dl_adl_results[3]["F1"] else dl_adl_results[3]

    print_final_summary(
        best_fall_ml,
        best_fall_dl,
        best_adl_ml,
        best_adl_dl,
        excel_path,
        args.results_dir,
        "models",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
