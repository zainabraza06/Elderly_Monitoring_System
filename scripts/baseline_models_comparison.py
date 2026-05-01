import argparse
import os

import numpy as np

from baseline.Dl import run_dl_experiments
from baseline.ML import get_ml_results
from baseline.results import print_final_summary, print_section, save_results_excel
from baseline.visulaition import plot_comparison


def main():
    parser = argparse.ArgumentParser(description="Baseline models comparison (DL + ML).")
    parser.add_argument(
        "--npz-path",
        required=True,
        help="Path to .npz containing arrays: X_train_raw, y_fall_train, X_test_raw, y_fall_test, X_train_adl, y_train_adl, X_test_adl, y_test_adl.",
    )
    parser.add_argument("--results-dir", default="results", help="Directory for plots and reports.")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    X_train_raw = data["X_train_raw"]
    y_fall_train = data["y_fall_train"]
    X_test_raw = data["X_test_raw"]
    y_fall_test = data["y_fall_test"]
    X_train_adl = data["X_train_adl"]
    y_train_adl = data["y_train_adl"]
    X_test_adl = data["X_test_adl"]
    y_test_adl = data["y_test_adl"]

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

    ml_fall_results, ml_adl_results = get_ml_results()

    if not ml_fall_results or not ml_adl_results:
        print("\nML results are empty. Add your ML results before plotting and saving.")
        return

    plot_comparison(ml_fall_results, dl_fall_results, ml_adl_results, dl_adl_results, args.results_dir)
    excel_path = save_results_excel(
        ml_fall_results,
        dl_fall_results,
        ml_adl_results,
        dl_adl_results,
        args.results_dir,
    )

    best_fall_ml = ml_fall_results[0]
    best_adl_ml = ml_adl_results[0]
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


if __name__ == "__main__":
    main()
