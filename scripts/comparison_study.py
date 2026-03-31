from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_evaluation_results(artifacts_dir: Path) -> pd.DataFrame:
    """Load all evaluation.json files from the artifacts directory."""
    results = []
    for eval_path in sorted(artifacts_dir.glob("**/evaluation.json")):
        try:
            with eval_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            
            model_name = data.get("model_name", "Unknown")
            if "temporal" in str(eval_path):
                model_name = f"Temporal_{model_name}"

            # Extract fall detection metrics
            fall_metrics = data.get("classification_report", {}).get("1", {})
            if not fall_metrics:
                # Handle case where report is a string
                continue

            results.append({
                "model": model_name,
                "fall_precision": fall_metrics.get("precision"),
                "fall_recall": fall_metrics.get("recall"),
                "fall_f1_score": fall_metrics.get("f1-score"),
                "accuracy": data.get("accuracy"),
                "source_file": str(eval_path.relative_to(artifacts_dir)),
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Could not process {eval_path}: {e}")
            continue
            
    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare model performance from evaluation artifacts.")
    p.add_argument("--artifacts", type=Path, default=Path("results/artifacts"),
                   help="Directory containing model evaluation files.")
    p.add_argument("--output", type=Path, default=Path("results/comparison"),
                   help="Directory to save comparison results.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    results_df = load_evaluation_results(args.artifacts)

    if results_df.empty:
        print("No evaluation results found to compare.")
        return

    # Identify the best model based on F1 score for fall detection
    best_model = results_df.loc[results_df["fall_f1_score"].idxmax()]

    print("--- Model Comparison ---")
    print(results_df.to_string(index=False))
    print("\n--- Best Model for Fall Detection ---")
    print(f"Model: {best_model['model']}")
    print(f"Fall F1-Score: {best_model['fall_f1_score']:.4f}")
    print(f"Source: {best_model['source_file']}")

    # Save the comparison table
    output_path = args.output / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nComparison results saved to {output_path}")


if __name__ == "__main__":
    main()
