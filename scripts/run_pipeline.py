from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run preprocessing -> feature extraction -> model training")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--artifacts", type=Path, default=Path("results/artifacts"))
    p.add_argument("--sisfall-only", action="store_true")
    p.add_argument("--max-files-per-dataset", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)

    windows = args.artifacts / "windows.pkl"
    features = args.artifacts / "features.pkl"
    fall_model = args.artifacts / "fall_detector.joblib"
    integration_notes = Path("results/mobile_export/integration_notes.md")

    if not windows.exists():
        preprocess_cmd = [
            sys.executable,
            str(Path(__file__).parent / "preprocessing.py"),
            "--data-root",
            str(args.data_root),
            "--output",
            str(windows),
            "--target-fs",
            "50",
            "--window-sec",
            "2.56",
            "--overlap",
            "0.5",
        ]

        if args.sisfall_only:
            preprocess_cmd.append("--sisfall-only")
        if args.max_files_per_dataset is not None:
            preprocess_cmd.extend(["--max-files-per-dataset", str(args.max_files_per_dataset)])

        run_cmd(preprocess_cmd)
    else:
        print(f"Skipping preprocessing, {windows} already exists.")

    if not features.exists():
        run_cmd(
            [
                sys.executable,
                str(Path(__file__).parent / "feature_extraction.py"),
                "--windows",
                str(windows),
                "--output",
                str(features),
                "--target-fs",
                "50",
            ]
        )
    else:
        print(f"Skipping feature extraction, {features} already exists.")

    if not fall_model.exists():
        run_cmd(
            [
                sys.executable,
                str(Path(__file__).parent / "modeling.py"),
                "--features",
                str(features),
                "--output-dir",
                str(args.artifacts),
            ]
        )
    else:
        print(f"Skipping modeling, {fall_model} already exists.")

    if not integration_notes.exists():
        run_cmd(
            [
                sys.executable,
                str(Path(__file__).parent / "app_integration.py"),
                "--artifacts",
                str(args.artifacts),
                "--output",
                str(Path("results/mobile_export")),
            ]
        )
    else:
        print(f"Skipping app integration, {integration_notes} already exists.")

    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
