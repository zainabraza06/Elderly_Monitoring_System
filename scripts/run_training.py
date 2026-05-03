"""
Unified training dispatcher — saves **best models** under ``models/`` and **reports** under ``results/``.

Run from repository root:

  set PYTHONPATH=scripts
  py scripts/run_training.py fall-detection --data-root "path\\to\\Annotated Data"
  py scripts/run_training.py adl --data-root "path\\to\\Annotated Data"
  py scripts/run_training.py fall-type --data-root "path\\to\\Annotated Data"
  py scripts/run_training.py all --data-root "path\\to\\Annotated Data"
  py scripts/run_training.py sync-manifest

``all`` runs the three pipelines in order and then ``sync_inference_manifest.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_scripts_path() -> None:
    root = _repo_root()
    s = str(root / "scripts")
    if s not in sys.path:
        sys.path.insert(0, s)


def _py_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root() / "scripts")
    return env


def main() -> int:
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Train baseline models (fall / ADL / fall-type).")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fd = sub.add_parser("fall-detection", help="Binary fall vs ADL (116-D)")
    p_fd.add_argument("--data-root", type=Path, required=True)
    p_fd.add_argument("--models-dir", type=Path, default=root / "models" / "baseline_fall")
    p_fd.add_argument("--results-dir", type=Path, default=root / "results" / "baseline_fall")
    p_fd.add_argument("--xgb-only", action="store_true")
    p_fd.add_argument("--skip-plots", action="store_true")
    p_fd.add_argument("--skip-cv", action="store_true")

    p_adl = sub.add_parser("adl", help="ADL multiclass (116-D)")
    p_adl.add_argument("--data-root", type=Path, required=True)

    p_ft = sub.add_parser("fall-type", help="Four-class fall type (263-D + MI)")
    p_ft.add_argument("--data-root", type=Path, required=True)

    p_all = sub.add_parser("all", help="fall-detection + adl + fall-type + sync manifest")
    p_all.add_argument("--data-root", type=Path, required=True)

    sub.add_parser("sync-manifest", help="Update models/inference_manifest.json from joblib shapes")

    args = parser.parse_args()

    if args.command == "sync-manifest":
        sync = root / "scripts" / "sync_inference_manifest.py"
        r = subprocess.run([sys.executable, str(sync)], cwd=str(root))
        return int(r.returncode)

    _ensure_scripts_path()

    if args.command == "fall-detection":
        from baseline_fall.train_fall_detection_mobiact import main as run_fd

        sys.argv = [
            "train_fall_detection_mobiact",
            "--data-root",
            str(args.data_root),
            "--models-dir",
            str(args.models_dir),
            "--results-dir",
            str(args.results_dir),
        ]
        if args.xgb_only:
            sys.argv.append("--xgb-only")
        if args.skip_plots:
            sys.argv.append("--skip-plots")
        if args.skip_cv:
            sys.argv.append("--skip-cv")
        return int(run_fd())

    if args.command == "adl":
        from baseline_adl.train_mobiact_adl import main as run_adl

        sys.argv = ["train_mobiact_adl", "--data-root", str(args.data_root)]
        return int(run_adl())

    if args.command == "fall-type":
        from baseline_falltype.train_fall_type_mobiact import main as run_ft

        sys.argv = ["train_fall_type_mobiact", "--data-root", str(args.data_root)]
        return int(run_ft())

    if args.command == "all":
        dr = args.data_root
        runner = root / "scripts" / "run_training.py"
        env = _py_env()
        for label, argv_extra in [
            ("fall-detection", ["fall-detection", "--data-root", str(dr)]),
            ("adl", ["adl", "--data-root", str(dr)]),
            ("fall-type", ["fall-type", "--data-root", str(dr)]),
        ]:
            rc = subprocess.run(
                [sys.executable, str(runner), *argv_extra],
                cwd=str(root),
                env=env,
            )
            if rc.returncode != 0:
                print(f"run_training [{label}] failed with code {rc.returncode}", file=sys.stderr)
                return int(rc.returncode)
        sync = root / "scripts" / "sync_inference_manifest.py"
        r = subprocess.run([sys.executable, str(sync)], cwd=str(root))
        return int(r.returncode)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
