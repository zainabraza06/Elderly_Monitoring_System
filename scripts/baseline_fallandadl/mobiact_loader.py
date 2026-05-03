"""
Build the baseline .npz from extracted MobiAct (v2) under ``data/`` (or a custom root).

Expects the same layout as the Colab notebook: ``.../MobiAct_Dataset_v2.0/Annotated Data/**.csv``
or any tree that contains a folder named ``Annotated Data`` (see ``find_annotated_data_dir``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Reuse column detection from the fall-type baseline (robust to CSV headers).
from baseline_falltype.data_loader import (
    _find_acc_columns,
    _find_gyro_columns,
    _find_ori_columns,
    find_annotated_data_dir,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ADL_TYPES = {
    "STD": "Standing",
    "WAL": "Walking",
    "JOG": "Jogging",
    "JUM": "Jumping",
    "STU": "Stairs Up",
    "STN": "Stairs Down",
    "SCH": "Sit to Stand",
    "SIT": "Sitting",
    "CHU": "Stand to Sit",
    "CSI": "Car Step In",
    "CSO": "Car Step Out",
    "LYI": "Lying",
}

FALL_TYPES = {"FOL": "Forward Fall", "FKL": "Knees Fall", "BSC": "Back Fall", "SDL": "Side Fall"}

ALL_ACTIVITIES = list(ADL_TYPES.keys()) + list(FALL_TYPES.keys())

# Adaptive window length (samples), then resampled to FIXED_LEN (Colab / project ML baseline).
ACTIVITY_DURATIONS = {
    "SCH": 6,
    "CHU": 6,
    "CSI": 6,
    "CSO": 6,
    "STU": 10,
    "STN": 10,
    "JOG": 30,
    "JUM": 30,
    "FOL": 10,
    "FKL": 10,
    "BSC": 10,
    "SDL": 10,
    "STD": 300,
    "WAL": 300,
    "SIT": 60,
    "LYI": 300,
}

FIXED_LEN = 300
MIN_ADL_SAMPLES_PER_CLASS = 100
FALL_TRAIN_SUBJECT_FRAC = 0.8
RANDOM_STATE = 42


def repo_root() -> Path:
    return _REPO_ROOT


def default_data_root() -> Path:
    return _REPO_ROOT / "data"


def default_npz_cache_path() -> Path:
    return _REPO_ROOT / "results" / "artifacts" / "mobiact_baseline.npz"


def _subject_from_filename_parts(parts: list[str]) -> str:
    """MobiAct names look like ACT_Sxx_Ryy — subject is usually the Sxx token."""
    if len(parts) >= 2 and parts[1].upper().startswith("S") and parts[1][1:].isdigit():
        return parts[1]
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def discover_annotated_dir(search_root: Path | None) -> Path:
    root = search_root if search_root is not None else default_data_root()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Data root does not exist: {root}\n"
            "Place extracted MobiAct under data/MobiAct_Dataset_v2.0 (with Annotated Data/)."
        )
    ann = find_annotated_data_dir(root)
    if ann is None or not ann.is_dir():
        tried = root / "MobiAct_Dataset_v2.0" / "Annotated Data"
        raise FileNotFoundError(
            "Could not find 'Annotated Data' under "
            f"{root}. Expected e.g. {tried} or a nested MobiAct tree."
        )
    return ann


def _resample_window(block: np.ndarray, target_len: int) -> np.ndarray:
    """Linear index resample along time (rows)."""
    t = block.shape[0]
    if t == target_len:
        return block.astype(np.float64, copy=False)
    idx = np.linspace(0, t - 1, target_len, dtype=int)
    return block[idx].astype(np.float64)


def load_sliding_windows_from_annotated_dir(annotated_dir: Path) -> dict[str, Any]:
    """
    Walk all *_annotated.csv files; sliding windows per activity (Colab-style).

    Returns dict with numpy arrays and parallel lists turned into arrays:
      X_acc, X_gyro, X_ori, y_fall, y_adl_str, subject_ids (object array)
    """
    all_files: list[Path] = []
    for root, _, files in os.walk(annotated_dir):
        for file in files:
            if file.endswith("_annotated.csv"):
                all_files.append(Path(root) / file)

    if not all_files:
        raise RuntimeError(f"No *_annotated.csv files under {annotated_dir}")

    X_acc_list: list[np.ndarray] = []
    X_gyro_list: list[np.ndarray] = []
    X_ori_list: list[np.ndarray] = []
    y_fall_list: list[int] = []
    y_adl_str_list: list[str] = []
    subject_ids_list: list[str] = []

    for file_path in tqdm(all_files, desc="MobiAct CSV windows"):
        try:
            df = pd.read_csv(file_path)
            acc_cols = _find_acc_columns(df)
            if acc_cols is None:
                continue

            gyro_cols = _find_gyro_columns(df)
            ori_cols = _find_ori_columns(df)

            filename = file_path.name
            parts = filename.replace("_annotated.csv", "").split("_")
            activity_code = parts[0].upper() if parts else ""
            subject_id = _subject_from_filename_parts(parts)

            if activity_code not in ALL_ACTIVITIES:
                continue

            is_fall = activity_code in FALL_TYPES
            adl_label = "FALL" if is_fall else activity_code

            duration = ACTIVITY_DURATIONS.get(activity_code, 10)
            window_size = min(FIXED_LEN, max(100, int(duration * 10)))
            step = max(1, window_size // 2)

            acc_data = df[acc_cols[:3]].values.astype(np.float64)
            gyro_data = df[gyro_cols[:3]].values.astype(np.float64) if gyro_cols else None
            ori_data = df[ori_cols[:3]].values.astype(np.float64) if ori_cols else None

            if len(acc_data) <= window_size:
                continue

            for start in range(0, len(acc_data) - window_size, step):
                end = start + window_size
                acc_window = acc_data[start:end]

                acc_r = _resample_window(acc_window, FIXED_LEN)
                X_acc_list.append(acc_r)

                if gyro_data is not None and len(gyro_data) >= end:
                    gyro_window = gyro_data[start:end]
                    X_gyro_list.append(_resample_window(gyro_window, FIXED_LEN))
                else:
                    X_gyro_list.append(np.zeros((FIXED_LEN, 3), dtype=np.float64))

                if ori_data is not None and len(ori_data) >= end:
                    ori_window = ori_data[start:end]
                    X_ori_list.append(_resample_window(ori_window, FIXED_LEN))
                else:
                    X_ori_list.append(np.zeros((FIXED_LEN, 3), dtype=np.float64))

                y_fall_list.append(1 if is_fall else 0)
                y_adl_str_list.append(adl_label)
                subject_ids_list.append(subject_id)

        except Exception:
            continue

    if not X_acc_list:
        raise RuntimeError(
            f"No usable windows from {annotated_dir}. Check CSV columns (acc_x/acc_y/acc_z or similar)."
        )

    return {
        "X_acc": np.stack(X_acc_list, axis=0),
        "X_gyro": np.stack(X_gyro_list, axis=0),
        "X_ori": np.stack(X_ori_list, axis=0),
        "y_fall": np.asarray(y_fall_list, dtype=np.int64),
        "y_adl_str": np.asarray(y_adl_str_list, dtype=object),
        "subject_ids": np.asarray(subject_ids_list, dtype=object),
    }


def build_train_test_arrays(raw: dict[str, Any]) -> dict[str, np.ndarray]:
    """Subject-aware split for fall task; stratified random split for ADL (non-fall), matching Colab."""
    X_acc = raw["X_acc"]
    X_gyro = raw["X_gyro"]
    X_ori = raw["X_ori"]
    y_fall = raw["y_fall"]
    y_adl_str = raw["y_adl_str"]
    subject_ids = raw["subject_ids"]

    # Encode full ADL string labels for indexing (same classes as Colab before filtering).
    adl_enc = LabelEncoder()
    y_adl_all = adl_enc.fit_transform(y_adl_str)

    unique_subjects = list({str(s) for s in subject_ids})
    rng = np.random.default_rng(RANDOM_STATE)
    rng.shuffle(unique_subjects)

    if len(unique_subjects) < 2:
        idx = np.arange(len(X_acc))
        try:
            tr, te = train_test_split(
                idx,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y_fall,
            )
        except ValueError:
            tr, te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)
        train_mask = np.zeros(len(X_acc), dtype=bool)
        test_mask = np.zeros(len(X_acc), dtype=bool)
        train_mask[tr] = True
        test_mask[te] = True
    else:
        n_train = max(1, int(FALL_TRAIN_SUBJECT_FRAC * len(unique_subjects)))
        train_subjects = set(unique_subjects[:n_train])
        test_subjects = set(unique_subjects[n_train:])
        train_mask = np.array([str(sid) in train_subjects for sid in subject_ids])
        test_mask = np.array([str(sid) in test_subjects for sid in subject_ids])

    out: dict[str, np.ndarray] = {
        "X_train_raw": X_acc[train_mask],
        "X_test_raw": X_acc[test_mask],
        "y_fall_train": y_fall[train_mask],
        "y_fall_test": y_fall[test_mask],
        "X_gyro_train": X_gyro[train_mask],
        "X_gyro_test": X_gyro[test_mask],
        "X_ori_train": X_ori[train_mask],
        "X_ori_test": X_ori[test_mask],
    }

    non_fall = y_fall == 0
    X_nf = X_acc[non_fall]
    G_nf = X_gyro[non_fall]
    O_nf = X_ori[non_fall]
    y_nf = y_adl_all[non_fall]

    unique_cls, counts = np.unique(y_nf, return_counts=True)
    valid_cls = unique_cls[counts >= MIN_ADL_SAMPLES_PER_CLASS]
    keep = np.isin(y_nf, valid_cls)
    X_nf = X_nf[keep]
    G_nf = G_nf[keep]
    O_nf = O_nf[keep]
    y_nf = y_nf[keep]

    if len(X_nf) < 10:
        raise RuntimeError(
            "Too few ADL windows after rare-class filtering. Lower MIN_ADL_SAMPLES_PER_CLASS or check data."
        )

    le_clean = LabelEncoder()
    y_clean = le_clean.fit_transform(y_nf)

    try:
        (
            X_train_adl,
            X_test_adl,
            G_train_adl,
            G_test_adl,
            O_train_adl,
            O_test_adl,
            y_train_adl,
            y_test_adl,
        ) = train_test_split(
            X_nf,
            G_nf,
            O_nf,
            y_clean,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_clean,
        )
    except ValueError:
        (
            X_train_adl,
            X_test_adl,
            G_train_adl,
            G_test_adl,
            O_train_adl,
            O_test_adl,
            y_train_adl,
            y_test_adl,
        ) = train_test_split(
            X_nf,
            G_nf,
            O_nf,
            y_clean,
            test_size=0.2,
            random_state=RANDOM_STATE,
        )

    out["X_train_adl"] = X_train_adl
    out["X_test_adl"] = X_test_adl
    out["y_train_adl"] = y_train_adl.astype(np.int64)
    out["y_test_adl"] = y_test_adl.astype(np.int64)
    out["X_gyro_train_adl"] = G_train_adl
    out["X_gyro_test_adl"] = G_test_adl
    out["X_ori_train_adl"] = O_train_adl
    out["X_ori_test_adl"] = O_test_adl

    return out


def load_or_build_npz(
    data_root: Path | None = None,
    output_npz: Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """
    Return path to a .npz compatible with ``baseline_fallandadl.runner``.

    Caches under ``results/artifacts/mobiact_baseline.npz`` unless ``output_npz`` is set.
    """
    out_path = output_npz if output_npz is not None else default_npz_cache_path()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.is_file() and not force_rebuild:
        return out_path

    ann = discover_annotated_dir(data_root)
    print(f"Using MobiAct annotated data: {ann}")
    raw = load_sliding_windows_from_annotated_dir(ann)
    print(
        f"Built {len(raw['X_acc'])} windows "
        f"({raw['y_fall'].sum()} fall, {len(raw['y_fall']) - raw['y_fall'].sum()} non-fall)."
    )
    packed = build_train_test_arrays(raw)
    np.savez_compressed(out_path, **packed)
    print(f"Saved baseline bundle: {out_path}")
    return out_path


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Build mobiact_baseline.npz from extracted MobiAct under data/.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Folder containing MobiAct (default: {default_data_root()})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Output .npz (default: {default_npz_cache_path()})",
    )
    p.add_argument("--force", action="store_true", help="Rebuild even if output exists.")
    args = p.parse_args()
    try:
        path = load_or_build_npz(data_root=args.data_root, output_npz=args.output, force_rebuild=args.force)
        print(path)
        return 0
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
