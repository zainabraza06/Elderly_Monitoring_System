"""MobiAct annotated CSV → sliding windows (Colab CELL 2 logic)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from baseline_falltype.data_loader import (
    _find_acc_columns,
    _find_gyro_columns,
    _find_ori_columns,
    find_annotated_data_dir,
)

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


def _subject_from_parts(parts: list[str]) -> str:
    if len(parts) >= 2 and parts[1].upper().startswith("S") and parts[1][1:].isdigit():
        return parts[1]
    if len(parts) >= 3:
        return parts[2]
    return "unknown"


def load_sliding_windows(annotated_dir: Path) -> dict[str, np.ndarray | list]:
    all_files: list[Path] = []
    for root, _, files in os.walk(annotated_dir):
        for file in files:
            if file.endswith("_annotated.csv"):
                all_files.append(Path(root) / file)

    if not all_files:
        raise RuntimeError(f"No *_annotated.csv under {annotated_dir}")

    X_acc_list: list[np.ndarray] = []
    X_gyro_list: list[np.ndarray] = []
    X_ori_list: list[np.ndarray] = []
    y_fall_list: list[int] = []
    y_adl_str_list: list[str] = []
    subject_ids_list: list[str] = []

    for file_path in tqdm(all_files, desc="MobiAct CSV"):
        try:
            df = pd.read_csv(file_path)
            acc_cols = _find_acc_columns(df)
            if acc_cols is None:
                continue
            gyro_cols = _find_gyro_columns(df)
            ori_cols = _find_ori_columns(df)

            parts = file_path.name.replace("_annotated.csv", "").split("_")
            activity_code = parts[0].upper() if parts else ""
            subject_id = _subject_from_parts(parts)

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
                if len(acc_window) != FIXED_LEN:
                    indices = np.linspace(0, len(acc_window) - 1, FIXED_LEN, dtype=int)
                    acc_window = acc_window[indices]
                else:
                    indices = np.arange(FIXED_LEN, dtype=int)

                X_acc_list.append(acc_window.astype(np.float64))

                if gyro_data is not None and len(gyro_data) >= end:
                    gyro_window = gyro_data[start:end]
                    if len(gyro_window) != FIXED_LEN:
                        gyro_window = gyro_window[indices]
                    X_gyro_list.append(gyro_window.astype(np.float64))
                else:
                    X_gyro_list.append(np.zeros((FIXED_LEN, 3), dtype=np.float64))

                if ori_data is not None and len(ori_data) >= end:
                    ori_window = ori_data[start:end]
                    if len(ori_window) != FIXED_LEN:
                        ori_window = ori_window[indices]
                    X_ori_list.append(ori_window.astype(np.float64))
                else:
                    X_ori_list.append(np.zeros((FIXED_LEN, 3), dtype=np.float64))

                y_fall_list.append(1 if is_fall else 0)
                y_adl_str_list.append(adl_label)
                subject_ids_list.append(subject_id)

        except Exception:
            continue

    if not X_acc_list:
        raise RuntimeError("No windows extracted — check CSV columns / paths.")

    adl_enc = LabelEncoder()
    y_adl = adl_enc.fit_transform(np.asarray(y_adl_str_list, dtype=object))

    return {
        "X_acc": np.stack(X_acc_list),
        "X_gyro": np.stack(X_gyro_list),
        "X_ori": np.stack(X_ori_list),
        "y_fall": np.asarray(y_fall_list, dtype=np.int64),
        "y_adl": y_adl,
        "adl_encoder": adl_enc,
        "subject_ids": np.asarray(subject_ids_list, dtype=object),
    }


def discover_data_root(explicit: Path | None) -> Path:
    root = explicit if explicit is not None else Path(__file__).resolve().parents[2] / "data"
    ann = find_annotated_data_dir(root)
    if ann is None:
        raise FileNotFoundError(f"No Annotated Data under {root}")
    return ann
