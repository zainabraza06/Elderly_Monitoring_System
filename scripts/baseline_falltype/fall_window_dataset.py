"""
Load MobiAct fall-only windows (impact-centered, 300 samples) — Colab STEP 3.

Walks `.../Annotated Data` for `FOL_*`, `FKL_*`, `BSC_*`, `SDL_*_annotated.csv`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from baseline_falltype.config import FALL_CODES, WINDOW_SAMPLES


def _pick_columns(df, patterns: list[list[str]]) -> list[str] | None:
    for pat in patterns:
        if all(c in df.columns for c in pat):
            return pat
    return None


def load_fall_windows_from_annotated_dir(
    annotated_root: Path | str,
    *,
    fall_codes: tuple[str, ...] = FALL_CODES,
    window_samples: int = WINDOW_SAMPLES,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Returns dict with keys:
      acc_windows (N, 300, 3), gyro_windows, ori_windows, y_labels (str array),
      label_encoder (fitted on fall codes)
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "pandas is required for fall CSV loading: pip install pandas"
        ) from e

    root = Path(annotated_root)
    if not root.is_dir():
        raise FileNotFoundError(str(root))

    acc_list: list[np.ndarray] = []
    gyro_list: list[np.ndarray] = []
    ori_list: list[np.ndarray] = []
    y_labels: list[str] = []

    for fall_code in fall_codes:
        fall_files: list[Path] = []
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.startswith(f"{fall_code}_") and fn.endswith("_annotated.csv"):
                    fall_files.append(Path(dirpath) / fn)

        iterator = tqdm(fall_files, desc=f"{fall_code}", leave=False) if show_progress else fall_files

        for file_path in iterator:
            try:
                df = pd.read_csv(file_path)

                acc_cols = _pick_columns(df, [["acc_x", "acc_y", "acc_z"], ["x", "y", "z"]])
                if acc_cols is None:
                    continue

                gyro_cols = _pick_columns(df, [["gyro_x", "gyro_y", "gyro_z"]])
                ori_cols = _pick_columns(df, [["azimuth", "pitch", "roll"], ["Azimuth", "Pitch", "Roll"]])

                acc_data = df[acc_cols[:3]].values.astype(np.float64)

                gyro_data = df[gyro_cols[:3]].values.astype(np.float64) if gyro_cols else None
                ori_data = df[ori_cols[:3]].values.astype(np.float64) if ori_cols else None

                magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
                impact_idx = int(np.argmax(magnitude))

                half = window_samples // 2
                start = max(0, impact_idx - half)
                end = min(len(acc_data), impact_idx + half)

                if end - start < window_samples:
                    continue

                acc_window = acc_data[start:end]
                idx = np.linspace(0, len(acc_window) - 1, window_samples, dtype=int)
                if len(acc_window) != window_samples:
                    acc_window = acc_window[idx]

                acc_list.append(acc_window)

                if gyro_data is not None and len(gyro_data) >= end:
                    gw = gyro_data[start:end]
                    if len(gw) != window_samples:
                        gw = gw[idx]
                    gyro_list.append(gw)
                else:
                    gyro_list.append(np.zeros((window_samples, 3), dtype=np.float64))

                if ori_data is not None and len(ori_data) >= end:
                    ow = ori_data[start:end]
                    if len(ow) != window_samples:
                        ow = ow[idx]
                    ori_list.append(ow)
                else:
                    ori_list.append(np.zeros((window_samples, 3), dtype=np.float64))

                y_labels.append(fall_code)
            except Exception:
                continue

    if not acc_list:
        raise RuntimeError(f"No fall windows found under {root}")

    acc_windows = np.stack(acc_list, axis=0)
    gyro_windows = np.stack(gyro_list, axis=0)
    ori_windows = np.stack(ori_list, axis=0)
    y_str = np.asarray(y_labels, dtype=object)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)

    return {
        "acc_windows": acc_windows,
        "gyro_windows": gyro_windows,
        "ori_windows": ori_windows,
        "y_labels": y_str,
        "y_encoded": y_enc,
        "label_encoder": le,
    }
