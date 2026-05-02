"""Locate MobiAct annotated data, optional RAR extraction, multi-sensor fall windows."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .config import FALL_CODES, WINDOW_SAMPLES


def find_annotated_data_dir(root: str | Path) -> Path | None:
    """Return path to 'Annotated Data' under root (walk if needed)."""
    root = Path(root)
    candidate = root / "MobiAct_Dataset_v2.0" / "Annotated Data"
    if candidate.is_dir():
        return candidate
    for dirpath, dirnames, _ in os.walk(root):
        if "Annotated Data" in dirnames:
            return Path(dirpath) / "Annotated Data"
    return None


def extract_rar(rar_path: str | Path, extract_to: str | Path) -> bool:
    """Extract RAR using unrar or 7z; return True on success."""
    rar_path = Path(rar_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    if shutil.which("unrar"):
        try:
            subprocess.run(
                ["unrar", "x", "-o+", str(rar_path), str(extract_to)],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            pass

    if shutil.which("7z"):
        try:
            subprocess.run(
                ["7z", "x", str(rar_path), f"-o{extract_to}", "-y"],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            pass

    try:
        import patoolib  # type: ignore

        patoolib.extract_archive(str(rar_path), outdir=str(extract_to))
        return True
    except Exception:
        pass

    print(
        "RAR extraction failed. Install unrar or 7-Zip, or: pip install patool",
        file=sys.stderr,
    )
    return False


def _norm_col(name: str) -> str:
    return re.sub(r"\s+", "", str(name).strip().lower())


def _find_triplet(df: pd.DataFrame, patterns: list[tuple[str, str, str]]) -> list[str] | None:
    by_norm = {_norm_col(c): c for c in df.columns}
    for a, b, c in patterns:
        ka, kb, kc = _norm_col(a), _norm_col(b), _norm_col(c)
        if ka in by_norm and kb in by_norm and kc in by_norm:
            return [by_norm[ka], by_norm[kb], by_norm[kc]]
    return None


def _find_acc_columns(df: pd.DataFrame) -> list[str] | None:
    acc = _find_triplet(
        df,
        [
            ("acc_x", "acc_y", "acc_z"),
            ("accelerometer_x", "accelerometer_y", "accelerometer_z"),
        ],
    )
    if acc:
        return acc
    candidates = []
    for c in df.columns:
        cl = _norm_col(c)
        if "acc" in cl and any(ax in cl for ax in ("x", "y", "z")):
            candidates.append(c)
    if len(candidates) >= 3:
        return candidates[:3]
    return None


def _find_gyro_columns(df: pd.DataFrame) -> list[str] | None:
    g = _find_triplet(df, [("gyro_x", "gyro_y", "gyro_z")])
    if g:
        return g
    candidates = []
    for c in df.columns:
        cl = _norm_col(c)
        if "gyro" in cl and any(ax in cl for ax in ("x", "y", "z")):
            candidates.append(c)
    if len(candidates) >= 3:
        return candidates[:3]
    return None


def _find_ori_columns(df: pd.DataFrame) -> list[str] | None:
    o = _find_triplet(
        df,
        [
            ("azimuth", "pitch", "roll"),
            ("Azimuth", "Pitch", "Roll"),
        ],
    )
    if o:
        return o
    candidates = []
    for name in ("azimuth", "pitch", "roll"):
        for c in df.columns:
            if _norm_col(c) == name:
                candidates.append(c)
                break
    if len(candidates) == 3:
        return candidates
    return None


def load_multisensor_fall_windows(
    mobiact_annotated_path: str | Path,
    fall_codes: tuple[str, ...] = FALL_CODES,
    window_size: int = WINDOW_SAMPLES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    One impact-centered window per annotated fall file (ACC, gyro, orientation).

    Returns acc_windows (n, W, 3), gyro_windows, ori_windows, y_encoded, LabelEncoder.
    Missing gyro/ori are filled with zeros for that window.
    """
    mobiact_annotated_path = Path(mobiact_annotated_path)
    if not mobiact_annotated_path.is_dir():
        raise FileNotFoundError(f"Not a directory: {mobiact_annotated_path}")

    acc_windows: list[np.ndarray] = []
    gyro_windows: list[np.ndarray] = []
    ori_windows: list[np.ndarray] = []
    y_labels: list[str] = []

    for code in fall_codes:
        fall_files: list[Path] = []
        for root, _, files in os.walk(mobiact_annotated_path):
            for file in files:
                if file.startswith(f"{code}_") and file.endswith("_annotated.csv"):
                    fall_files.append(Path(root) / file)

        for file_path in tqdm(fall_files, desc=f"Loading {code}", leave=False):
            try:
                df = pd.read_csv(file_path)
                acc_cols = _find_acc_columns(df)
                if acc_cols is None:
                    continue

                acc_data = df[acc_cols].values.astype(np.float64)
                gyro_cols = _find_gyro_columns(df)
                gyro_data = df[gyro_cols].values.astype(np.float64) if gyro_cols else None
                ori_cols = _find_ori_columns(df)
                ori_data = df[ori_cols].values.astype(np.float64) if ori_cols else None

                magnitude = np.sqrt(np.sum(acc_data**2, axis=1))
                impact_idx = int(np.argmax(magnitude))

                half = window_size // 2
                start = max(0, impact_idx - half)
                end = min(len(acc_data), impact_idx + half)

                if end - start < window_size:
                    continue

                acc_window = acc_data[start:end]
                if len(acc_window) != window_size:
                    indices = np.linspace(0, len(acc_window) - 1, window_size, dtype=int)
                    acc_window = acc_window[indices]
                else:
                    indices = np.arange(window_size, dtype=int)

                acc_windows.append(acc_window)

                if gyro_data is not None and len(gyro_data) >= end:
                    gyro_window = gyro_data[start:end]
                    if len(gyro_window) != window_size:
                        gyro_window = gyro_window[indices]
                    gyro_windows.append(gyro_window)
                else:
                    gyro_windows.append(np.zeros((window_size, 3), dtype=np.float64))

                if ori_data is not None and len(ori_data) >= end:
                    ori_window = ori_data[start:end]
                    if len(ori_window) != window_size:
                        ori_window = ori_window[indices]
                    ori_windows.append(ori_window)
                else:
                    ori_windows.append(np.zeros((window_size, 3), dtype=np.float64))

                y_labels.append(code)
            except Exception:
                continue

    if not acc_windows:
        raise RuntimeError(
            f"No multi-sensor fall windows found under {mobiact_annotated_path}."
        )

    acc_arr = np.array(acc_windows, dtype=np.float64)
    gyro_arr = np.array(gyro_windows, dtype=np.float64)
    ori_arr = np.array(ori_windows, dtype=np.float64)
    y_raw = np.array(y_labels)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    return acc_arr, gyro_arr, ori_arr, y_encoded, label_encoder
