"""Convert ingest samples → 116-D features + 300×3 windows (SisFall / MobiAct training parity)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from flask_backend.app.settings import repo_root

_WINDOW = 300


def _ensure_scripts() -> None:
    s = repo_root() / "scripts"
    p = str(s)
    if p not in sys.path:
        sys.path.insert(0, p)


def _resample_rows(data: np.ndarray, target_len: int) -> np.ndarray:
    """data: (n, 3) -> (target_len, 3)"""
    n = data.shape[0]
    if n == target_len:
        return data
    if n < 2:
        return np.zeros((target_len, 3), dtype=np.float64)
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = np.interp(x_new, x_old, data[:, j])
    return out


def _sample_ori_degrees(s: dict[str, Any]) -> tuple[float, float, float]:
    """MobiAct ori columns: azimuth (z), pitch (x), roll (y) in degrees — optional per axis."""

    def g(key: str) -> float:
        v = s.get(key)
        if v is None:
            return 0.0
        return float(v)

    return g("azimuth"), g("pitch"), g("roll")


def samples_to_feature_vector(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns enhanced 116-D vector, acc (300,3), gyro (300,3), ori (300,3) for fall-type path.
    Orientation defaults to zeros when [azimuth, pitch, roll] are absent (degrees).
    """
    if not samples:
        raise ValueError("empty samples")
    n = len(samples)
    acc = np.zeros((n, 3), dtype=np.float64)
    gyro = np.zeros((n, 3), dtype=np.float64)
    ori = np.zeros((n, 3), dtype=np.float64)
    for i, s in enumerate(samples):
        acc[i, 0] = float(s.get("acc_x", 0.0))
        acc[i, 1] = float(s.get("acc_y", 0.0))
        acc[i, 2] = float(s.get("acc_z", 0.0))
        gyro[i, 0] = float(s.get("gyro_x", 0.0))
        gyro[i, 1] = float(s.get("gyro_y", 0.0))
        gyro[i, 2] = float(s.get("gyro_z", 0.0))
        az, pit, rol = _sample_ori_degrees(s)
        ori[i, 0] = az
        ori[i, 1] = pit
        ori[i, 2] = rol

    acc_300 = _resample_rows(acc, _WINDOW)
    gyro_300 = _resample_rows(gyro, _WINDOW)
    ori_300 = _resample_rows(ori, _WINDOW)

    _ensure_scripts()
    from baseline_fall.enhanced_features import extract_enhanced_features

    xb = acc_300[np.newaxis, ...]
    yb = gyro_300[np.newaxis, ...]
    zb = ori_300[np.newaxis, ...]
    feat = extract_enhanced_features(xb, yb, zb)
    return feat[0], acc_300, gyro_300, ori_300


def acc_gyro_ori_to_window_lists(
    acc: np.ndarray, gyro: np.ndarray, ori: np.ndarray
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    return acc.tolist(), gyro.tolist(), ori.tolist()
