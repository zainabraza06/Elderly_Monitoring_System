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


def samples_to_feature_vector(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (enhanced_116,) float vector, acc (300,3), gyro (300,3) for fall-type path.
    """
    if not samples:
        raise ValueError("empty samples")
    n = len(samples)
    acc = np.zeros((n, 3), dtype=np.float64)
    gyro = np.zeros((n, 3), dtype=np.float64)
    for i, s in enumerate(samples):
        acc[i, 0] = float(s.get("acc_x", 0.0))
        acc[i, 1] = float(s.get("acc_y", 0.0))
        acc[i, 2] = float(s.get("acc_z", 0.0))
        gyro[i, 0] = float(s.get("gyro_x", 0.0))
        gyro[i, 1] = float(s.get("gyro_y", 0.0))
        gyro[i, 2] = float(s.get("gyro_z", 0.0))

    acc_300 = _resample_rows(acc, _WINDOW)
    gyro_300 = _resample_rows(gyro, _WINDOW)
    ori_300 = np.zeros((_WINDOW, 3), dtype=np.float64)

    _ensure_scripts()
    from baseline_fall.enhanced_features import extract_enhanced_features

    xb = acc_300[np.newaxis, ...]
    yb = gyro_300[np.newaxis, ...]
    zb = ori_300[np.newaxis, ...]
    feat = extract_enhanced_features(xb, yb, zb)
    return feat[0], acc_300, gyro_300


def acc_gyro_to_window_lists(acc: np.ndarray, gyro: np.ndarray) -> tuple[list[list[float]], list[list[float]]]:
    acc_l = acc.tolist()
    gyro_l = gyro.tolist()
    return acc_l, gyro_l
