"""Hand-crafted time/frequency features for acc (+ optional gyro/ori), matching the Colab baseline."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew
from tqdm import tqdm


def extract_enhanced_features(
    acc_windows: np.ndarray,
    gyro_windows: np.ndarray | None = None,
    ori_windows: np.ndarray | None = None,
) -> np.ndarray:
    n = len(acc_windows)
    if gyro_windows is None:
        gyro_windows = np.zeros((n, acc_windows.shape[1], 3), dtype=np.float64)
    if ori_windows is None:
        ori_windows = np.zeros((n, acc_windows.shape[1], 3), dtype=np.float64)

    features: list[list[float]] = []

    for idx in tqdm(range(n), desc="Extracting enhanced features"):
        window = acc_windows[idx]
        feat: list[float] = []

        for axis in range(3):
            data = window[:, axis]
            feat.extend(
                [
                    float(np.mean(data)),
                    float(np.std(data)),
                    float(np.median(data)),
                    float(np.min(data)),
                    float(np.max(data)),
                    float(np.ptp(data)),
                    float(np.percentile(data, 5)),
                    float(np.percentile(data, 25)),
                    float(np.percentile(data, 75)),
                    float(np.percentile(data, 95)),
                    float(np.sqrt(np.mean(data**2))),
                    float(np.mean(np.abs(np.diff(data)))),
                    float(np.sum(np.abs(np.diff(data)))),
                    float(skew(data)),
                    float(kurtosis(data)),
                    float(np.var(data)),
                    float(np.sum(data**2) / len(data)),
                    float(np.max(np.abs(data))),
                    float(np.argmax(np.abs(data)) / len(data)),
                ]
            )

            fft_vals = np.abs(fft(data))[: len(data) // 2]
            if len(fft_vals) > 0:
                s = float(np.sum(fft_vals) + 1e-6)
                feat.extend(
                    [
                        float(np.mean(fft_vals)),
                        float(np.std(fft_vals)),
                        float(np.max(fft_vals)),
                        float(s),
                        float(np.argmax(fft_vals) / len(fft_vals)),
                        float(np.sum(fft_vals[:10]) / s),
                    ]
                )
            else:
                feat.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            if len(data) > 1:
                zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
                feat.append(float(zero_crossings / len(data)))
            else:
                feat.append(0.0)

        gyro = gyro_windows[idx]
        if np.any(gyro != 0):
            for axis in range(3):
                data = gyro[:, axis]
                feat.extend(
                    [
                        float(np.mean(data)),
                        float(np.std(data)),
                        float(np.max(np.abs(data))),
                        float(np.sum(np.abs(data))),
                        float(np.sum(data**2) / len(data)),
                    ]
                )
        else:
            feat.extend([0.0] * 15)

        ori = ori_windows[idx]
        if np.any(ori != 0):
            for axis in range(3):
                data = ori[:, axis]
                feat.extend(
                    [
                        float(np.mean(data)),
                        float(np.std(data)),
                        float(data[-1] - data[0]),
                    ]
                )
        else:
            feat.extend([0.0] * 9)

        for i_idx, j_idx in [(0, 1), (0, 2), (1, 2)]:
            corr = np.corrcoef(window[:, i_idx], window[:, j_idx])[0, 1]
            feat.append(float(corr) if not np.isnan(corr) else 0.0)

        magnitude = np.sqrt(np.sum(window**2, axis=1))
        feat.extend(
            [
                float(np.mean(magnitude)),
                float(np.std(magnitude)),
                float(np.max(magnitude)),
                float(np.percentile(magnitude, 95)),
                float(np.argmax(magnitude) / len(magnitude)),
                float(np.sum(magnitude)),
                float(np.mean(np.diff(magnitude))),
            ]
        )

        if len(magnitude) > 10:
            peaks, _ = find_peaks(magnitude, height=np.std(magnitude), distance=5)
            feat.extend(
                [
                    float(len(peaks)),
                    float(np.max(magnitude[peaks])) if len(peaks) > 0 else 0.0,
                    float(np.mean(magnitude[peaks])) if len(peaks) > 0 else 0.0,
                    float(peaks[0] / len(magnitude)) if len(peaks) > 0 else 0.0,
                ]
            )
        else:
            feat.extend([0.0, 0.0, 0.0, 0.0])

        features.append(feat)

    return np.asarray(features, dtype=np.float64)
