"""Complete multi-sensor fall feature extractor (ACC, gyro, orientation)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm

from .config import MULTISENSOR_FEATURE_DIM, SAMPLING_RATE_HZ


class CompleteFallFeatureExtractor:
    """Extract comprehensive features from ACC + gyro + orientation windows."""

    def __init__(self, fs: float = SAMPLING_RATE_HZ):
        self.fs = fs

    def butter_lowpass_filter(self, data: np.ndarray, cutoff: float = 10.0, order: int = 4) -> np.ndarray:
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff / nyquist
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            return np.asarray(data, dtype=np.float64).flatten()
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return filtfilt(b, a, data)

    def extract_time_features(self, signal: np.ndarray) -> list[float]:
        if len(signal) == 0:
            return [0.0] * 30

        signal = np.array(signal, dtype=np.float64).flatten()

        features: list[float] = [
            float(np.mean(signal)),
            float(np.std(signal)),
            float(np.median(signal)),
            float(np.min(signal)),
            float(np.max(signal)),
            float(np.ptp(signal)),
            float(np.sqrt(np.mean(signal**2))),
            float(np.mean(np.abs(signal))),
        ]

        for p in (10, 25, 75, 90, 95, 99):
            features.append(float(np.percentile(signal, p)))

        if len(signal) > 3:
            features.extend([float(skew(signal)), float(kurtosis(signal))])
        else:
            features.extend([0.0, 0.0])

        if len(signal) > 1:
            diff_signal = np.diff(signal)
            features.extend(
                [
                    float(np.mean(np.abs(diff_signal))),
                    float(np.max(np.abs(diff_signal))),
                    float(np.std(diff_signal)),
                    float(np.sum(np.abs(diff_signal))),
                    float(np.sum(diff_signal**2) / len(signal)),
                    float(np.percentile(np.abs(diff_signal), 95)),
                ]
            )
        else:
            features.extend([0.0] * 6)

        if len(signal) > 1:
            mean_crossings = np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            features.extend(
                [float(mean_crossings / len(signal)), float(zero_crossings / len(signal))]
            )
        else:
            features.extend([0.0, 0.0])

        features.extend(
            [
                float(np.sum(signal**2)),
                float(np.sum(signal**2) / len(signal)),
                float(np.max(np.abs(signal))),
            ]
        )

        hist, _ = np.histogram(signal, bins=20, density=True)
        hist = hist[hist > 0]
        if len(hist) > 0:
            features.append(float(entropy(hist)))
        else:
            features.append(0.0)

        return features[:30]

    def extract_frequency_features(self, signal: np.ndarray) -> list[float]:
        if len(signal) < 10:
            return [0.0] * 12

        signal = np.array(signal, dtype=np.float64).flatten()
        nperseg = min(128, max(16, len(signal) // 4))
        freqs, psd = welch(signal, fs=self.fs, nperseg=nperseg)

        features: list[float] = []
        bands = [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 25)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs < high)
            band_power = float(np.sum(psd[band_mask])) if np.any(band_mask) else 0.0
            features.append(band_power)

        total_power = float(np.sum(psd))
        features.append(total_power)

        if total_power > 0:
            centroid = float(np.sum(freqs * psd) / total_power)
            features.append(centroid)
            spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power))
            features.append(spread)
            psd_norm = psd / total_power
            spec_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-6)))
            features.append(spec_entropy)
        else:
            features.extend([0.0, 0.0, 0.0])

        if len(psd) > 0:
            dominant_idx = int(np.argmax(psd))
            features.append(float(freqs[dominant_idx]))
            if len(psd) > 1:
                sorted_psd = np.sort(psd)
                peak_ratio = float(psd[dominant_idx] / (sorted_psd[-2] + 1e-6))
                features.append(peak_ratio)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])

        return features[:12]

    def extract_fall_impact_features(
        self,
        acc_signal: np.ndarray,
        gyro_signal: np.ndarray | None = None,
    ) -> list[float]:
        if len(acc_signal.shape) > 1 and acc_signal.shape[1] >= 3:
            magnitude = np.sqrt(np.sum(acc_signal**2, axis=1))
        else:
            magnitude = np.abs(acc_signal.flatten())

        features: list[float] = []

        if len(magnitude) > 20:
            impact_idx = int(np.argmax(magnitude))
            features.append(float(impact_idx / len(magnitude)))

            pre_start = max(0, impact_idx - int(self.fs))
            pre_end = max(0, impact_idx - int(self.fs * 0.2))
            pre_impact_mean = 0.0
            if pre_end > pre_start:
                pre_impact = magnitude[pre_start:pre_end]
                pre_impact_mean = float(np.mean(pre_impact))
                features.extend(
                    [
                        pre_impact_mean,
                        float(np.std(pre_impact)),
                        float(np.max(pre_impact)),
                        float(np.min(pre_impact)),
                        float(np.percentile(pre_impact, 95)) if len(pre_impact) > 0 else 0.0,
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            impact_start = max(0, impact_idx - int(self.fs * 0.1))
            impact_end = min(len(magnitude), impact_idx + int(self.fs * 0.1))
            impact_region = magnitude[impact_start:impact_end]
            if len(impact_region) > 0:
                features.extend(
                    [
                        float(np.max(impact_region)),
                        float(np.std(impact_region)),
                        float(np.mean(impact_region)),
                        float(np.argmax(impact_region) / len(impact_region)),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])

            post_start = min(len(magnitude), impact_idx + int(self.fs * 0.2))
            post_end = min(len(magnitude), impact_idx + int(self.fs * 2))
            if post_end > post_start:
                post_impact = magnitude[post_start:post_end]
                features.extend(
                    [
                        float(np.mean(post_impact)),
                        float(np.std(post_impact)),
                        float(np.max(post_impact)),
                        float(np.min(post_impact)),
                        float(len(post_impact) / len(magnitude)),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0])

            mi = float(magnitude[impact_idx])
            features.append(mi)
            pre_mean_for_delta = pre_impact_mean if pre_end > pre_start else 0.0
            features.append(mi - pre_mean_for_delta)
            features.append(mi / (pre_mean_for_delta + 1e-6))

            peaks, props = find_peaks(magnitude, height=np.std(magnitude), distance=int(self.fs * 0.1))
            features.append(float(len(peaks)))
            if len(peaks) > 0:
                features.append(float(np.mean(props["peak_heights"])))
                features.append(float(np.std(props["peak_heights"])) if len(peaks) > 1 else 0.0)
                features.append(float(peaks[0] / len(magnitude)))
            else:
                features.extend([0.0, 0.0, 0.0])
        else:
            features.extend([0.0] * 30)

        if gyro_signal is not None and len(gyro_signal) > 0:
            if len(gyro_signal.shape) > 1 and gyro_signal.shape[1] >= 3:
                angular_mag = np.sqrt(np.sum(gyro_signal**2, axis=1))
                features.append(float(np.max(angular_mag)))
                features.append(float(np.mean(angular_mag)))
                features.append(float(np.std(angular_mag)))
                features.append(float(np.argmax(angular_mag) / len(angular_mag)))
                for axis in range(3):
                    features.append(float(np.max(np.abs(gyro_signal[:, axis]))))
                    features.append(float(np.mean(np.abs(gyro_signal[:, axis]))))
                angular_change = np.sum(np.abs(gyro_signal), axis=0) / self.fs
                features.extend([float(x) for x in angular_change])
            else:
                features.extend([0.0] * 13)
        else:
            features.extend([0.0] * 13)

        return features[:50]

    def extract_orientation_features(self, ori_signal: np.ndarray | None) -> list[float]:
        if ori_signal is None or len(ori_signal) == 0:
            return [0.0] * 15

        features: list[float] = []
        for axis in range(min(3, ori_signal.shape[1])):
            signal = ori_signal[:, axis]
            features.extend(
                [
                    float(np.mean(signal)),
                    float(np.std(signal)),
                    float(np.median(signal)),
                    float(np.min(signal)),
                    float(np.max(signal)),
                    float(np.ptp(signal)),
                ]
            )
            if len(signal) > 1:
                diff_signal = np.diff(signal)
                features.extend(
                    [
                        float(np.mean(np.abs(diff_signal))),
                        float(np.max(np.abs(diff_signal))),
                        float(np.sum(np.abs(diff_signal))),
                    ]
                )
            else:
                features.extend([0.0, 0.0, 0.0])

        if len(ori_signal) > 0:
            features.extend([float(x) for x in ori_signal[-1]])
        else:
            features.extend([0.0, 0.0, 0.0])

        return features[:15]

    def extract_cross_sensor_features(
        self,
        acc_signal: np.ndarray,
        gyro_signal: np.ndarray | None = None,
        ori_signal: np.ndarray | None = None,
    ) -> list[float]:
        features: list[float] = []

        if len(acc_signal.shape) > 1 and acc_signal.shape[1] >= 3:
            for i, j in ((0, 1), (0, 2), (1, 2)):
                corr = np.corrcoef(acc_signal[:, i], acc_signal[:, j])[0, 1]
                features.append(float(corr) if not np.isnan(corr) else 0.0)

        if gyro_signal is not None and len(gyro_signal) > 0:
            acc_mag = np.sqrt(np.sum(acc_signal**2, axis=1))
            gyro_mag = np.sqrt(np.sum(gyro_signal**2, axis=1))
            if len(acc_mag) == len(gyro_mag) and len(acc_mag) > 0:
                corr = np.corrcoef(acc_mag, gyro_mag)[0, 1]
                features.append(float(corr) if not np.isnan(corr) else 0.0)
                acc_peak = int(np.argmax(acc_mag))
                gyro_peak = int(np.argmax(gyro_mag))
                features.append(float((gyro_peak - acc_peak) / len(acc_mag)))
            else:
                features.extend([0.0, 0.0])

        if (
            ori_signal is not None
            and len(ori_signal) > 0
            and len(acc_signal) > 0
            and len(ori_signal.shape) > 1
            and ori_signal.shape[1] >= 3
        ):
            magnitude = np.sqrt(np.sum(acc_signal**2, axis=1))
            impact_idx = int(np.argmax(magnitude))
            if impact_idx < len(ori_signal):
                pre_ori = ori_signal[max(0, impact_idx - int(self.fs))]
                post_ori = ori_signal[min(len(ori_signal) - 1, impact_idx + int(self.fs))]
                orientation_change = np.abs(post_ori - pre_ori)
                features.extend([float(x) for x in orientation_change])

        while len(features) < 15:
            features.append(0.0)
        return features[:15]

    def extract_signal_magnitude_area(self, acc_signal: np.ndarray) -> float:
        if len(acc_signal.shape) > 1 and acc_signal.shape[1] >= 3:
            sma = float(np.mean(np.sum(np.abs(acc_signal), axis=1)))
            return sma
        return 0.0

    def extract_window_features(
        self,
        acc_window: np.ndarray,
        gyro_window: np.ndarray | None = None,
        ori_window: np.ndarray | None = None,
    ) -> np.ndarray:
        filtered_acc = np.zeros_like(acc_window, dtype=np.float64)
        for axis in range(acc_window.shape[1]):
            filtered_acc[:, axis] = self.butter_lowpass_filter(acc_window[:, axis])

        all_features: list[float] = []

        for axis in range(acc_window.shape[1]):
            signal = filtered_acc[:, axis]
            all_features.extend(self.extract_time_features(signal))
            all_features.extend(self.extract_frequency_features(signal))

        if gyro_window is not None and len(gyro_window) > 0:
            for axis in range(gyro_window.shape[1]):
                signal = gyro_window[:, axis]
                all_features.extend(self.extract_time_features(signal)[:20])
                all_features.extend(self.extract_frequency_features(signal)[:8])
        else:
            all_features.extend([0.0] * 84)

        if ori_window is not None and len(ori_window) > 0:
            all_features.extend(self.extract_orientation_features(ori_window))
        else:
            all_features.extend([0.0] * 15)

        all_features.extend(self.extract_fall_impact_features(filtered_acc, gyro_window))
        all_features.extend(self.extract_cross_sensor_features(filtered_acc, gyro_window, ori_window))
        all_features.append(self.extract_signal_magnitude_area(filtered_acc))

        out = np.zeros(MULTISENSOR_FEATURE_DIM, dtype=np.float64)
        arr = np.array(all_features[:MULTISENSOR_FEATURE_DIM], dtype=np.float64)
        out[: len(arr)] = arr
        return out

    def extract_batch(
        self,
        acc_windows: np.ndarray,
        gyro_windows: np.ndarray | None = None,
        ori_windows: np.ndarray | None = None,
        desc: str = "Extracting features",
    ) -> np.ndarray:
        n = len(acc_windows)
        if gyro_windows is None:
            gyro_iter: list[np.ndarray | None] = [None] * n
        else:
            gyro_iter = [gyro_windows[i] for i in range(n)]
        if ori_windows is None:
            ori_iter: list[np.ndarray | None] = [None] * n
        else:
            ori_iter = [ori_windows[i] for i in range(n)]

        features: list[np.ndarray] = []
        for i in tqdm(range(n), desc=desc):
            features.append(
                self.extract_window_features(acc_windows[i], gyro_iter[i], ori_iter[i])
            )

        return np.array(features)
