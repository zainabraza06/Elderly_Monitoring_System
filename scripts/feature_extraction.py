from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import signal, stats
import pywt


def load_windows(windows_path: Path) -> pd.DataFrame:
    """Load windows from either pickle, parquet+npz split format, or parquet."""
    if windows_path.suffix.lower() == ".pkl" and windows_path.exists():
        try:
            return pd.read_pickle(windows_path)
        except Exception as e:
            print(f"  Could not read pickle: {e}")
    
    # Try parquet + signals.npz format
    parquet_path = windows_path.with_suffix(".parquet")
    signals_path = windows_path.with_stem(windows_path.stem + "_signals").with_suffix(".npz")
    
    if parquet_path.exists() and signals_path.exists():
        print(f"  Loading split format: metadata from {parquet_path}, signals from {signals_path}")
        df = pd.read_parquet(parquet_path)
        signals_data = np.load(signals_path, allow_pickle=True)
        signals = signals_data['signals']
        df['signal'] = list(signals)
        return df
    elif parquet_path.exists():
        print(f"  Loading from {parquet_path}")
        return pd.read_parquet(parquet_path)
    
    raise FileNotFoundError(f"Could not find windows file: {windows_path} or alternatives")



def spectral_entropy(power: np.ndarray, eps: float = 1e-12) -> float:
    p = power / (np.sum(power) + eps)
    return float(-(p * np.log2(p + eps)).sum())


def calculate_entropy_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """Calculate different types of entropy."""
    try:
        # Approximate Entropy
        # U is a list of vectors of size m
        def _phi(m, data, r):
            U = [data[i:i+m] for i in range(len(data) - m + 1)]
            C = np.zeros(len(U))
            for i in range(len(U)):
                # Chebychev distance
                dist = np.max(np.abs(U[i] - np.array(U)), axis=1)
                C[i] = np.sum(dist <= r) / len(U)
            return np.sum(np.log(C)) / len(U)

        r = 0.2 * np.std(x)
        apen = _phi(2, x, r) - _phi(3, x, r)
    except Exception:
        apen = 0.0

    try:
        # Sample Entropy
        def _phi_samp(m, data, r):
            U = [data[i:i+m] for i in range(len(data) - m + 1)]
            C = np.zeros(len(U))
            for i in range(len(U)):
                dist = np.max(np.abs(U[i] - np.array(U)), axis=1)
                # Exclude self-matching
                C[i] = (np.sum(dist <= r) - 1) / (len(U) - 1)
            return np.sum(C) / len(U)

        sampen = -np.log(_phi_samp(3, x, r) / _phi_samp(2, x, r))
    except Exception:
        sampen = 0.0

    # Wavelet Entropy
    try:
        coeffs = pywt.wavedec(x, 'db4', level=4)
        energy = [np.sum(c**2) for c in coeffs]
        total_energy = np.sum(energy)
        p = energy / total_energy
        wavelet_entropy = spectral_entropy(p)
    except Exception:
        wavelet_entropy = 0.0

    return {
        f"{prefix}_approximate_entropy": float(apen),
        f"{prefix}_sample_entropy": float(sampen),
        f"{prefix}_wavelet_entropy": float(wavelet_entropy),
    }


def band_energy(freq: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freq >= low) & (freq < high)
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(power[mask], freq[mask]))


def autocorr_peak(x: np.ndarray) -> float:
    x = x - np.mean(x)
    if np.std(x) < 1e-10:
        return 0.0
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    corr = corr / (corr[0] + 1e-12)
    if corr.size < 3:
        return 0.0
    return float(np.max(corr[1:]))


def step_stats(acc_mag: np.ndarray, fs: int) -> Dict[str, float]:
    min_dist = max(1, int(0.3 * fs))
    peaks, _ = signal.find_peaks(acc_mag, distance=min_dist)
    if peaks.size < 2:
        return {
            "step_count": float(peaks.size),
            "step_interval_mean": 0.0,
            "step_interval_std": 0.0,
            "step_interval_cv": 0.0,
        }

    intervals = np.diff(peaks) / float(fs)
    m = float(np.mean(intervals))
    s = float(np.std(intervals))
    return {
        "step_count": float(peaks.size),
        "step_interval_mean": m,
        "step_interval_std": s,
        "step_interval_cv": s / (m + 1e-12),
    }


def channel_time_features(x: np.ndarray, prefix: str) -> Dict[str, float]:
    dx = np.diff(x, prepend=x[0])
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_var": float(np.var(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_skew": float(stats.skew(x, bias=False) if np.std(x) > 1e-12 else 0.0),
        f"{prefix}_kurtosis": float(stats.kurtosis(x, bias=False) if np.std(x) > 1e-12 else 0.0),
        f"{prefix}_zcr": float(np.mean(np.abs(np.diff(np.signbit(x).astype(int))))),
        f"{prefix}_jerk_rms": float(np.sqrt(np.mean(dx ** 2))),
    }


def channel_freq_features(x: np.ndarray, fs: int, prefix: str) -> Dict[str, float]:
    f, pxx = signal.welch(x, fs=fs, nperseg=min(128, len(x)))
    if pxx.size == 0:
        return {
            f"{prefix}_spec_energy": 0.0,
            f"{prefix}_spec_entropy": 0.0,
            f"{prefix}_dom_freq": 0.0,
            f"{prefix}_e_0_3": 0.0,
            f"{prefix}_e_3_6": 0.0,
            f"{prefix}_e_6_10": 0.0,
            f"{prefix}_e_10_20": 0.0,
        }

    dom_idx = int(np.argmax(pxx))
    return {
        f"{prefix}_spec_energy": float(np.trapezoid(pxx, f)),
        f"{prefix}_spec_entropy": spectral_entropy(pxx),
        f"{prefix}_dom_freq": float(f[dom_idx]),
        f"{prefix}_e_0_3": band_energy(f, pxx, 0.0, 3.0),
        f"{prefix}_e_3_6": band_energy(f, pxx, 3.0, 6.0),
        f"{prefix}_e_6_10": band_energy(f, pxx, 6.0, 10.0),
        f"{prefix}_e_10_20": band_energy(f, pxx, 10.0, 20.0),
    }


def harmonic_ratio(acc_mag: np.ndarray, fs: int) -> float:
    f, pxx = signal.welch(acc_mag, fs=fs, nperseg=min(128, len(acc_mag)))
    if pxx.size == 0:
        return 0.0
    valid = (f >= 0.5) & (f <= 3.0)
    if not np.any(valid):
        return 0.0
    base_f = f[valid][int(np.argmax(pxx[valid]))]
    if base_f <= 0.0:
        return 0.0

    even = 0.0
    odd = 0.0
    for k in range(1, 7):
        target = k * base_f
        idx = int(np.argmin(np.abs(f - target)))
        if k % 2 == 0:
            even += pxx[idx]
        else:
            odd += pxx[idx]
    return float(even / (odd + 1e-12))


def build_single_feature_row(raw_signal: np.ndarray, fs: int) -> Dict[str, float]:
    # Ensure signal is a numeric numpy array
    if not isinstance(raw_signal, np.ndarray) or raw_signal.dtype == np.object_:
        raw_signal = np.asarray(raw_signal, dtype=np.float64)

    row: Dict[str, float] = {}
    names = ["acc_x", "acc_y", "acc_z", "acc_mag", "gyro_x", "gyro_y", "gyro_z", "gyro_mag"]

    for i, name in enumerate(names):
        x = raw_signal[:, i]
        row.update(channel_time_features(x, name))
        row.update(channel_freq_features(x, fs, name))
        row.update(calculate_entropy_features(x, name))

    acc_mag = raw_signal[:, 3]
    gyro_mag = raw_signal[:, 7]

    row.update(step_stats(acc_mag, fs))
    row["acc_autocorr_peak"] = autocorr_peak(acc_mag)
    row["gyro_autocorr_peak"] = autocorr_peak(gyro_mag)
    row["harmonic_ratio"] = harmonic_ratio(acc_mag, fs)

    f_g, p_g = signal.welch(gyro_mag, fs=fs, nperseg=min(128, len(gyro_mag)))
    tremor_energy = band_energy(f_g, p_g, 4.0, 6.0)
    low_move_energy = band_energy(f_g, p_g, 0.2, 2.0)
    total_g = float(np.trapezoid(p_g, f_g)) + 1e-12

    row["tremor_4_6_energy"] = tremor_energy
    row["bradykinesia_proxy"] = 1.0 - (low_move_energy / total_g)

    x_e = row.get("acc_x_spec_energy", 0.0)
    y_e = row.get("acc_y_spec_energy", 0.0)
    row["asymmetry_proxy"] = abs(x_e - y_e) / (x_e + y_e + 1e-12)

    variability = row.get("step_interval_cv", 0.0)
    jerk_cost = row.get("acc_mag_jerk_rms", 0.0)
    amp_reduction = 1.0 / (row.get("acc_mag_rms", 0.0) + 1e-6)

    row["frailty_proxy_raw"] = 0.45 * variability + 0.35 * jerk_cost + 0.20 * amp_reduction
    row["gait_stability_proxy_raw"] = (
        0.40 * row.get("acc_autocorr_peak", 0.0)
        + 0.35 * row.get("harmonic_ratio", 0.0)
        - 0.25 * variability
    )
    row["movement_disorder_proxy_raw"] = (
        0.5 * tremor_energy + 0.3 * row["asymmetry_proxy"] + 0.2 * row["bradykinesia_proxy"]
    )

    return row


def minmax01(x: np.ndarray) -> np.ndarray:
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def build_feature_table(windows_df: pd.DataFrame, fs: int) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for _, r in windows_df.iterrows():
        feats = build_single_feature_row(r["signal"], fs)
        feats["dataset"] = r["dataset"]
        feats["subject"] = r["subject"]
        feats["age_group"] = r["age_group"]
        feats["activity_code"] = r["activity_code"]
        feats["fall_label"] = int(r["fall_label"])
        feats["met_class"] = r["met_class"]
        rows.append(feats)

    out = pd.DataFrame(rows)

    out["frailty_proxy"] = minmax01(out["frailty_proxy_raw"].to_numpy())
    g_raw = out["gait_stability_proxy_raw"].to_numpy()
    out["gait_stability_proxy"] = minmax01(g_raw)
    out["movement_disorder_proxy"] = minmax01(out["movement_disorder_proxy_raw"].to_numpy())

    out = out.drop(columns=["frailty_proxy_raw", "gait_stability_proxy_raw", "movement_disorder_proxy_raw"])
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract engineered features from preprocessed windows.")
    p.add_argument("--windows", type=Path, required=True, help="Input windows file (.pkl or .parquet)")
    p.add_argument("--output", type=Path, required=True, help="Output features file (.parquet or .csv)")
    p.add_argument("--target-fs", type=int, default=50)
    return p.parse_args()


def save_features(features_df: pd.DataFrame, output_path: Path) -> None:
    """Save features, trying Parquet first and falling back to CSV."""
    # Try Parquet first
    parquet_path = output_path.with_suffix(".parquet")
    try:
        features_df.to_parquet(parquet_path, index=False)
        print(f"  Saved features to {parquet_path}")
        # If successful, remove old CSV if it exists
        csv_path = output_path.with_suffix(".csv")
        if csv_path.exists():
            csv_path.unlink()
        return
    except Exception as e:
        print(f"  Could not save to Parquet: {e}. Falling back to CSV.")

    # Fallback to CSV
    csv_path = output_path.with_suffix(".csv")
    features_df.to_csv(csv_path, index=False)
    print(f"  Saved features to {csv_path}")


def main() -> None:
    args = parse_args()
    print(f"Loading windows from {args.windows}")
    windows_df = load_windows(args.windows)
    if windows_df.empty:
        print("Input windows are empty.")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Use Parquet with batching to save memory
    output_format = "parquet"
    
    # Process in batches to avoid memory overflow
    batch_size = 10000
    n_batches = (len(windows_df) + batch_size - 1) // batch_size
    
    parquet_output = args.output.with_suffix(".parquet")
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(windows_df))
        batch = windows_df.iloc[start:end].reset_index(drop=True)
        
        print(f"Processing batch {batch_idx + 1}/{n_batches} (rows {start}–{end})")
        features_batch = build_feature_table(batch, fs=args.target_fs)
        
        try:
            if batch_idx == 0:
                # Write first batch
                features_batch.to_parquet(parquet_output, index=False)
            else:
                # Append subsequent batches
                existing = pd.read_parquet(parquet_output)
                combined = pd.concat([existing, features_batch], ignore_index=True)
                combined.to_parquet(parquet_output, index=False)
        except Exception as e:
            print(f"  Parquet write failed: {e}, falling back to CSV append")
            csv_output = parquet_output.with_suffix(".csv")
            if batch_idx == 0:
                features_batch.to_csv(csv_output, index=False)
            else:
                features_batch.to_csv(csv_output, mode='a', header=False, index=False)
            output_format = "csv"
    
    # Verify output was written
    if output_format == "parquet" and parquet_output.exists():
        print(f"✓ Saved features to {parquet_output}")
        verify_df = pd.read_parquet(parquet_output)
        print(f"  Shape: {verify_df.shape}")
        print(verify_df[["dataset", "fall_label", "met_class"]].head())
    elif output_format == "csv":
        csv_output = parquet_output.with_suffix(".csv")
        print(f"✓ Saved features to {csv_output}")
        verify_df = pd.read_csv(csv_output)
        print(f"  Shape: {verify_df.shape}")
        print(verify_df[["dataset", "fall_label", "met_class"]].head())


if __name__ == "__main__":
    main()
