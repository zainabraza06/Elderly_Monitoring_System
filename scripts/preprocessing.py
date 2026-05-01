from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import signal


SISFALL_FS = 200
UCI_FS = 50

SISFALL_FALL_CODES = {f"F{i:02d}" for i in range(1, 16)}
SISFALL_ADL_CODES = {f"D{i:02d}" for i in range(1, 20)}

MOBIACT_ADL_CODES = {
    "STD",
    "WAL",
    "JOG",
    "JUM",
    "STU",
    "STN",
    "SCH",
    "SIT",
    "CHU",
    "CSI",
    "CSO",
    "LYI",
}
MOBIACT_FALL_CODES = {"FOL", "FKL", "BSC", "SDL"}


@dataclass
class WindowSample:
    dataset: str
    subject: str
    age_group: str
    activity_code: str
    fall_label: int
    met_class: str
    signal: np.ndarray  # [window_len, 8] => acc xyz mag + gyro xyz mag


def lowpass_filter(data: np.ndarray, fs: int, cutoff_hz: float = 20.0) -> np.ndarray:
    nyq = fs * 0.5
    cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(4, cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def resample_array(data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    if orig_fs == target_fs:
        return data
    gcd = np.gcd(orig_fs, target_fs)
    up = target_fs // gcd
    down = orig_fs // gcd
    return signal.resample_poly(data, up=up, down=down, axis=0)


def make_windows(data: np.ndarray, window_size: int, step: int) -> List[np.ndarray]:
    if data.shape[0] < window_size:
        return []
    out = []
    for start in range(0, data.shape[0] - window_size + 1, step):
        out.append(data[start : start + window_size])
    return out


def robust_parse_sisfall_file(file_path: Path) -> np.ndarray:
    rows = []
    splitter = re.compile(r"[;,\s]+")
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p for p in splitter.split(line) if p]
            if len(parts) < 9:
                continue
            rows.append([float(v) for v in parts[:9]])
    if not rows:
        return np.empty((0, 9), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def sisfall_bits_to_units(raw9: np.ndarray) -> np.ndarray:
    adxl_scale = (2 * 16.0) / (2 ** 13)
    gyro_scale = (2 * 2000.0) / (2 ** 16)
    acc = raw9[:, 0:3] * adxl_scale
    gyro = raw9[:, 3:6] * gyro_scale
    return np.hstack([acc, gyro])


def add_magnitude(acc_gyro_6: np.ndarray) -> np.ndarray:
    acc = acc_gyro_6[:, 0:3]
    gyro = acc_gyro_6[:, 3:6]
    acc_mag = np.linalg.norm(acc, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    return np.hstack([acc, acc_mag, gyro, gyro_mag])


def sisfall_met_class(code: str) -> str:
    if code in {"D01", "D02", "D07", "D08", "D09", "D10", "D11", "D12", "D13", "D14"}:
        return "light"
    if code in {"D05", "D06", "D15", "D16", "D17", "D18"}:
        return "moderate"
    if code in {"D03", "D04", "D19"}:
        return "vigorous"
    return "unknown"


def mobiact_met_class(code: str) -> str:
    if code in {"STD", "SIT", "LYI", "SCH", "CHU", "CSI", "CSO"}:
        return "light"
    if code in {"WAL", "STU", "STN"}:
        return "moderate"
    if code in {"JOG", "JUM"}:
        return "vigorous"
    return "unknown"


def uci_met_class(activity_name: str) -> str:
    if activity_name in {"SITTING", "STANDING", "LAYING"}:
        return "light"
    if activity_name in {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}:
        return "moderate"
    return "unknown"


def load_sisfall_windows(
    sisfall_root: Path,
    target_fs: int,
    window_sec: float,
    overlap: float,
    max_files: Optional[int] = None,
) -> List[WindowSample]:
    window_size = int(target_fs * window_sec)
    step = max(1, int(window_size * (1.0 - overlap)))

    files = sorted(sisfall_root.glob("S*/**/*.txt"))
    if max_files is not None:
        files = files[: max_files]

    samples: List[WindowSample] = []
    for file_path in files:
        stem = file_path.stem
        parts = stem.split("_")
        if len(parts) != 3:
            continue
        code, subject, _trial = parts
        if code not in SISFALL_ADL_CODES and code not in SISFALL_FALL_CODES:
            continue

        raw9 = robust_parse_sisfall_file(file_path)
        if raw9.shape[0] < 10:
            continue

        sig6 = sisfall_bits_to_units(raw9)
        sig6 = lowpass_filter(sig6, SISFALL_FS, cutoff_hz=20.0)
        sig6 = resample_array(sig6, SISFALL_FS, target_fs)
        sig8 = add_magnitude(sig6)

        age_group = "elderly" if subject.startswith("SE") else "adult"
        fall_label = int(code.startswith("F"))
        met = "unknown" if fall_label else sisfall_met_class(code)

        for w in make_windows(sig8, window_size, step):
            samples.append(
                WindowSample(
                    dataset="sisfall",
                    subject=subject,
                    age_group=age_group,
                    activity_code=code,
                    fall_label=fall_label,
                    met_class=met,
                    signal=w.astype(np.float32),
                )
            )
    return samples


def parse_mobiact_csv(file_path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(file_path)
    except Exception:
        return None

    required = {"acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
    if not required.issubset(set(df.columns)):
        return None

    if "rel_time" in df.columns:
        t = pd.to_numeric(df["rel_time"], errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(t).any():
            t = np.arange(len(df), dtype=np.float64) * 0.02
    else:
        t = np.arange(len(df), dtype=np.float64) * 0.02

    out = pd.DataFrame(
        {
            "t": t,
            "acc_x": pd.to_numeric(df["acc_x"], errors="coerce"),
            "acc_y": pd.to_numeric(df["acc_y"], errors="coerce"),
            "acc_z": pd.to_numeric(df["acc_z"], errors="coerce"),
            "gyro_x": pd.to_numeric(df["gyro_x"], errors="coerce"),
            "gyro_y": pd.to_numeric(df["gyro_y"], errors="coerce"),
            "gyro_z": pd.to_numeric(df["gyro_z"], errors="coerce"),
        }
    ).dropna()
    if out.empty:
        return None
    return out


def load_mobiact_windows(
    mobiact_root: Path,
    target_fs: int,
    window_sec: float,
    overlap: float,
    max_files: Optional[int] = None,
) -> List[WindowSample]:
    window_size = int(target_fs * window_sec)
    step = max(1, int(window_size * (1.0 - overlap)))

    ann_root = mobiact_root / "Annotated Data"
    if not ann_root.exists():
        return []

    valid_codes = MOBIACT_ADL_CODES.union(MOBIACT_FALL_CODES)

    csv_files: List[Path] = []
    for code_dir in sorted(ann_root.iterdir()):
        if not code_dir.is_dir() or code_dir.name not in valid_codes:
            continue
        csv_files.extend(sorted(code_dir.glob("*_annotated.csv")))

    if max_files is not None:
        csv_files = csv_files[: max_files]

    samples: List[WindowSample] = []
    for file_path in csv_files:
        stem = file_path.stem.replace("_annotated", "")
        parts = stem.split("_")
        if len(parts) < 3:
            continue

        code = parts[0]
        subject = f"M{parts[1]}"
        df = parse_mobiact_csv(file_path)
        if df is None or len(df) < 20:
            continue

        sig = df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float64)
        t = df["t"].to_numpy(dtype=np.float64)

        if t[-1] <= t[0]:
            continue

        target_t = np.arange(t[0], t[-1], 1.0 / target_fs)
        if target_t.size < window_size:
            continue

        sig_res = np.vstack([np.interp(target_t, t, sig[:, i]) for i in range(sig.shape[1])]).T
        sig_res = lowpass_filter(sig_res, target_fs, cutoff_hz=20.0)
        sig8 = add_magnitude(sig_res)

        fall_label = int(code in MOBIACT_FALL_CODES)
        met = "unknown" if fall_label else mobiact_met_class(code)

        for w in make_windows(sig8, window_size, step):
            samples.append(
                WindowSample(
                    dataset="mobiact",
                    subject=subject,
                    age_group="adult",
                    activity_code=code,
                    fall_label=fall_label,
                    met_class=met,
                    signal=w.astype(np.float32),
                )
            )
    return samples


def load_uci_windows(uci_root: Path, include_test: bool = True, max_rows: Optional[int] = None) -> List[WindowSample]:
    activity_map = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING",
    }

    splits = ["train"] + (["test"] if include_test else [])

    samples: List[WindowSample] = []
    for split in splits:
        x_base = uci_root / split / "Inertial Signals"
        if not x_base.exists():
            continue

        def read_signal(name: str) -> np.ndarray:
            return np.loadtxt(x_base / name)

        acc_x = read_signal(f"total_acc_x_{split}.txt")
        acc_y = read_signal(f"total_acc_y_{split}.txt")
        acc_z = read_signal(f"total_acc_z_{split}.txt")
        gyr_x = read_signal(f"body_gyro_x_{split}.txt")
        gyr_y = read_signal(f"body_gyro_y_{split}.txt")
        gyr_z = read_signal(f"body_gyro_z_{split}.txt")

        y = np.loadtxt(uci_root / split / f"y_{split}.txt").astype(int)
        sub = np.loadtxt(uci_root / split / f"subject_{split}.txt").astype(int)

        n = len(y)
        if max_rows is not None:
            n = min(n, max_rows)

        for i in range(n):
            acc = np.vstack([acc_x[i], acc_y[i], acc_z[i]]).T
            gyr = np.vstack([gyr_x[i], gyr_y[i], gyr_z[i]]).T
            sig8 = add_magnitude(np.hstack([acc, gyr]))

            act_name = activity_map.get(int(y[i]), "UNKNOWN")
            samples.append(
                WindowSample(
                    dataset="uci",
                    subject=f"U{sub[i]:02d}",
                    age_group="adult",
                    activity_code=act_name,
                    fall_label=0,
                    met_class=uci_met_class(act_name),
                    signal=sig8.astype(np.float32),
                )
            )
    return samples


def build_unified_windows(
    data_root: Path,
    target_fs: int = 50,
    window_sec: float = 2.56,
    overlap: float = 0.5,
    include_uci: bool = True,
    include_mobiact: bool = True,
    include_sisfall: bool = True,
    max_files_per_dataset: Optional[int] = None,
) -> List[WindowSample]:
    all_samples: List[WindowSample] = []

    if include_sisfall:
        all_samples.extend(
            load_sisfall_windows(
                sisfall_root=data_root / "SisFall_dataset",
                target_fs=target_fs,
                window_sec=window_sec,
                overlap=overlap,
                max_files=max_files_per_dataset,
            )
        )

    if include_mobiact:
        all_samples.extend(
            load_mobiact_windows(
                mobiact_root=data_root / "MobiAct_Dataset_v2.0",
                target_fs=target_fs,
                window_sec=window_sec,
                overlap=overlap,
                max_files=max_files_per_dataset,
            )
        )

    if include_uci:
        all_samples.extend(
            load_uci_windows(
                uci_root=data_root / "UCI HAR Dataset",
                include_test=True,
                max_rows=max_files_per_dataset,
            )
        )

    return all_samples


def save_windows(samples: List[WindowSample], output_path: Path) -> None:
    """Save windows as separate metadata CSV and signals NPZ for memory efficiency."""
    rows: List[Dict] = []
    signals_list = []
    
    for s in samples:
        rows.append(
            {
                "dataset": s.dataset,
                "subject": s.subject,
                "age_group": s.age_group,
                "activity_code": s.activity_code,
                "fall_label": s.fall_label,
                "met_class": s.met_class,
            }
        )
        signals_list.append(s.signal)
    
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata as Parquet (more compact than CSV)
    metadata_path = output_path.with_suffix(".parquet")
    df.to_parquet(metadata_path, index=False)
    
    # Save signals as compressed NPZ
    signals_path = output_path.with_stem(output_path.stem + "_signals").with_suffix(".npz")
    
    try:
        # Using dtype=object for ragged arrays
        np.savez_compressed(signals_path, signals=np.array(signals_list, dtype=object))
    except OSError as e:
        print(f"✗ ERROR: Failed to save signals to {signals_path}: {e}")
        print("  This is likely due to running out of disk space.")
        # Clean up the partially written (corrupted) file to prevent errors downstream
        if signals_path.exists():
            print(f"  Deleting corrupted file: {signals_path}")
            signals_path.unlink()
        raise  # Re-raise the exception to halt the pipeline

    print(f"✓ Saved metadata to {metadata_path}")
    print(f"✓ Saved signals to {signals_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unified window dataset from SisFall/MobiAct/UCI.")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("results/artifacts/windows.pkl"))
    p.add_argument("--target-fs", type=int, default=50)
    p.add_argument("--window-sec", type=float, default=2.56)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--max-files-per-dataset", type=int, default=None)
    p.add_argument("--sisfall-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    include_uci = not args.sisfall_only
    include_mobiact = not args.sisfall_only

    samples = build_unified_windows(
        data_root=args.data_root,
        target_fs=args.target_fs,
        window_sec=args.window_sec,
        overlap=args.overlap,
        include_uci=include_uci,
        include_mobiact=include_mobiact,
        include_sisfall=True,
        max_files_per_dataset=args.max_files_per_dataset,
    )
    save_windows(samples, args.output)

    if not samples:
        print("No windows were produced. Check dataset paths.")
        return

    summary = pd.DataFrame(
        {
            "dataset": [s.dataset for s in samples],
            "fall": [s.fall_label for s in samples],
            "met": [s.met_class for s in samples],
            "subject": [s.subject for s in samples],
        }
    )

    print(f"Saved {len(samples)} windows to {args.output}")
    print("Windows by dataset:")
    print(summary.groupby("dataset").size())
    print("Fall ratio:")
    print(summary["fall"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
