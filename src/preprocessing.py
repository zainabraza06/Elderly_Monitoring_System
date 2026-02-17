# src/preprocessing.py
import numpy as np
from scipy.signal import resample

def resample_signal(data, original_hz=200, target_hz=50):
    new_samples = int(data.shape[0] * target_hz / original_hz)
    return resample(data, new_samples, axis=0)

def sliding_window(data, window_size=128, overlap=0.5):
    step = int(window_size * (1 - overlap))
    windows = []

    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])

    return windows
