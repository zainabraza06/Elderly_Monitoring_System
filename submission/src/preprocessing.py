# src/preprocessing.py
"""
Preprocessing pipeline for SisFall dataset.
Includes resampling, filtering, magnitude computation, windowing, and normalization.
"""
import numpy as np
from scipy.signal import resample, butter, filtfilt
from scipy.interpolate import interp1d


# =============================================================================
# CONSTANTS
# =============================================================================
ORIGINAL_HZ = 200  # SisFall sampling rate
TARGET_HZ = 50     # Target for smartphone compatibility
WINDOW_SEC = 2.5   # Window duration in seconds
OVERLAP = 0.5      # 50% overlap
WINDOW_SIZE = int(TARGET_HZ * WINDOW_SEC)  # 125 samples at 50 Hz


# =============================================================================
# SENSOR CONVERSION
# =============================================================================
def convert_adxl345(raw_data):
    """
    Convert ADXL345 accelerometer raw data to g units.
    ADXL345 sensitivity: 256 LSB/g (at ±2g range)
    """
    return raw_data / 256.0

def convert_itg3200(raw_data):
    """
    Convert ITG3200 gyroscope raw data to degrees/sec.
    ITG3200 sensitivity: 14.375 LSB/(deg/s)
    """
    return raw_data / 14.375

def convert_mma8451q(raw_data):
    """
    Convert MMA8451Q accelerometer raw data to g units.
    MMA8451Q sensitivity: 4096 LSB/g (at ±2g range, 14-bit mode)
    """
    return raw_data / 4096.0

def convert_sensors(data):
    """
    Convert all sensor readings to physical units.
    Input shape: (n_samples, 9)
    Columns: [ADXL345_X, Y, Z, ITG3200_X, Y, Z, MMA8451Q_X, Y, Z]
    """
    converted = np.zeros_like(data, dtype=np.float64)
    
    # ADXL345 accelerometer (columns 0-2)
    converted[:, 0:3] = convert_adxl345(data[:, 0:3])
    
    # ITG3200 gyroscope (columns 3-5)
    converted[:, 3:6] = convert_itg3200(data[:, 3:6])
    
    # MMA8451Q accelerometer (columns 6-8)
    converted[:, 6:9] = convert_mma8451q(data[:, 6:9])
    
    return converted


# =============================================================================
# RESAMPLING
# =============================================================================
def resample_signal(data, original_hz=ORIGINAL_HZ, target_hz=TARGET_HZ):
    """
    Resample signal from original_hz to target_hz.
    Uses scipy.signal.resample for anti-aliased resampling.
    """
    if original_hz == target_hz:
        return data
    
    new_samples = int(data.shape[0] * target_hz / original_hz)
    return resample(data, new_samples, axis=0)


def resample_with_interpolation(data, original_hz=ORIGINAL_HZ, target_hz=TARGET_HZ):
    """
    Alternative resampling using interpolation.
    Can be more stable for some signals.
    """
    if original_hz == target_hz:
        return data
    
    n_samples = data.shape[0]
    original_times = np.arange(n_samples) / original_hz
    
    new_n_samples = int(n_samples * target_hz / original_hz)
    new_times = np.arange(new_n_samples) / target_hz
    
    resampled = np.zeros((new_n_samples, data.shape[1]))
    for i in range(data.shape[1]):
        f = interp1d(original_times, data[:, i], kind='linear', fill_value='extrapolate')
        resampled[:, i] = f(new_times)
    
    return resampled


# =============================================================================
# FILTERING
# =============================================================================
def butterworth_lowpass_filter(data, cutoff=20, fs=TARGET_HZ, order=4):
    """
    Apply 4th order Butterworth low-pass filter.
    Removes high-frequency noise while preserving fall/activity characteristics.
    
    Args:
        data: Input signal (n_samples, n_channels)
        cutoff: Cutoff frequency in Hz (default 20 Hz)
        fs: Sampling frequency in Hz
        order: Filter order (default 4)
    
    Returns:
        Filtered signal
    """
    # Nyquist frequency
    nyq = 0.5 * fs
    
    # Normalized cutoff frequency
    normal_cutoff = cutoff / nyq
    
    # Ensure cutoff is valid
    if normal_cutoff >= 1:
        normal_cutoff = 0.99
    
    # Design Butterworth filter
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter to each channel
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        # Use filtfilt for zero-phase filtering
        filtered[:, i] = filtfilt(b, a, data[:, i])
    
    return filtered


def apply_bandpass_filter(data, lowcut=0.5, highcut=20, fs=TARGET_HZ, order=4):
    """
    Apply bandpass filter to remove DC offset and high-frequency noise.
    Useful for isolating movement-related signals.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure valid range
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.01, min(high, 0.99))
    
    b, a = butter(order, [low, high], btype='band')
    
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])
    
    return filtered


# =============================================================================
# MAGNITUDE COMPUTATION
# =============================================================================
def compute_magnitude(data_xyz):
    """
    Compute signal magnitude from 3-axis data.
    Acc_mag = sqrt(x^2 + y^2 + z^2)
    
    Args:
        data_xyz: 3-column array [X, Y, Z]
    
    Returns:
        1D magnitude array
    """
    return np.sqrt(np.sum(data_xyz ** 2, axis=1))


def compute_all_magnitudes(data):
    """
    Compute magnitudes for all sensor groups.
    
    Input shape: (n_samples, 9)
    Returns: (n_samples, 12) with original 9 channels + 3 magnitudes
    """
    # Original 9 channels
    result = data.copy()
    
    # ADXL345 magnitude
    acc1_mag = compute_magnitude(data[:, 0:3])
    
    # ITG3200 gyroscope magnitude
    gyro_mag = compute_magnitude(data[:, 3:6])
    
    # MMA8451Q magnitude  
    acc2_mag = compute_magnitude(data[:, 6:9])
    
    # Stack all together
    magnitudes = np.column_stack([acc1_mag, gyro_mag, acc2_mag])
    result = np.column_stack([result, magnitudes])
    
    return result


# =============================================================================
# WINDOWING
# =============================================================================
def sliding_window(data, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """
    Split signal into overlapping windows.
    
    Args:
        data: Input signal
        window_size: Number of samples per window (default 125 for 2.5s at 50Hz)
        overlap: Overlap ratio (default 0.5 for 50%)
    
    Returns:
        List of window arrays
    """
    step = int(window_size * (1 - overlap))
    windows = []
    
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])
    
    return windows


def get_window_timestamps(data_length, window_size=WINDOW_SIZE, overlap=OVERLAP, fs=TARGET_HZ):
    """
    Get timestamps for each window center.
    Useful for aligning with labels or events.
    """
    step = int(window_size * (1 - overlap))
    timestamps = []
    
    for start in range(0, data_length - window_size + 1, step):
        center = start + window_size // 2
        timestamps.append(center / fs)
    
    return timestamps


# =============================================================================
# NORMALIZATION
# =============================================================================
class ZScoreNormalizer:
    """
    Z-score normalizer that fits on training data only.
    Essential for proper evaluation - prevents data leakage.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, X):
        """Compute mean and std from training data."""
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero
        self.std[self.std < 1e-8] = 1.0
        self.fitted = True
        return self
    
    def transform(self, X):
        """Apply normalization using fitted parameters."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        """Reverse normalization."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return X * self.std + self.mean


class MinMaxNormalizer:
    """
    Min-Max normalizer to scale features to [0, 1] range.
    """
    def __init__(self, feature_range=(0, 1)):
        self.min = None
        self.max = None
        self.feature_range = feature_range
        self.fitted = False
    
    def fit(self, X):
        """Compute min and max from training data."""
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        # Avoid division by zero
        diff = self.max - self.min
        diff[diff < 1e-8] = 1.0
        self.max = self.min + diff
        self.fitted = True
        return self
    
    def transform(self, X):
        """Apply normalization using fitted parameters."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        
        X_std = (X - self.min) / (self.max - self.min)
        X_scaled = X_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return X_scaled
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# =============================================================================
# COMPLETE PREPROCESSING PIPELINE
# =============================================================================
def preprocess_single_trial(data, convert_units=True, apply_filter=True, 
                            add_magnitudes=True, target_hz=TARGET_HZ):
    """
    Apply full preprocessing pipeline to a single trial.
    
    Steps:
    1. Convert raw sensor values to physical units
    2. Resample from 200 Hz to target Hz
    3. Apply Butterworth low-pass filter
    4. Compute signal magnitudes
    
    Args:
        data: Raw trial data (n_samples, 9)
        convert_units: Whether to convert to physical units
        apply_filter: Whether to apply low-pass filter
        add_magnitudes: Whether to add magnitude channels
        target_hz: Target sampling frequency
    
    Returns:
        Preprocessed data
    """
    processed = data.copy().astype(np.float64)
    
    # Step 1: Convert units
    if convert_units:
        processed = convert_sensors(processed)
    
    # Step 2: Resample
    processed = resample_signal(processed, ORIGINAL_HZ, target_hz)
    
    # Step 3: Apply filter (after resampling to save computation)
    if apply_filter:
        processed = butterworth_lowpass_filter(processed, cutoff=20, fs=target_hz)
    
    # Step 4: Add magnitudes
    if add_magnitudes:
        processed = compute_all_magnitudes(processed)
    
    return processed


def preprocess_trials(X_trials, convert_units=True, apply_filter=True,
                      add_magnitudes=True, target_hz=TARGET_HZ, verbose=True):
    """
    Preprocess a list of trials.
    
    Args:
        X_trials: List of raw trial arrays
        convert_units: Whether to convert to physical units
        apply_filter: Whether to apply low-pass filter
        add_magnitudes: Whether to add magnitude channels
        target_hz: Target sampling frequency
        verbose: Print progress
    
    Returns:
        List of preprocessed trial arrays
    """
    preprocessed = []
    
    for i, trial in enumerate(X_trials):
        if verbose and (i + 1) % 100 == 0:
            print(f"Preprocessing trial {i + 1}/{len(X_trials)}")
        
        processed = preprocess_single_trial(
            trial, convert_units, apply_filter, add_magnitudes, target_hz
        )
        preprocessed.append(processed)
    
    return preprocessed
