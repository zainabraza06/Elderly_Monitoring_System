# src/features.py
"""
Comprehensive feature extraction for fall detection and gait analysis.
Includes time-domain, frequency-domain, and nonlinear features.
"""
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy, skew, kurtosis
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# CONSTANTS
# =============================================================================
SAMPLING_RATE = 50  # Hz after resampling


# =============================================================================
# TIME-DOMAIN FEATURES
# =============================================================================
def compute_mean(signal):
    """Compute mean of signal."""
    return np.mean(signal, axis=0)

def compute_std(signal):
    """Compute standard deviation."""
    return np.std(signal, axis=0)

def compute_var(signal):
    """Compute variance."""
    return np.var(signal, axis=0)

def compute_rms(signal):
    """
    Compute Root Mean Square.
    RMS = sqrt(mean(x^2))
    """
    return np.sqrt(np.mean(signal ** 2, axis=0))

def compute_peak(signal):
    """Compute peak (maximum absolute) value."""
    return np.max(np.abs(signal), axis=0)

def compute_min(signal):
    """Compute minimum value."""
    return np.min(signal, axis=0)

def compute_max(signal):
    """Compute maximum value."""
    return np.max(signal, axis=0)

def compute_range(signal):
    """Compute range (max - min)."""
    return np.max(signal, axis=0) - np.min(signal, axis=0)

def compute_iqr(signal):
    """Compute interquartile range (75th - 25th percentile)."""
    return np.percentile(signal, 75, axis=0) - np.percentile(signal, 25, axis=0)

def compute_skewness(signal):
    """Compute skewness of signal distribution."""
    return skew(signal, axis=0)

def compute_kurtosis(signal):
    """Compute kurtosis of signal distribution."""
    return kurtosis(signal, axis=0)

def compute_svm(signal_xyz):
    """
    Compute Signal Vector Magnitude (SVM).
    SVM = sqrt(x^2 + y^2 + z^2)
    Returns mean SVM over the window.
    """
    magnitude = np.sqrt(np.sum(signal_xyz ** 2, axis=1))
    return np.mean(magnitude)

def compute_svm_peak(signal_xyz):
    """Compute peak SVM value."""
    magnitude = np.sqrt(np.sum(signal_xyz ** 2, axis=1))
    return np.max(magnitude)

def compute_svm_std(signal_xyz):
    """Compute SVM standard deviation."""
    magnitude = np.sqrt(np.sum(signal_xyz ** 2, axis=1))
    return np.std(magnitude)


# =============================================================================
# JERK FEATURES (Rate of change of acceleration)
# =============================================================================
def compute_jerk(signal, fs=SAMPLING_RATE):
    """
    Compute jerk (derivative of acceleration).
    Jerk = d(acceleration)/dt
    """
    dt = 1.0 / fs
    jerk = np.diff(signal, axis=0) / dt
    return jerk

def compute_jerk_mean(signal, fs=SAMPLING_RATE):
    """Compute mean absolute jerk."""
    jerk = compute_jerk(signal, fs)
    return np.mean(np.abs(jerk), axis=0)

def compute_jerk_std(signal, fs=SAMPLING_RATE):
    """Compute jerk standard deviation."""
    jerk = compute_jerk(signal, fs)
    return np.std(jerk, axis=0)

def compute_jerk_peak(signal, fs=SAMPLING_RATE):
    """Compute peak jerk magnitude."""
    jerk = compute_jerk(signal, fs)
    return np.max(np.abs(jerk), axis=0)


# =============================================================================
# IMPACT DETECTION FEATURES
# =============================================================================
def compute_impact_duration(signal_mag, threshold=None, fs=SAMPLING_RATE):
    """
    Estimate impact duration based on acceleration magnitude.
    Impact is defined as signal above threshold.
    """
    if threshold is None:
        threshold = np.mean(signal_mag) + 2 * np.std(signal_mag)
    
    above_threshold = signal_mag > threshold
    if not np.any(above_threshold):
        return 0.0
    
    # Find longest continuous segment above threshold
    changes = np.diff(above_threshold.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    
    if len(starts) == 0 or len(ends) == 0:
        return np.sum(above_threshold) / fs
    
    # Handle edge cases
    if above_threshold[0]:
        starts = np.insert(starts, 0, 0)
    if above_threshold[-1]:
        ends = np.append(ends, len(signal_mag))
    
    # Get longest duration
    if len(starts) > 0 and len(ends) > 0:
        min_len = min(len(starts), len(ends))
        durations = (ends[:min_len] - starts[:min_len]) / fs
        return np.max(durations) if len(durations) > 0 else 0.0
    
    return 0.0


def compute_fall_index(signal_xyz):
    """
    Compute fall index based on impact characteristics.
    Higher values indicate more fall-like behavior.
    """
    magnitude = np.sqrt(np.sum(signal_xyz ** 2, axis=1))
    peak = np.max(magnitude)
    mean_val = np.mean(magnitude)
    
    # Fall index: ratio of peak to mean, normalized
    if mean_val > 0:
        return peak / mean_val
    return 0.0


# =============================================================================
# FREQUENCY-DOMAIN FEATURES
# =============================================================================
def compute_fft_features(signal, fs=SAMPLING_RATE):
    """
    Compute FFT-based frequency domain features.
    
    Returns:
        energy: Total spectral energy
        spectral_entropy: Entropy of power spectrum
        peak_freq: Dominant frequency
        mean_freq: Mean frequency (spectral centroid)
    """
    n = len(signal)
    
    # Compute FFT
    fft_vals = np.abs(fft(signal))[:n // 2]
    freqs = fftfreq(n, 1/fs)[:n // 2]
    
    # Power spectrum
    power = fft_vals ** 2
    total_power = np.sum(power)
    
    # Energy
    energy = total_power
    
    # Spectral entropy
    if total_power > 0:
        power_norm = power / total_power
        power_norm = power_norm[power_norm > 0]  # Remove zeros
        spectral_entropy = entropy(power_norm)
    else:
        spectral_entropy = 0.0
    
    # Peak frequency
    peak_freq = freqs[np.argmax(power)] if len(power) > 0 else 0.0
    
    # Mean frequency (spectral centroid)
    if total_power > 0:
        mean_freq = np.sum(freqs * power) / total_power
    else:
        mean_freq = 0.0
    
    return energy, spectral_entropy, peak_freq, mean_freq


def compute_energy_bands(signal, fs=SAMPLING_RATE):
    """
    Compute energy in different frequency bands.
    
    Returns:
        energy_low: Energy in 0-5 Hz (body movement)
        energy_high: Energy in 5-20 Hz (impact/vibration)
        energy_ratio: Ratio of low to high frequency energy
    """
    n = len(signal)
    
    fft_vals = np.abs(fft(signal))[:n // 2]
    freqs = fftfreq(n, 1/fs)[:n // 2]
    power = fft_vals ** 2
    
    # Energy in 0-5 Hz band
    low_mask = (freqs >= 0) & (freqs <= 5)
    energy_low = np.sum(power[low_mask])
    
    # Energy in 5-20 Hz band
    high_mask = (freqs > 5) & (freqs <= 20)
    energy_high = np.sum(power[high_mask])
    
    # Energy ratio
    if energy_high > 0:
        energy_ratio = energy_low / energy_high
    else:
        energy_ratio = energy_low if energy_low > 0 else 0.0
    
    return energy_low, energy_high, energy_ratio


def compute_spectral_edge_freq(signal, fs=SAMPLING_RATE, percentile=95):
    """
    Compute spectral edge frequency.
    Frequency below which 'percentile'% of total power is contained.
    """
    n = len(signal)
    
    fft_vals = np.abs(fft(signal))[:n // 2]
    freqs = fftfreq(n, 1/fs)[:n // 2]
    power = fft_vals ** 2
    
    total_power = np.sum(power)
    if total_power == 0:
        return 0.0
    
    cumsum = np.cumsum(power)
    threshold = percentile / 100.0 * total_power
    
    idx = np.where(cumsum >= threshold)[0]
    if len(idx) > 0:
        return freqs[idx[0]]
    return freqs[-1] if len(freqs) > 0 else 0.0


# =============================================================================
# NONLINEAR FEATURES
# =============================================================================
def compute_sample_entropy(signal, m=2, r=None):
    """
    Compute sample entropy for signal complexity analysis.
    
    Args:
        signal: 1D input signal
        m: Embedding dimension (default 2)
        r: Tolerance (default 0.2 * std)
    
    Returns:
        Sample entropy value
    """
    N = len(signal)
    if N < m + 1:
        return 0.0
    
    if r is None:
        r = 0.2 * np.std(signal)
    
    if r == 0:
        return 0.0
    
    def _count_matches(templates, r):
        n = len(templates)
        if n < 2:
            return 0
        # Fully vectorised: shape (n, n, m) → (n, n) → upper-triangle sum
        # For window of 123 templates with m=2 this is a 123×123×2 array — fast
        diffs = np.abs(templates[:, np.newaxis, :] - templates[np.newaxis, :, :])
        max_diffs = diffs.max(axis=2)  # Chebyshev distance matrix (n, n)
        return int(np.sum(np.triu(max_diffs < r, k=1)))
    
    # Build templates
    templates_m = np.array([signal[i:i+m] for i in range(N-m)])
    templates_m1 = np.array([signal[i:i+m+1] for i in range(N-m)])
    
    # Count matches
    B = _count_matches(templates_m, r)
    A = _count_matches(templates_m1, r)
    
    if B == 0:
        return 0.0
    
    return -np.log(A / B) if A > 0 else 0.0


def compute_approximate_entropy(signal, m=2, r=None):
    """
    Compute approximate entropy.
    Similar to sample entropy but includes self-matches.
    """
    N = len(signal)
    if N < m + 1:
        return 0.0
    
    if r is None:
        r = 0.2 * np.std(signal)
    
    if r == 0:
        return 0.0
    
    def _phi(m):
        templates = np.array([signal[i:i+m] for i in range(N-m+1)])
        n_templates = len(templates)
        
        C = np.zeros(n_templates)
        for i in range(n_templates):
            count = 0
            for j in range(n_templates):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
            C[i] = count / n_templates
        
        return np.mean(np.log(C + 1e-10))
    
    return _phi(m) - _phi(m + 1)


def compute_zero_crossing_rate(signal):
    """
    Compute zero crossing rate.
    Useful for frequency estimation and activity detection.
    """
    # Subtract mean to center signal
    signal_centered = signal - np.mean(signal)
    
    zero_crossings = np.where(np.diff(np.signbit(signal_centered)))[0]
    return len(zero_crossings) / len(signal)


def compute_autocorrelation(signal, lag=None):
    """
    Compute autocorrelation coefficient.
    Measures signal self-similarity at given lag.
    """
    if lag is None:
        lag = len(signal) // 4
    
    if lag >= len(signal):
        lag = len(signal) // 2
    
    n = len(signal)
    mean_val = np.mean(signal)
    var_val = np.var(signal)
    
    if var_val == 0:
        return 0.0
    
    autocorr = np.sum((signal[:n-lag] - mean_val) * (signal[lag:] - mean_val)) / ((n - lag) * var_val)
    return autocorr


def compute_correlation_coefficient(signal1, signal2):
    """Compute Pearson correlation coefficient between two signals."""
    if len(signal1) != len(signal2):
        min_len = min(len(signal1), len(signal2))
        signal1 = signal1[:min_len]
        signal2 = signal2[:min_len]
    
    return np.corrcoef(signal1, signal2)[0, 1]


# =============================================================================
# GAIT-SPECIFIC FEATURES (for walking activities)
# =============================================================================
def compute_step_regularity(signal_vertical, fs=SAMPLING_RATE):
    """
    Compute step regularity using autocorrelation.
    Higher values indicate more regular gait pattern.
    """
    # Typical step time: 0.4-0.8s, so lag ~ 20-40 samples at 50Hz
    step_lag = int(0.5 * fs)  # ~0.5 second step time
    
    n = len(signal_vertical)
    if n < step_lag * 2:
        return 0.0
    
    # Autocorrelation at step frequency
    return compute_autocorrelation(signal_vertical, step_lag)


def compute_stride_regularity(signal_vertical, fs=SAMPLING_RATE):
    """
    Compute stride regularity using autocorrelation.
    A stride = 2 steps.
    """
    stride_lag = int(1.0 * fs)  # ~1 second stride time
    
    n = len(signal_vertical)
    if n < stride_lag * 2:
        return 0.0
    
    return compute_autocorrelation(signal_vertical, stride_lag)


def compute_harmonic_ratio(signal, fs=SAMPLING_RATE):
    """
    Compute harmonic ratio - measure of gait smoothness.
    Higher values indicate smoother, more stable gait.
    """
    n = len(signal)
    
    fft_vals = np.abs(fft(signal))[:n // 2]
    
    # Sum even harmonics (stable components)
    even_sum = np.sum(fft_vals[::2])
    
    # Sum odd harmonics (asymmetric components)
    odd_sum = np.sum(fft_vals[1::2])
    
    if odd_sum > 0:
        return even_sum / odd_sum
    return 0.0


def compute_gait_symmetry_index(signal_left, signal_right):
    """
    Compute gait symmetry index.
    100% = perfectly symmetric gait.
    """
    rms_left = compute_rms(signal_left)
    rms_right = compute_rms(signal_right)
    
    max_val = max(rms_left, rms_right)
    min_val = min(rms_left, rms_right)
    
    if max_val > 0:
        return 100 * min_val / max_val
    return 100.0


def compute_stride_variability(signal, fs=SAMPLING_RATE):
    """
    Compute stride time variability using peak detection.
    Higher variability indicates less stable gait.
    """
    # Find peaks (heel strikes)
    peaks, _ = find_peaks(signal, distance=int(0.3 * fs))  # Min 0.3s between peaks
    
    if len(peaks) < 3:
        return 0.0
    
    # Compute stride times
    stride_times = np.diff(peaks) / fs
    
    # Return coefficient of variation
    if np.mean(stride_times) > 0:
        return np.std(stride_times) / np.mean(stride_times)
    return 0.0


# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================
def extract_time_features(window):
    """
    Extract time-domain features from window.
    
    Args:
        window: (n_samples, n_channels) array
    
    Returns:
        Feature array
    """
    features = []
    
    # Basic statistics
    features.extend(compute_mean(window).flatten())
    features.extend(compute_std(window).flatten())
    features.extend(compute_var(window).flatten())
    features.extend(compute_rms(window).flatten())
    features.extend(compute_min(window).flatten())
    features.extend(compute_max(window).flatten())
    features.extend(compute_range(window).flatten())
    features.extend(compute_iqr(window).flatten())
    
    # Shape statistics
    features.extend(compute_skewness(window).flatten())
    features.extend(compute_kurtosis(window).flatten())
    
    # Jerk features (for accelerometer channels)
    features.extend(compute_jerk_mean(window).flatten())
    features.extend(compute_jerk_std(window).flatten())
    features.extend(compute_jerk_peak(window).flatten())
    
    return features


def extract_frequency_features(window, fs=SAMPLING_RATE):
    """
    Extract frequency-domain features from window.
    
    Args:
        window: (n_samples, n_channels) array
        fs: Sampling frequency
    
    Returns:
        Feature array
    """
    features = []
    
    for i in range(window.shape[1]):
        sig = window[:, i]
        
        # FFT features
        energy, spec_entropy, peak_freq, mean_freq = compute_fft_features(sig, fs)
        features.extend([energy, spec_entropy, peak_freq, mean_freq])
        
        # Energy bands
        energy_low, energy_high, energy_ratio = compute_energy_bands(sig, fs)
        features.extend([energy_low, energy_high, energy_ratio])
        
        # Spectral edge frequency
        sef = compute_spectral_edge_freq(sig, fs)
        features.append(sef)
    
    return features


def extract_nonlinear_features(window, fs=SAMPLING_RATE):
    """
    Extract nonlinear features from window.
    
    Args:
        window: (n_samples, n_channels) array
        fs: Sampling frequency
    
    Returns:
        Feature array
    """
    features = []
    
    # Only compute for first few channels (accelerometer) to save time
    n_channels = min(window.shape[1], 6)
    
    for i in range(n_channels):
        sig = window[:, i]
        
        # Sample entropy (computationally expensive, use small m)
        samp_ent = compute_sample_entropy(sig, m=2)
        features.append(samp_ent)
        
        # Zero crossing rate
        zcr = compute_zero_crossing_rate(sig)
        features.append(zcr)
        
        # Autocorrelation
        autocorr = compute_autocorrelation(sig)
        features.append(autocorr)
    
    return features


def extract_impact_features(window, fs=SAMPLING_RATE):
    """
    Extract impact-related features (important for fall detection).
    
    Args:
        window: (n_samples, n_channels) array
        fs: Sampling frequency
    
    Returns:
        Feature array
    """
    features = []
    
    # Use accelerometer channels (0:3 = ADXL345, 6:9 = MMA8451Q)
    # And magnitude channel if available (channel 9)
    
    # ADXL345 accelerometer
    acc1 = window[:, 0:3]
    features.append(compute_svm(acc1))
    features.append(compute_svm_peak(acc1))
    features.append(compute_svm_std(acc1))
    features.append(compute_fall_index(acc1))
    
    # Magnitude-based impact duration (if magnitude column exists)
    if window.shape[1] > 9:
        acc_mag = window[:, 9]  # ADXL345 magnitude
        features.append(compute_impact_duration(acc_mag, fs=fs))
    else:
        acc_mag = np.sqrt(np.sum(acc1 ** 2, axis=1))
        features.append(compute_impact_duration(acc_mag, fs=fs))
    
    # MMA8451Q accelerometer
    if window.shape[1] >= 9:
        acc2 = window[:, 6:9]
        features.append(compute_svm(acc2))
        features.append(compute_svm_peak(acc2))
        features.append(compute_fall_index(acc2))
    
    return features


def extract_gait_features(window, fs=SAMPLING_RATE):
    """
    Extract gait-specific features (for walking activities).
    
    Args:
        window: (n_samples, n_channels) array
        fs: Sampling frequency
    
    Returns:
        Feature array
    """
    features = []
    
    # Use vertical axis (Y axis typically)
    # ADXL345 Y-axis
    vertical = window[:, 1]
    
    features.append(compute_step_regularity(vertical, fs))
    features.append(compute_stride_regularity(vertical, fs))
    features.append(compute_stride_variability(vertical, fs))
    
    # Harmonic ratio for each accelerometer axis
    for i in range(min(3, window.shape[1])):
        features.append(compute_harmonic_ratio(window[:, i], fs))
    
    # Cross-axis correlation (ML vs AP)
    if window.shape[1] >= 3:
        # X (ML) vs Z (AP) correlation
        features.append(compute_correlation_coefficient(window[:, 0], window[:, 2]))
    
    return features


def extract_features(window, include_gait=False, fs=SAMPLING_RATE):
    """
    Extract all features from a window.
    
    Args:
        window: (n_samples, n_channels) array
        include_gait: Whether to include gait-specific features
        fs: Sampling frequency
    
    Returns:
        1D feature array
    """
    features = []
    
    # Time-domain features
    features.extend(extract_time_features(window))
    
    # Frequency-domain features
    features.extend(extract_frequency_features(window, fs))
    
    # Nonlinear features
    features.extend(extract_nonlinear_features(window, fs))
    
    # Impact features
    features.extend(extract_impact_features(window, fs))
    
    # Gait features (optional)
    if include_gait:
        features.extend(extract_gait_features(window, fs))
    
    return np.array(features, dtype=np.float64)


def get_feature_names(n_channels=12, include_gait=False):
    """
    Get feature names for documentation and analysis.
    
    Args:
        n_channels: Number of channels in input
        include_gait: Whether gait features are included
    
    Returns:
        List of feature names
    """
    names = []
    
    # Time-domain feature names
    for stat in ['mean', 'std', 'var', 'rms', 'min', 'max', 'range', 'iqr', 
                 'skew', 'kurt', 'jerk_mean', 'jerk_std', 'jerk_peak']:
        for ch in range(n_channels):
            names.append(f'{stat}_ch{ch}')
    
    # Frequency-domain feature names
    for ch in range(n_channels):
        names.extend([
            f'fft_energy_ch{ch}', f'spec_entropy_ch{ch}', 
            f'peak_freq_ch{ch}', f'mean_freq_ch{ch}',
            f'energy_low_ch{ch}', f'energy_high_ch{ch}', 
            f'energy_ratio_ch{ch}', f'sef_ch{ch}'
        ])
    
    # Nonlinear feature names (only for first 6 channels)
    for ch in range(min(n_channels, 6)):
        names.extend([
            f'sample_entropy_ch{ch}', f'zcr_ch{ch}', f'autocorr_ch{ch}'
        ])
    
    # Impact feature names
    names.extend([
        'svm_acc1', 'svm_peak_acc1', 'svm_std_acc1', 'fall_index_acc1',
        'impact_duration',
        'svm_acc2', 'svm_peak_acc2', 'fall_index_acc2'
    ])
    
    # Gait feature names
    if include_gait:
        names.extend([
            'step_regularity', 'stride_regularity', 'stride_variability',
            'harmonic_ratio_x', 'harmonic_ratio_y', 'harmonic_ratio_z',
            'cross_axis_corr'
        ])
    
    return names


# =============================================================================
# BATCH FEATURE EXTRACTION
# =============================================================================
def extract_features_batch(windows, include_gait=False, fs=SAMPLING_RATE, verbose=True):
    """
    Extract features from multiple windows.
    
    Args:
        windows: List of window arrays
        include_gait: Whether to include gait features
        fs: Sampling frequency
        verbose: Print progress
    
    Returns:
        2D feature array (n_windows, n_features)
    """
    features_list = []
    
    for i, window in enumerate(windows):
        if verbose and (i + 1) % 500 == 0:
            print(f"Extracting features: {i + 1}/{len(windows)}")
        
        feat = extract_features(window, include_gait, fs)
        features_list.append(feat)
    
    return np.array(features_list)
