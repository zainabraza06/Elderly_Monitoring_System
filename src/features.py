# src/features.py
import numpy as np
from scipy.fft import fft
from scipy.stats import entropy

def extract_features(window):
    features = []

    # Time-domain
    features.extend(np.mean(window, axis=0))
    features.extend(np.std(window, axis=0))
    features.extend(np.min(window, axis=0))
    features.extend(np.max(window, axis=0))

    # Frequency-domain
    for i in range(window.shape[1]):
        sig = window[:, i]
        fft_vals = np.abs(fft(sig))
        energy = np.sum(fft_vals**2)
        prob = fft_vals / (np.sum(fft_vals) + 1e-8)
        spec_entropy = entropy(prob)
        dom_freq = np.argmax(fft_vals)

        features.extend([energy, spec_entropy, dom_freq])

    return np.array(features)
