# src/dataset_builder.py
import numpy as np
from src.preprocessing import resample_signal, sliding_window
from src.features import extract_features

def build_dataset(X_trials, y_trials):
    X_all = []
    y_all = []

    for trial, label in zip(X_trials, y_trials):

        trial_rs = resample_signal(trial)

        windows = sliding_window(trial_rs)

        for w in windows:
            feats = extract_features(w)
            X_all.append(feats)
            y_all.append(label)

    return np.array(X_all), np.array(y_all)
