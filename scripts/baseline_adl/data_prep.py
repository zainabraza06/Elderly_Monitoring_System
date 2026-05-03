"""Build ADL-only matrices from full-window features + activity labels."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from baseline_adl.config import MIN_SAMPLES_PER_CLASS, RANDOM_STATE, TEST_SIZE


def prepare_adl_dataset(
    X_feat: np.ndarray,
    y_fall: np.ndarray,
    y_adl_mixed: np.ndarray,
    *,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Non-fall rows only; drop rare classes; re-encode labels 0..C-1.

    Parameters
    ----------
    X_feat : (N, 116)
    y_fall : (N,) binary 1=fall
    y_adl_mixed : (N,) original multiclass labels from dataset (same length as windows).
    """
    mask = y_fall == 0
    X = X_feat[mask]
    y = y_adl_mixed[mask]

    uniq, cnt = np.unique(y, return_counts=True)
    valid = uniq[cnt >= min_samples]
    keep = np.isin(y, valid)
    X, y = X[keep], y[keep]

    le = LabelEncoder()
    y_clean = le.fit_transform(y)
    return X, y_clean, le


def stratified_train_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
