"""Subject-level train/test split for MobiAct windows (leakage-safe)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from baseline_fall.config import RANDOM_STATE, SUBJECT_TRAIN_FRAC


def subject_masks(
    subject_ids: np.ndarray,
    y_fall: np.ndarray,
    *,
    frac: float = SUBJECT_TRAIN_FRAC,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split by subject id when possible; otherwise stratified index split.
    Returns boolean masks of length len(y_fall).
    """
    y_len = len(y_fall)
    subs = list({str(s) for s in subject_ids[:y_len]})
    rng = np.random.default_rng(random_state)
    rng.shuffle(subs)
    if len(subs) < 2:
        idx = np.arange(y_len)
        try:
            tr, te = train_test_split(
                idx, test_size=0.2, random_state=random_state, stratify=y_fall
            )
        except ValueError:
            tr, te = train_test_split(idx, test_size=0.2, random_state=random_state)
        train_m = np.zeros(y_len, dtype=bool)
        test_m = np.zeros(y_len, dtype=bool)
        train_m[tr] = True
        test_m[te] = True
        return train_m, test_m

    n_tr = max(1, int(frac * len(subs)))
    train_s = set(subs[:n_tr])
    test_s = set(subs[n_tr:])
    train_m = np.array([str(s) in train_s for s in subject_ids[:y_len]])
    test_m = np.array([str(s) in test_s for s in subject_ids[:y_len]])
    return train_m, test_m
