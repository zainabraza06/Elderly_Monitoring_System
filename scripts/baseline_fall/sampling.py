"""Class-imbalance helpers for fall-binary and ADL training."""

from __future__ import annotations

import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE

from baseline_fall.config import RANDOM_STATE


def balance_fall_train(X: np.ndarray, y: np.ndarray, *, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """SMOTETomek (0.5) with SMOTE fallback — matches original baseline notebook."""
    try:
        smt = SMOTETomek(random_state=random_state, sampling_strategy=0.5)
        return smt.fit_resample(X, y)
    except Exception:
        smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
        return smote.fit_resample(X, y)


def balance_adl_train(X: np.ndarray, y: np.ndarray, *, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """ADASYN with SMOTE fallback for multiclass ADL."""
    try:
        return ADASYN(random_state=random_state, sampling_strategy="auto").fit_resample(X, y)
    except Exception:
        return SMOTE(random_state=random_state, sampling_strategy="auto").fit_resample(X, y)
