"""Mutual-information feature selection on scaled raw vectors (Colab STEP 5)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_selection import mutual_info_classif


def select_top_features_mi(
    X_scaled: np.ndarray,
    y: np.ndarray,
    *,
    top_n: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (column_indices_sorted_ascending_by_mi, mi_scores_full).

    Training uses X_scaled[:, top_indices] where top_indices = indices[-top_n:].
    """
    mi_scores = mutual_info_classif(X_scaled, y, random_state=random_state)
    order = np.argsort(mi_scores)
    top_idx = order[-top_n:]
    return top_idx, mi_scores
