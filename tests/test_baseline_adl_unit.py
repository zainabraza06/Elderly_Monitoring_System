"""Unit tests for ADL data prep (no MobiAct files required)."""

from __future__ import annotations

import numpy as np

from baseline_adl.data_prep import prepare_adl_dataset, stratified_train_test


def test_prepare_adl_filters_and_encodes() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 116))
    y_fall = np.zeros(200, dtype=int)
    y_fall[150:] = 1
    y_adl = np.array([0] * 120 + [1] * 30 + [2] * 50, dtype=int)

    Xo, yo, le = prepare_adl_dataset(X, y_fall, y_adl, min_samples=40)
    assert len(Xo) == len(yo)
    assert Xo.shape[1] == 116
    assert len(le.classes_) >= 1


def test_stratified_split_shapes() -> None:
    X = np.zeros((100, 10))
    y = np.array([0] * 50 + [1] * 50)
    a, b, c, d = stratified_train_test(X, y, test_size=0.2, random_state=42)
    assert len(a) == 80 and len(b) == 20
