"""Import smoke test for baseline_falltype pipeline modules (no MobiAct required)."""

from __future__ import annotations

import pytest


def test_baseline_falltype_pipeline_imports() -> None:
    import baseline_falltype.feature_selection_mi  # noqa: F401
    import baseline_falltype.fall_type_visualization  # noqa: F401
    from baseline_falltype import FALL_NAMES, TOP_MI_FEATURES_DEFAULT

    assert TOP_MI_FEATURES_DEFAULT == 150
    assert FALL_NAMES["FOL"] == "Forward Fall"

    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    from baseline_falltype.fall_type_models import build_models_colab

    m = build_models_colab()
    assert "LightGBM" in m and "Voting Ensemble" in m
