"""Import smoke tests for baseline_fall modular pipeline (no MobiAct required)."""

from __future__ import annotations

import pytest


def test_baseline_fall_pipeline_imports() -> None:
    import baseline_fall.config  # noqa: F401
    import baseline_fall.fall_detection_visualization  # noqa: F401
    import baseline_fall.subject_split  # noqa: F401

    pytest.importorskip("imblearn")
    import baseline_fall.sampling  # noqa: F401

    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    from baseline_fall.fall_detection_models import build_models_fall_detection

    m = build_models_fall_detection(random_state=0)
    assert "XGBoost" in m and len(m) == 5
