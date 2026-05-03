"""263-D multi-sensor fall-type features + training helpers (MobiAct 4-class)."""

from __future__ import annotations

from baseline_falltype.config import (
    FALL_CODES,
    FALL_NAMES,
    FALL_TYPE_RAW_DIM,
    SAMPLING_RATE_HZ,
    TOP_MI_FEATURES_DEFAULT,
    WINDOW_DURATION_S,
    WINDOW_SAMPLES,
)
from baseline_falltype.feature_extractors import (
    CompleteFallFeatureExtractor,
    extract_fall_type_raw_vector,
)

__all__ = [
    "CompleteFallFeatureExtractor",
    "FALL_CODES",
    "FALL_NAMES",
    "FALL_TYPE_RAW_DIM",
    "SAMPLING_RATE_HZ",
    "TOP_MI_FEATURES_DEFAULT",
    "WINDOW_DURATION_S",
    "WINDOW_SAMPLES",
    "extract_fall_type_raw_vector",
]


def __getattr__(name: str):
    if name == "load_fall_windows_from_annotated_dir":
        from baseline_falltype.fall_window_dataset import load_fall_windows_from_annotated_dir

        return load_fall_windows_from_annotated_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
