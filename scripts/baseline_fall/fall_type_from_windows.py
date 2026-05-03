"""
Bridge to 263-D fall-type features used after binary fall detection.

Training / definition lives in `scripts/baseline_falltype/` (Colab-aligned). The FastAPI
service calls the same `extract_fall_type_raw_vector` when `acc_window` is posted.
"""

from __future__ import annotations

from baseline_falltype.config import FALL_TYPE_RAW_DIM, WINDOW_SAMPLES
from baseline_falltype.feature_extractors import (
    CompleteFallFeatureExtractor,
    extract_fall_type_raw_vector,
)

__all__ = [
    "CompleteFallFeatureExtractor",
    "FALL_TYPE_RAW_DIM",
    "WINDOW_SAMPLES",
    "extract_fall_type_raw_vector",
]
