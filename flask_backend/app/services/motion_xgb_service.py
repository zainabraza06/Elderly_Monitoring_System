"""Re-export; canonical implementation: ``scripts/inference/motion_pipeline.py``."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from inference.motion_pipeline import (
    InferenceArtifacts,
    _fall_type_vector_from_windows,
    load_artifacts,
    run_inference,
)

__all__ = [
    "InferenceArtifacts",
    "load_artifacts",
    "run_inference",
    "_fall_type_vector_from_windows",
]
