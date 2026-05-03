"""Constants for MobiAct fall-binary (116-D) training."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
CV_FOLDS = 5
SUBJECT_TRAIN_FRAC = 0.8
# Feature width: `enhanced_features.ENHANCED_FEATURE_DIM` (116)

# Legacy Colab / notebook single-model name on disk (when training XGB-only combined script)
LEGACY_FALL_MODEL_NAME = "fall_detection_xgboost.pkl"

# Recommended inference artifact (best model after multi-model comparison)
BEST_FALL_MODEL_NAME = "best_fall_model.pkl"
SCALER_FALL_NAME = "scaler_fall.pkl"


def repo_scripts_parents() -> Path:
    """Parent of `scripts` (repo root)."""
    return Path(__file__).resolve().parents[2]
