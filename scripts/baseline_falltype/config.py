"""Paths and hyperparameters for the multi-sensor fall-type baseline."""

from __future__ import annotations

from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _PKG_ROOT.parent.parent

DEFAULT_MODELS_DIR = REPO_ROOT / "models" / "baseline_falltype_multisensor"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "baseline_falltype_multisensor"

FALL_CODES = ("BSC", "FOL", "FKL", "SDL")
FALL_NAMES: dict[str, str] = {
    "BSC": "Back Fall",
    "FOL": "Forward Fall",
    "FKL": "Knees Fall",
    "SDL": "Side Fall",
}

# MobiAct annotated CSVs: impact-centered window (multi-sensor script)
SAMPLING_RATE_HZ = 50
WINDOW_DURATION_S = 6
WINDOW_SAMPLES = SAMPLING_RATE_HZ * WINDOW_DURATION_S  # 300

# Feature selection (mutual information)
TOP_N_FEATURES = 150
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Models (match Colab multi-sensor baseline)
LGBM_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric="mlogloss",
)
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
GB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    random_state=RANDOM_STATE,
)

MULTISENSOR_FEATURE_DIM = 350

# Pie chart (approximate feature counts by block — same totals as original notebook)
SENSOR_FEATURE_PIE = {
    "Accelerometer": 180,
    "Gyroscope": 84,
    "Orientation": 15,
    "Fall-specific": 50,
    "Cross-sensor": 15,
    "SMA": 1,
}
