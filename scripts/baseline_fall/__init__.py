"""Fall detection + ADL training (116-D enhanced MobiAct features).

Main fall-binary trainer (multi-model): `train_fall_detection_mobiact`.
Combined fall + ADL XGBoost: `train_mobiact_baselines`.
"""

from baseline_fall.config import BEST_FALL_MODEL_NAME, RANDOM_STATE, SCALER_FALL_NAME

__all__ = ["BEST_FALL_MODEL_NAME", "SCALER_FALL_NAME", "RANDOM_STATE"]
