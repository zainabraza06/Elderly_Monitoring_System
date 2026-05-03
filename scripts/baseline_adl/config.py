"""ADL multiclass (Task 2) — reproducibility and defaults aligned with Colab."""

from __future__ import annotations

# Same RNG everywhere (notebook uses np.random.seed / RandomState 42 in splits).
RANDOM_STATE = 42

MIN_SAMPLES_PER_CLASS = 100
TEST_SIZE = 0.2
CV_FOLDS = 5

# 116-D enhanced fusion — must match `baseline_fall.enhanced_features`.
FEATURE_DIM = 116

# Colab ADL cell used X_train_adl_bal = X_train_adl_scaled (no SMOTE/ADASYN in that run).
DEFAULT_BALANCE_ADL = False

RESULTS_SUBDIR = "baseline_adl"
