"""Fall vs non-fall estimators: legacy XGBoost + multi-model comparison (Colab-style)."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

from baseline_fall.config import RANDOM_STATE


def build_xgb_fall_baseline(*, random_state: int = RANDOM_STATE) -> XGBClassifier:
    """Single XGBoost used in the original `train_mobiact_baselines.py` fall block."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        eval_metric="logloss",
    )


def build_models_fall_detection(*, random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Same family as fall-type script; hyperparameters tuned for tabular binary classification."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as e:
        raise ImportError("Install lightgbm for multi-model fall detection: pip install lightgbm") from e

    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=random_state,
    )
    voting = VotingClassifier(
        estimators=[
            ("lgbm", clone(lgbm)),
            ("xgb", clone(xgb)),
            ("rf", clone(rf)),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return {
        "LightGBM": lgbm,
        "XGBoost": xgb,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Voting Ensemble": voting,
    }
