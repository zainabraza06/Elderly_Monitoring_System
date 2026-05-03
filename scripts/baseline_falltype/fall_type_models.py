"""Estimators matching Colab fall-type comparison (300 trees, etc.)."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def build_models_colab(random_state: int = 42) -> dict[str, Any]:
    """Same hyperparameters as Colab STEP 6."""
    lgbm = LGBMClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
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
        eval_metric="mlogloss",
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
