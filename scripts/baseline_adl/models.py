"""Estimator zoo — Colab Task 2 defaults (lighter trees than fall binary for speed)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from baseline_adl.config import RANDOM_STATE


def build_model_candidates() -> dict[str, Any]:
    """Same families as Colab ADL comparison cell (n_estimators=100 etc.)."""
    return {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=7,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
