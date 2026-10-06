"""
models.py
---------
Defines tuned candidate regression models.
"""

from typing import Dict, Any
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor

from .config import RANDOM_STATE


def get_base_models(random_state: int = RANDOM_STATE) -> Dict[str, Any]:
    """
    Returns a dictionary of tuned base regression models.
    """
    models = {
        "Ridge": Ridge(
            alpha=10.0
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            random_state=random_state
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1
        )
    }

    return models
