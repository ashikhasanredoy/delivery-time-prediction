"""
stacking.py
-----------
Stacking ensemble regression model.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone


def build_stacking_model(
    preprocessor: Any,
    models: Dict[str, Any]
) -> TransformedTargetRegressor:
    """
    Constructs a StackingRegressor pipeline enclosing the base models
    with cross-validated RidgeCV meta-estimator and log1p target transformation.
    """
    estimators = [
        (name, Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", clone(model))]))
        for name, model in models.items()
    ]

    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-2, 3, 20)),
        cv=5,
        n_jobs=-1
    )

    return TransformedTargetRegressor(
        regressor=stacking_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
