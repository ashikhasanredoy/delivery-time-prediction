"""
voting.py
---------
Voting ensemble regression model.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone


def build_voting_model(
    preprocessor: Any,
    models: Dict[str, Any]
) -> TransformedTargetRegressor:
    """
    Constructs a VotingRegressor pipeline enclosing the provided base models
    with log1p target transformation.
    """
    estimators = [
        (name, Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", clone(model))]))
        for name, model in models.items()
    ]

    voting_model = VotingRegressor(
        estimators=estimators,
        weights=None,
        n_jobs=-1
    )

    return TransformedTargetRegressor(
        regressor=voting_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
