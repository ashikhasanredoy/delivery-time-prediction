from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from .models import get_base_models
from .preprocessing import build_preprocessor
from .voting import build_voting_model
from .stacking import build_stacking_model
from .evaluation import evaluate_model


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    preprocessor = build_preprocessor(X_train)
    models = get_base_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(model))
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_validation, y_validation)
        metrics["Model"] = name
        results.append(metrics)
        trained_models[name] = pipeline

    print("Training Voting Regressor...")
    voting_model = build_voting_model(clone(preprocessor), get_base_models())
    voting_model.fit(X_train, y_train)

    voting_metrics = evaluate_model(voting_model, X_validation, y_validation)
    voting_metrics["Model"] = "Voting"
    results.append(voting_metrics)
    trained_models["Voting"] = voting_model

    print("Training Stacking Regressor...")
    stacking_model = build_stacking_model(clone(preprocessor), get_base_models())
    stacking_model.fit(X_train, y_train)

    stacking_metrics = evaluate_model(stacking_model, X_validation, y_validation)
    stacking_metrics["Model"] = "Stacking"
    results.append(stacking_metrics)
    trained_models["Stacking"] = stacking_model

    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "MAE", "MSE", "RMSE", "R2"]]
    results_df = results_df.sort_values(by="MAE", ascending=True).reset_index(drop=True)

    return results_df, trained_models
