from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from .config import (
    TARGET_COLUMN,
    TRAIN_PATH,
    VALIDATION_PATH,
    TEST_PATH,
    VALIDATION_RESULTS_PATH,
    TEST_RESULTS_PATH
)
from .data_loader import load_raw_data
from .cleaning import clean_train
from .feature_engineering import add_time_features, drop_unused_columns
from .split_data import split_data
from .preprocessing import build_preprocessor
from .models import get_base_models
from .voting import build_voting_model
from .stacking import build_stacking_model
from .model_selection import compare_models
from .evaluation import evaluate_model


def run_final_pipeline() -> Tuple[Any, Any, str, pd.DataFrame, Dict[str, float]]:
    print("=" * 70)
    print("DELIVERY TIME PREDICTION - OPTIMIZED END-TO-END PIPELINE")
    print("=" * 70)

    raw_train, _ = load_raw_data()
    print(f"Raw training records loaded: {len(raw_train)}")

    df = add_time_features(raw_train)
    df = drop_unused_columns(df)

    df, _ = clean_train(df)
    print(f"Cleaned records available: {len(df)}")

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = split_data(df, TARGET_COLUMN)

    print(f"\nDataset Splits:")
    print(f"  • Train (70%):       {len(X_train)} samples")
    print(f"  • Validation (15%):  {len(X_validation)} samples")
    print(f"  • Test (15%):        {len(X_test)} samples")

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_PATH, index=False)
    pd.concat([X_validation, y_validation], axis=1).to_csv(VALIDATION_PATH, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_PATH, index=False)
    print(f"Saved processed train, validation, and test datasets to {TRAIN_PATH.parent}")

    results_df, _ = compare_models(X_train, y_train, X_validation, y_validation)

    VALIDATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(VALIDATION_RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("VALIDATION MODEL COMPARISON (Sorted by MAE)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["Model"]
    print("\n" + "=" * 70)
    print(f"BEST MODEL ARCHITECTURE: {best_model_name} (Validation MAE: {results_df.iloc[0]['MAE']:.4f})")
    print("=" * 70)

    X_train_final = pd.concat([X_train, X_validation], axis=0).reset_index(drop=True)
    y_train_final = pd.concat([y_train, y_validation], axis=0).reset_index(drop=True)

    final_preprocessor = build_preprocessor(X_train_final)
    final_preprocessor.fit(X_train_final)

    if best_model_name == "Voting":
        final_model = build_voting_model(final_preprocessor, get_base_models())
    elif best_model_name == "Stacking":
        final_model = build_stacking_model(final_preprocessor, get_base_models())
    else:
        base_models = get_base_models()
        final_model = TransformedTargetRegressor(
            regressor=Pipeline(
                steps=[
                    ("preprocessor", final_preprocessor),
                    ("model", clone(base_models[best_model_name]))
                ]
            ),
            func=np.log1p,
            inverse_func=np.expm1
        )

    print(f"\nRetraining final {best_model_name} model on full 85% dataset ({len(X_train_final)} samples)...")
    final_model.fit(X_train_final, y_train_final)

    test_metrics = evaluate_model(final_model, X_test, y_test)
    test_metrics_df = pd.DataFrame([{"Model": best_model_name, **test_metrics}])
    test_metrics_df.to_csv(TEST_RESULTS_PATH, index=False)

    print("\n" + "=" * 70)
    print("FINAL TEST PERFORMANCE (15% Untouched Test Set)")
    print("=" * 70)
    for metric, value in test_metrics.items():
        print(f"  • {metric:<6}: {value:.4f}")
    print("=" * 70)

    return final_model, final_preprocessor, best_model_name, results_df, test_metrics


if __name__ == "__main__":
    run_final_pipeline()
