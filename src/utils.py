from typing import Union, Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)


def compute_regression_metrics(
    y_true: Union[pd.Series, np.ndarray, List[float]],
    y_pred: Union[pd.Series, np.ndarray, List[float]]
) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    mse = float(mean_squared_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mape = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))
    medae = float(median_absolute_error(y_true_arr, y_pred_arr))

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "MedAE": medae
    }


def evaluate_model(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray, List[float]]
) -> Dict[str, float]:
    y_pred = model.predict(X)
    return compute_regression_metrics(y, y_pred)


def calculate_residuals(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray]
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    error = y_true_arr - y_pred_arr
    abs_error = np.abs(error)
    pct_error = np.abs(error / np.maximum(y_true_arr, 1e-6)) * 100.0

    return pd.DataFrame({
        "actual": y_true_arr,
        "predicted": y_pred_arr,
        "error": error,
        "absolute_error": abs_error,
        "percentage_error": pct_error
    })


def calculate_confidence_interval(
    prediction: float,
    mae_margin: float = 3.06,
    floor_value: float = 5.0
) -> Tuple[float, float]:
    lower = max(floor_value, round(prediction - mae_margin, 2))
    upper = round(prediction + mae_margin, 2)
    return lower, upper


def format_metrics_table(
    metrics_list: List[Dict[str, Any]],
    sort_by: str = "MAE",
    ascending: bool = True
) -> pd.DataFrame:
    df = pd.DataFrame(metrics_list)
    cols = [c for c in ["Model", "MAE", "MSE", "RMSE", "R2", "MAPE", "MedAE"] if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in cols]
    df = df[cols + remaining_cols]

    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

    return df


def save_metrics_to_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path]
) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_model_artifact(
    model_path: Union[str, Path]
) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    return joblib.load(path)
