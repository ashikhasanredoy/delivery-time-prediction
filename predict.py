from typing import Union, Dict, List, Any
import joblib
import pandas as pd

# Compatibility shim for unpickling scikit-learn ColumnTransformer across versions
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

from src.config import BEST_MODEL_PATH
from src.feature_engineering import add_time_float, drop_unused_columns


def load_inference_model(model_path=BEST_MODEL_PATH):
    return joblib.load(model_path)


def predict_delivery_time(
    data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
    model_path=BEST_MODEL_PATH
) -> Union[float, List[float]]:
    model = load_inference_model(model_path)

    if isinstance(data, dict):
        df = pd.DataFrame([data])
        is_single = True
    elif isinstance(data, list):
        df = pd.DataFrame(data)
        is_single = False
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        is_single = len(df) == 1
    else:
        raise ValueError("Input data must be a dict, list of dicts, or pandas DataFrame.")

    if "time_float" not in df.columns and "order_placed_at" in df.columns:
        df = add_time_float(df)

    df = drop_unused_columns(df)

    predictions = model.predict(df)

    if is_single:
        return float(predictions[0])
    return [float(p) for p in predictions]


if __name__ == "__main__":
    sample_order = {
        "restaurant_id": 10,
        "cuisine": "pizza",
        "restaurant_avg_prep_minutes": 12.9,
        "city_zone": "suburbs_north",
        "distance_km": 2.2,
        "items_count": 2,
        "order_subtotal": 19.24,
        "courier_vehicle": "scooter",
        "courier_trips_completed": 50.0,
        "weather": "cloudy",
        "order_placed_at": "2025-03-01 10:00:00"
    }

    try:
        predicted_time = predict_delivery_time(sample_order)
        print("=" * 60)
        print("SAMPLE PREDICTION")
        print("=" * 60)
        print(f"Input Order Details:\n{sample_order}\n")
        print(f"Predicted Delivery Time: {predicted_time:.2f} minutes")
        print("=" * 60)
    except FileNotFoundError:
        print(f"Model file not found at {BEST_MODEL_PATH}. Please run src/save_model.py first.")
