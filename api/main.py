"""
main.py
FastAPI endpoints for Food Delivery Time Prediction.
"""

from typing import List
from datetime import datetime
from pathlib import Path
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Compatibility shim for unpickling scikit-learn ColumnTransformer across versions
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

from src.config import BEST_MODEL_PATH, TEST_RESULTS_PATH
from src.feature_engineering import add_time_features, drop_unused_columns
from .schemas import (
    OrderItem,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse
)

MODEL = None
MODEL_METADATA = {}


def load_artifacts():
    """Load serialized model pipeline and benchmark metrics."""
    global MODEL, MODEL_METADATA

    if Path(BEST_MODEL_PATH).exists():
        MODEL = joblib.load(BEST_MODEL_PATH)

    if Path(TEST_RESULTS_PATH).exists():
        metrics_df = pd.read_csv(TEST_RESULTS_PATH)
        MODEL_METADATA["test_metrics"] = metrics_df.to_dict(orient="records")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown lifecycle."""
    load_artifacts()
    yield


app = FastAPI(
    title="Food Delivery Time Prediction API 🛵⏱️",
    description=(
        "Production-grade RESTful API to predict food delivery duration in minutes "
        "using tuned ensemble regression (Stacking Regressor) with data leakage protection."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

load_artifacts()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_input_dataframe(orders: List[OrderItem]) -> pd.DataFrame:
    """Converts Pydantic OrderItem objects into the required model feature DataFrame."""
    records = []
    current_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for order in orders:
        data = order.dict()
        if not data.get("order_placed_at"):
            data["order_placed_at"] = current_time_str
        records.append(data)

    df = pd.DataFrame(records)
    df = add_time_features(df)
    df = drop_unused_columns(df)
    return df


@app.get("/", tags=["Health"])
def root():
    """Welcome endpoint with service information."""
    return {
        "message": "Food Delivery Time Prediction API is running 🛵💨",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Liveness probe and model load status."""
    is_ready = MODEL is not None
    return HealthResponse(
        status="healthy" if is_ready else "unhealthy",
        model_loaded=is_ready,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Retrieve model architecture details and test set evaluation benchmarks."""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    return ModelInfoResponse(
        model_name="Stacking Regressor (RidgeCV Meta-Learner)",
        base_estimators=["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost"],
        target_transform="log1p / expm1",
        primary_metric="MAE (Mean Absolute Error)",
        benchmark_metrics=MODEL_METADATA.get("test_metrics", [
            {"Model": "Stacking", "MAE": 3.0576, "MSE": 39.2115, "RMSE": 6.2619, "R2": 0.6630}
        ])
    )


@app.post(
    "/predict",
    response_model=SinglePredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"]
)
def predict_single_order(order: OrderItem):
    """
    Predict delivery time in minutes for a single food order.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    try:
        df = format_input_dataframe([order])
        prediction = float(MODEL.predict(df)[0])

        ci_margin = 3.06
        lower_bound = max(5.0, round(prediction - ci_margin, 2))
        upper_bound = round(prediction + ci_margin, 2)

        return SinglePredictionResponse(
            predicted_delivery_minutes=round(prediction, 2),
            confidence_interval_lower=lower_bound,
            confidence_interval_upper=upper_bound,
            order_summary=order.dict(),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"]
)
def predict_batch_orders(payload: BatchPredictionRequest):
    """
    Predict delivery time in minutes for multiple orders concurrently.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact is not loaded."
        )

    if not payload.orders:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Order list cannot be empty."
        )

    try:
        df = format_input_dataframe(payload.orders)
        predictions = [round(float(p), 2) for p in MODEL.predict(df)]

        return BatchPredictionResponse(
            predictions=predictions,
            total_orders=len(predictions),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8088, reload=True)
