"""
schemas.py
----------
Pydantic data models for API requests and responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class OrderItem(BaseModel):
    restaurant_id: int = Field(..., example=10, description="Unique ID of restaurant")
    cuisine: str = Field(..., example="pizza", description="Cuisine category (e.g. pizza, indian, mexican, burger)")
    restaurant_avg_prep_minutes: float = Field(..., ge=1.0, le=120.0, example=12.9, description="Average kitchen prep duration in minutes")
    city_zone: str = Field(..., example="suburbs_north", description="City area/zone (e.g. downtown, midtown, uptown, suburbs_north, suburbs_south)")
    distance_km: float = Field(..., ge=0.1, le=50.0, example=2.2, description="Distance from restaurant to customer in kilometers")
    items_count: int = Field(..., ge=1, le=50, example=2, description="Number of items in the order")
    order_subtotal: float = Field(..., ge=1.0, example=19.24, description="Order total cost before taxes/fees")
    courier_vehicle: str = Field(..., example="scooter", description="Vehicle type (bike, scooter, car)")
    courier_trips_completed: Optional[float] = Field(default=50.0, ge=0.0, example=50.0, description="Lifetime trips completed by courier")
    weather: str = Field(..., example="cloudy", description="Current weather (clear, cloudy, rain, fog, windy, storm)")
    order_placed_at: Optional[str] = Field(
        default=None,
        example="2025-03-01 10:00:00",
        description="Timestamp when order was placed (YYYY-MM-DD HH:MM:SS). Defaults to current UTC if omitted."
    )


class SinglePredictionResponse(BaseModel):
    predicted_delivery_minutes: float = Field(..., example=23.49, description="Estimated delivery time in minutes")
    confidence_interval_lower: float = Field(..., example=20.43, description="Estimated lower bound (95% CI)")
    confidence_interval_upper: float = Field(..., example=26.55, description="Estimated upper bound (95% CI)")
    order_summary: Dict[str, Any]
    timestamp: str


class BatchPredictionRequest(BaseModel):
    orders: List[OrderItem]


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    total_orders: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_name: str
    base_estimators: List[str]
    target_transform: str
    primary_metric: str
    benchmark_metrics: List[Dict[str, Any]]
