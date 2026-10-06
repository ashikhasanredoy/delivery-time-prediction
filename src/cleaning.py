from typing import Tuple, Optional
import numpy as np
import pandas as pd


def clean_train(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    df = df.copy()

    if "weather" in df.columns:
        df["weather"] = df["weather"].replace(r"^\s*$", np.nan, regex=True)
        df = df.dropna(subset=["weather"])

    median_trips = 0.0
    if "courier_trips_completed" in df.columns:
        median_trips = float(df["courier_trips_completed"].median())
        df["courier_trips_completed"] = df["courier_trips_completed"].fillna(median_trips)

    return df.reset_index(drop=True), median_trips


def clean_test(
    df: pd.DataFrame,
    median_trips: Optional[float] = None
) -> pd.DataFrame:
    df = df.copy()

    if "weather" in df.columns:
        df["weather"] = df["weather"].replace(r"^\s*$", np.nan, regex=True)
        df = df.dropna(subset=["weather"])

    if "courier_trips_completed" in df.columns and median_trips is not None:
        df["courier_trips_completed"] = df["courier_trips_completed"].fillna(median_trips)

    return df.reset_index(drop=True)
