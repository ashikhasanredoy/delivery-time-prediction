import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "order_placed_at" in df.columns:
        df["datetime"] = pd.to_datetime(df["order_placed_at"])
    elif "datetime" not in df.columns:
        return df

    df["hour"] = df["datetime"].dt.hour
    df["minute"] = df["datetime"].dt.minute
    df["time_float"] = df["hour"] + df["minute"] / 60.0
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

    if "order_subtotal" in df.columns and "items_count" in df.columns:
        df["avg_item_price"] = df["order_subtotal"] / np.maximum(df["items_count"], 1)

    return df


def add_time_float(df: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(df)


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(
        columns=[
            "order_placed_at",
            "datetime",
            "order_id"
        ],
        errors="ignore"
    )
