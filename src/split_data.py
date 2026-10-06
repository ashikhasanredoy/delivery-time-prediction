from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE


def split_data(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state
    )

    return (
        X_train.reset_index(drop=True),
        X_validation.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_validation.reset_index(drop=True),
        y_test.reset_index(drop=True)
    )
