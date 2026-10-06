"""
data_loader.py
--------------
Loads raw delivery datasets.
"""

from typing import Tuple
import pandas as pd
from .config import RAW_TRAIN_PATH, RAW_TEST_PATH


def load_raw_data(
    train_path=RAW_TRAIN_PATH,
    test_path=RAW_TEST_PATH
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw train and test CSV files.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test
