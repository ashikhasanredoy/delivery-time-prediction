from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

RAW_TRAIN_PATH = RAW_DATA_DIR / "deliveries_train.csv"
RAW_TEST_PATH = RAW_DATA_DIR / "deliveries_test.csv"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

BEST_MODEL_PATH = MODEL_DIR / "best" / "model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocess.pkl"

MODEL_RESULTS_PATH = RESULTS_DIR / "model_results.csv"
VALIDATION_RESULTS_PATH = RESULTS_DIR / "validation_results.csv"
TEST_RESULTS_PATH = RESULTS_DIR / "test_results.csv"

TARGET_COLUMN = "delivery_minutes"

RANDOM_STATE = 42
