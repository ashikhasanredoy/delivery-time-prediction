from typing import Tuple, Any
import joblib

from .config import BEST_MODEL_PATH, PREPROCESSOR_PATH
from .final_model import run_final_pipeline


def save_final_model() -> Tuple[Any, Any]:
    (
        model,
        preprocessor,
        best_model_name,
        results_df,
        test_metrics
    ) = run_final_pipeline()

    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, BEST_MODEL_PATH)
    print(f"Saved final {best_model_name} model pipeline to: {BEST_MODEL_PATH}")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved fitted preprocessor to: {PREPROCESSOR_PATH}")

    return model, preprocessor


if __name__ == "__main__":
    save_final_model()
