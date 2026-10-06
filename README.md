# 🛵 Food Delivery Time Prediction — Production ML System

An end-to-end, modular, and production-grade Machine Learning system designed to accurately predict food delivery durations in minutes. The system leverages advanced ensemble learning (**Stacking & Voting Regressors**), strict data leakage prevention mechanisms, cyclical temporal feature engineering, and a **70% Train / 15% Validation / 15% Test** evaluation architecture.

---

## 📑 Table of Contents
1. [Project Overview & Problem Statement](#-project-overview--problem-statement)
2. [Folder & File Architecture](#-folder--file-architecture)
3. [Machine Learning Pipeline Breakdown](#-machine-learning-pipeline-breakdown)
   - [1. Data Ingestion & Integrity](#1-data-ingestion--integrity)
   - [2. Cleaning & Leakage Prevention](#2-cleaning--leakage-prevention)
   - [3. Feature Engineering & Cyclical Encodings](#3-feature-engineering--cyclical-encodings)
   - [4. Data Splitting Strategy (70 / 15 / 15)](#4-data-splitting-strategy-70--15--15)
   - [5. Preprocessing Pipelines](#5-preprocessing-pipelines)
4. [Models & Ensemble Architecture](#-models--ensemble-architecture)
   - [Base Estimators](#base-estimators)
   - [Voting Regressor](#voting-regressor)
   - [Stacking Regressor (Winner)](#stacking-regressor-winner)
   - [Target Log-Transformation (log1p / expm1)](#target-log-transformation-log1p--expm1)
5. [📊 Model Evaluation & Benchmarks](#-model-evaluation)
   - [Validation Comparison Table](#validation-model-comparison-1166-samples)
   - [Final Test Set Performance](#final-model-performance-untouched-15-test-set)
   - [Why MAE is the Primary Metric](#why-mae-is-the-primary-metric)
6. [🌐 Production REST API (FastAPI)](#-production-rest-api-fastapi)
   - [API Endpoints Overview](#api-endpoints-overview)
   - [Request & Response Payloads](#request--response-payloads)
7. [⚡ Quickstart & Execution Guide](#-quickstart--execution-guide)

---

## 🎯 Project Overview & Problem Statement

Accurate Estimated Time of Arrival (ETA) is a cornerstone of on-demand food delivery platforms (such as DoorDash, UberEats, and Zomato). An inaccurate ETA leads to customer dissatisfaction, driver idle times, and kitchen bottlenecks.

This project solves the ETA challenge by formulating it as a supervised regression task:
- **Input Features**: Order specifics (subtotal, items count, cuisine), spatial factors (distance, city zone), courier characteristics (vehicle type, historical trips completed), environmental conditions (weather), and temporal patterns (order timestamp).
- **Target Variable**: `delivery_minutes` (total elapsed time from order placement to customer delivery).

---

## 📁 Folder & File Architecture

The codebase is organized into isolated, single-responsibility modules:

```text
food-delivery-time-prediction/
│
├── data/
│   ├── raw/                           # Untouched raw CSV files
│   │   ├── deliveries_train.csv
│   │   └── deliveries_test.csv
│   │
│   └── processed/                     # Reproducible split datasets
│       ├── train.csv                  # 70% Training split (5,440 rows)
│       ├── validation.csv             # 15% Validation split (1,166 rows)
│       └── test.csv                   # 15% Untouched Test split (1,166 rows)
│
├── models/
│   ├── best/
│   │   └── model.pkl                  # Serialized Stacking pipeline (85% fitted)
│   └── preprocess.pkl                 # Fitted preprocessor ColumnTransformer
│
├── results/
│   ├── validation_results.csv         # Full model validation comparison
│   └── test_results.csv               # Final untouched test set metrics
│
├── api/                               # Production REST API
│   ├── __init__.py
│   ├── main.py                        # FastAPI application & routes
│   └── schemas.py                     # Pydantic data schemas & validation
│
├── src/                               # Core Machine Learning Modules
│   ├── __init__.py
│   ├── config.py                      # Central constants & path management
│   ├── data_loader.py                 # Raw data loading utilities
│   ├── cleaning.py                    # Missing value handling & leakage guards
│   ├── feature_engineering.py         # Cyclical time & domain feature creation
│   ├── split_data.py                  # 70/15/15 train/val/test splitting
│   ├── preprocessing.py               # Imputation, scaling, one-hot encoding
│   ├── models.py                      # Candidate base model definitions
│   ├── train.py                       # Training runner
│   ├── voting.py                      # Voting ensemble construction
│   ├── stacking.py                    # Stacking ensemble construction
│   ├── evaluation.py                  # Regression metric calculation
│   ├── model_selection.py             # Validation-based benchmark comparison
│   ├── final_model.py                 # End-to-end orchestration & 85% refit
│   └── save_model.py                  # Model & preprocessor disk serialization
│
├── app.py                             # Root ASGI server entrypoint
├── predict.py                         # Standalone CLI inference
├── streamlit_app.py                   # Interactive Streamlit Web UI
├── requirements.txt                   # Dependency specifications
└── README.md                          # Comprehensive system documentation
```

---

## 🔄 Machine Learning Pipeline Breakdown

```text
                    RAW DATASET
                         │
                         ▼
                 [data_loader.py]
                         │
                         ▼
                   [cleaning.py]
            (Impute with Train median)
                         │
                         ▼
             [feature_engineering.py]
        (Hour sin/cos, Day sin/cos, Ratios)
                         │
                         ▼
                  [split_data.py]
                         │
         ┌───────────────┼───────────────┐
         │               │               │
        70%             15%             15%
       TRAIN        VALIDATION         TEST
         │               │          (Holdout)
         └───────┬───────┘              │
                 │                      │
                 ▼                      │
       Candidate Base Models            │
     (RF, XGB, ET, GBDT, Ridge)         │
                 │                      │
                 ▼                      │
       Ensemble Construction            │
         (Voting & Stacking)            │
                 │                      │
                 ▼                      │
       Validation Evaluation            │
      (Sorted by MAE & RMSE)            │
                 │                      │
                 ▼                      │
      WINNING ARCHITECTURE              │
      (Stacking Regressor)              │
                 │                      │
                 ▼                      │
         85% COMBINED REFIT             │
        (Train + Validation)            │
                 │                      │
                 ▼                      │
         FINAL EVALUATION ──────────────┘
                 │
                 ▼
       MAE: 3.06m | R²: 0.6630
```

### 1. Data Ingestion & Integrity
`src/data_loader.py` loads raw CSV files into pandas DataFrames using standardized paths defined in `src/config.py`.

### 2. Cleaning & Leakage Prevention
`src/cleaning.py` addresses missing and noisy records:
- **Weather Attribute**: Rows with missing or blank whitespace strings (`^\s*$`) are dropped from training.
- **Courier Trips Completed**: Missing values are imputed using the **median calculated exclusively on the training set**. The test and validation sets reuse the training median to prevent data leakage.

### 3. Feature Engineering & Cyclical Encodings
`src/feature_engineering.py` extracts rich information from temporal and spatial features:
- **Cyclical Trigonometric Transformations**: Hours ($0 \dots 23$) and Days of the Week ($0 \dots 6$) repeat in continuous cycles. Ordinary numerical encodings imply that 23:59 and 00:00 are far apart (distance of 23), whereas trigonometrically they are adjacent:
  $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
  $$\text{dow\_sin} = \sin\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right), \quad \text{dow\_cos} = \cos\left(\frac{2\pi \cdot \text{day\_of\_week}}{7}\right)$$
- **Temporal Identifiers**: `hour`, `minute`, `time_float` (decimal hour), `day_of_week`, and `is_weekend`.
- **Domain Ratios**: `avg_item_price = order_subtotal / max(items_count, 1)`.

### 4. Data Splitting Strategy (70 / 15 / 15)
`src/split_data.py` enforces a two-stage split with a fixed random seed (`42`):
1. **Initial Split**: 70% Train, 30% Temporary.
2. **Second Split**: Temporary split 50/50 into 15% Validation and 15% Test.
The 15% Test set remains completely untouched until the final winning model architecture is selected.

### 5. Preprocessing Pipelines
`src/preprocessing.py` uses scikit-learn `ColumnTransformer` for feature transformation:
- **Numeric Features** (`distance_km`, `restaurant_avg_prep_minutes`, `items_count`, `order_subtotal`, `time_float`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `avg_item_price`, etc.): Imputed via median strategy and scaled via `StandardScaler`.
- **Categorical Features** (`cuisine`, `city_zone`, `courier_vehicle`, `weather`): Imputed via most frequent strategy and encoded with `OneHotEncoder(handle_unknown="ignore")`.

---

## 🤖 Models & Ensemble Architecture

### Base Estimators
Defined in `src/models.py`:
1. **Ridge Regression**: L2-regularized linear model ($\alpha=10.0$) capturing linear baseline dynamics.
2. **Random Forest Regressor**: 300 decision trees with `max_depth=12`, `min_samples_leaf=4`.
3. **Extra Trees Regressor**: 300 randomized trees reducing variance across high-cardinality features.
4. **Gradient Boosting Regressor**: 300 sequential boosting stages with `learning_rate=0.03`, `max_depth=4`, `subsample=0.8`.
5. **XGBoost Regressor**: 500 gradient-boosted trees with `learning_rate=0.03`, `max_depth=5`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`.

### Voting Regressor
`src/voting.py` combines predictions from all base estimators by computing the unweighted average.

### Stacking Regressor (Winner)
`src/stacking.py` implements a meta-learning ensemble:
- Base estimators generate out-of-fold cross-validated predictions ($k=5$).
- A meta-regressor (`RidgeCV` with 20 logarithmic $\alpha$ values) learns optimal weights for each base model's predictions.

### Target Log-Transformation (`log1p` / `expm1`)
Delivery time datasets exhibit long-tail positive skewness. Models are wrapped in `TransformedTargetRegressor` with:
- **Forward Transform**: $y_{\text{train}} = \log(1 + y)$
- **Inverse Transform**: $\hat{y} = \exp(\hat{y}_{\text{pred}}) - 1$
This penalizes proportional relative error rather than letting extreme tail outliers distort model parameters.

---

## 📊 Model Evaluation

The models were evaluated using **MAE, MSE, RMSE, and R²**. Since this is a delivery-time prediction problem, **MAE (Mean Absolute Error)** is considered the primary evaluation metric because it directly represents the average prediction error in minutes.

### Validation Model Comparison (1,166 samples)

| Rank | Model Architecture | MAE (min) ⭐ | MSE | RMSE (min) | R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 **1st** | **Stacking Regressor (RidgeCV Meta)** | **3.7743** | **100.3804** | **10.0190** | **0.4009** |
| 🥈 **2nd** | **Gradient Boosting** | 3.8347 | 100.8569 | 10.0428 | 0.3981 |
| 🥉 **3rd** | **XGBoost** | 3.8461 | 100.8360 | 10.0417 | 0.3982 |
| 4th | **Voting Regressor** | 3.9308 | 102.0150 | 10.1002 | 0.3912 |
| 5th | **Random Forest** | 4.2815 | 105.7931 | 10.2856 | 0.3686 |
| 6th | **Extra Trees** | 4.3107 | 105.7310 | 10.2826 | 0.3690 |
| 7th | **Ridge Regression** | 4.3739 | 107.7334 | 10.3795 | 0.3571 |

---

### Final Model Performance (Untouched 15% Test Set)

After selecting the winning **Stacking Regressor** architecture on validation data, the model was retrained on **85% combined data (6,606 samples)** and evaluated once on the **15% Test Set (1,166 samples)**:

| Metric   |            Score | Interpretation                                                           |
| -------- | ---------------: | ------------------------------------------------------------------------ |
| **MAE**  | **3.06 minutes** | Predictions are off by approximately 3.06 minutes on average             |
| **MSE**  |        **39.21** | Measures the average squared prediction error                            |
| **RMSE** | **6.26 minutes** | Penalizes larger prediction errors more heavily                          |
| **R²**   |       **0.6630** | The model explains approximately 66.3% of the variation in delivery time |

### Why MAE is the Primary Metric?

**MAE is the most important metric for this project** because delivery time is measured in minutes. It provides an intuitive interpretation of model performance.

An **MAE of 3.06 minutes** means that, on average, the predicted delivery time differs from the actual delivery time by approximately **3 minutes**.

RMSE is used as a secondary metric to identify the impact of larger prediction errors, while R² provides an overall measure of how well the model explains variations in delivery time.

### Best Model

The **Stacking Regressor** was selected as the final model based on its overall performance.

**Final Test Performance:**
* **MAE:** 3.06 minutes
* **RMSE:** 6.26 minutes
* **R²:** 0.6630

---

## 🌐 Production REST API (FastAPI)

The model is served through a production-ready asynchronous **FastAPI** application housed inside the `api/` directory.

### API Endpoints Overview
| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Root welcoming endpoint |
| `GET` | `/health` | Liveness probe and model load status |
| `GET` | `/model/info` | Architecture metadata and test benchmarks |
| `POST` | `/predict` | Single food order ETA prediction |
| `POST` | `/predict/batch` | High-throughput batch ETA predictions |
| `GET` | `/docs` | Interactive Swagger UI documentation |
| `GET` | `/redoc` | OpenAPI ReDoc technical documentation |

---

### Request & Response Payloads

#### Single Order Prediction (`POST /predict`)

**Request:**
```json
{
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
```

**Response (`200 OK`):**
```json
{
  "predicted_delivery_minutes": 23.49,
  "confidence_interval_lower": 20.43,
  "confidence_interval_upper": 26.55,
  "order_summary": {
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
  },
  "timestamp": "2026-10-06T15:29:06.060164"
}
```

---

## ⚡ Quickstart & Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Full Pipeline & Save Best Model Artifacts
Executes cleaning, feature engineering, 70/15/15 validation, model selection, 85% refit, test evaluation, and saves `.pkl` artifacts to `models/`:
```bash
python3 -m src.save_model
```

### 3. Run Standalone CLI Prediction
```bash
python3 predict.py
```

### 4. Start FastAPI Production Server 🌐
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8088 --reload
```
*(or via root entrypoint `uvicorn app:app --port 8088 --reload`)*

- **Interactive Swagger Docs**: [http://localhost:8088/docs](http://localhost:8088/docs)
- **ReDoc Documentation**: [http://localhost:8088/redoc](http://localhost:8088/redoc)
- **Health Check**: `curl -X GET http://localhost:8088/health`

### 5. Launch Interactive Streamlit User Interface 🎨
```bash
streamlit run streamlit_app.py --server.port 8502
```
- Opens interactive web portal at [http://localhost:8502](http://localhost:8502) with live ETA calculations, quick presets, batch CSV uploads, and model benchmarking dashboards.


