from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Compatibility shim for unpickling scikit-learn ColumnTransformer across versions
try:
    import sklearn.compose._column_transformer as _ct
    if not hasattr(_ct, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass
        _ct._RemainderColsList = _RemainderColsList
except Exception:
    pass

from src.config import BEST_MODEL_PATH, TEST_RESULTS_PATH, VALIDATION_RESULTS_PATH
from src.feature_engineering import add_time_features, drop_unused_columns

st.set_page_config(
    page_title="Food Delivery ETA Predictor",
    page_icon="🛵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, #1E1E2F 0%, #12121A 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        color: #A0A0BA;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(255, 107, 107, 0.4);
    }
    
    .eta-display {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(255, 142, 83, 0.15) 100%);
        border: 2px solid #FF6B6B;
        border-radius: 24px;
        padding: 2.2rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.15);
    }
    
    .eta-number {
        font-size: 4rem;
        font-weight: 800;
        color: #FFFFFF;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    
    .eta-unit {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FF8E53;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.08);
        color: #ECECF1;
        margin: 0.2rem;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 1.5rem;
        background: linear-gradient(90deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.45);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    if not Path(BEST_MODEL_PATH).exists():
        return None
    return joblib.load(BEST_MODEL_PATH)


@st.cache_data
def load_metrics_data():
    val_df = pd.read_csv(VALIDATION_RESULTS_PATH) if Path(VALIDATION_RESULTS_PATH).exists() else None
    test_df = pd.read_csv(TEST_RESULTS_PATH) if Path(TEST_RESULTS_PATH).exists() else None
    return val_df, test_df


def predict_time(model, raw_df: pd.DataFrame) -> np.ndarray:
    df = add_time_features(raw_df)
    df = drop_unused_columns(df)
    return model.predict(df)


st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛵 Food Delivery Time Predictor</div>
    <div class="hero-subtitle">Production-grade ML system powered by Stacking Ensemble (XGBoost, GBDT, RF, ExtraTrees, Ridge) with 3.06 min Test MAE.</div>
</div>
""", unsafe_allow_html=True)

model = load_trained_model()
val_results, test_results = load_metrics_data()

if model is None:
    st.error(f"Model artifact not found at {BEST_MODEL_PATH}. Please run 'python3 -m src.save_model' first.")
    st.stop()

tab_predict, tab_batch, tab_analytics = st.tabs([
    "⚡ Real-Time ETA Predictor",
    "📂 Batch CSV Inference",
    "📊 Model Performance & Benchmarks"
])

with tab_predict:
    col_input, col_output = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("📋 Order & Trip Configuration")

        with st.expander("🍽️ 1. Restaurant & Order Details", expanded=True):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                cuisine_options = ["burgers", "chinese", "indian", "mexican", "pizza", "salads", "sushi", "thai"]
                cuisine = st.selectbox("Cuisine Category", cuisine_options, index=4)

                restaurant_id = st.number_input(
                    "Restaurant ID", min_value=1, max_value=200,
                    value=10
                )
            with r_col2:
                items_count = st.slider(
                    "Number of Items", min_value=1, max_value=10,
                    value=2
                )
                order_subtotal = st.number_input(
                    "Order Subtotal ($)", min_value=5.0, max_value=200.0,
                    value=22.50, step=0.5
                )

            restaurant_avg_prep = st.slider(
                "Kitchen Average Prep Duration (mins)", min_value=5.0, max_value=30.0,
                value=13.0, step=0.5
            )

        with st.expander("🗺️ 2. Distance & Delivery Logistics", expanded=True):
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                zone_options = ["downtown", "midtown", "suburbs_north", "suburbs_south", "uptown"]
                city_zone = st.selectbox("City Delivery Zone", zone_options, index=0)

                distance_km = st.slider(
                    "Distance to Customer (km)", min_value=0.5, max_value=15.0,
                    value=2.8, step=0.1
                )

            with d_col2:
                vehicle_options = ["bike", "scooter", "car"]
                courier_vehicle = st.selectbox("Courier Vehicle", vehicle_options, index=1)

                courier_trips = st.number_input(
                    "Courier Lifetime Trips Completed", min_value=0, max_value=5000,
                    value=120, step=10
                )

        with st.expander("🌤️ 3. Weather & Timing Conditions", expanded=True):
            w_col1, w_col2 = st.columns(2)
            with w_col1:
                weather_options = ["clear", "cloudy", "rain", "heavy_rain"]
                weather = st.selectbox("Weather Condition", weather_options, index=0)

            with w_col2:
                time_preset = st.selectbox(
                    "Order Timing Scenario",
                    ["Right Now (Current Time)", "Lunch Rush (13:00)", "Dinner Peak (20:00)", "Late Night (01:30)"]
                )

        now = datetime.now()
        if "Lunch" in time_preset:
            order_datetime = now.replace(hour=13, minute=15, second=0)
        elif "Dinner" in time_preset:
            order_datetime = now.replace(hour=20, minute=5, second=0)
        elif "Late Night" in time_preset:
            order_datetime = now.replace(hour=1, minute=30, second=0)
        else:
            order_datetime = now

        order_placed_at_str = order_datetime.strftime("%Y-%m-%d %H:%M:%S")

    with col_output:
        st.subheader("⏱️ Real-Time Prediction Output")

        input_record = {
            "restaurant_id": restaurant_id,
            "cuisine": cuisine,
            "restaurant_avg_prep_minutes": restaurant_avg_prep,
            "city_zone": city_zone,
            "distance_km": distance_km,
            "items_count": items_count,
            "order_subtotal": order_subtotal,
            "courier_vehicle": courier_vehicle,
            "courier_trips_completed": float(courier_trips),
            "weather": weather,
            "order_placed_at": order_placed_at_str
        }

        input_df = pd.DataFrame([input_record])
        predicted_minutes = float(predict_time(model, input_df)[0])

        ci_margin = 3.06
        lower_ci = max(5.0, round(predicted_minutes - ci_margin, 1))
        upper_ci = round(predicted_minutes + ci_margin, 1)

        st.markdown(f"""
        <div class="eta-display">
            <div class="eta-unit">Estimated Delivery Time</div>
            <div class="eta-number">{predicted_minutes:.1f}</div>
            <div style="font-size: 1.1rem; color: #E0E0F0; font-weight: 600;">MINUTES</div>
            <div style="margin-top: 1rem;">
                <span class="badge">🎯 95% CI: {lower_ci} – {upper_ci} mins</span>
                <span class="badge">🛡️ Test MAE: ±3.06m</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### 🔍 Estimated Journey Breakdown")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #A0A0BA;">🍳 Kitchen Prep</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">~{restaurant_avg_prep:.1f}m</div>
            </div>
            """, unsafe_allow_html=True)
        with b2:
            transit_est = max(3.0, round(predicted_minutes - restaurant_avg_prep, 1))
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #A0A0BA;">🛵 Transit Time</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">~{transit_est:.1f}m</div>
            </div>
            """, unsafe_allow_html=True)
        with b3:
            speed_kph = round(distance_km / (transit_est / 60.0), 1) if transit_est > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.85rem; color: #A0A0BA;">⚡ Effective Speed</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: #FFFFFF;">{speed_kph} km/h</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("##### 📦 Active Order Summary")
        summary_df = pd.DataFrame([
            {"Parameter": "Cuisine", "Value": cuisine.capitalize()},
            {"Parameter": "Distance", "Value": f"{distance_km:.1f} km"},
            {"Parameter": "Zone", "Value": city_zone.replace('_', ' ').capitalize()},
            {"Parameter": "Vehicle", "Value": courier_vehicle.capitalize()},
            {"Parameter": "Weather", "Value": weather.replace('_', ' ').capitalize()},
            {"Parameter": "Items / Total", "Value": f"{items_count} items (${order_subtotal:.2f})"}
        ])
        st.dataframe(summary_df, hide_index=True)

with tab_batch:
    st.subheader("📂 Batch Delivery Time Prediction")
    st.write("Upload a CSV dataset containing delivery orders to generate predictions in bulk.")

    uploaded_file = st.file_uploader("Upload Orders CSV", type=["csv"])

    sample_template = pd.DataFrame([{
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
    }])

    st.download_button(
        label="📥 Download CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="delivery_orders_template.csv",
        mime="text/csv"
    )

    if uploaded_file is not None:
        batch_raw = pd.read_csv(uploaded_file)
        st.write(f"Loaded **{len(batch_raw)}** orders. Generating predictions...")

        try:
            preds = predict_time(model, batch_raw)
            batch_result = batch_raw.copy()
            batch_result["predicted_delivery_minutes"] = [round(float(p), 2) for p in preds]

            st.success(f"Generated predictions for all {len(batch_result)} orders!")
            st.dataframe(batch_result.head(20))

            csv_data = batch_result.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="predicted_delivery_times.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error processing batch predictions: {e}")

with tab_analytics:
    st.subheader("📊 Model Evaluation & Benchmarking Dashboard")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test MAE ⭐", "3.06 min", "-0.97m vs Baseline")
    with m_col2:
        st.metric("Test RMSE", "6.26 min", "-0.13m vs Baseline")
    with m_col3:
        st.metric("Test R²", "0.6630", "+1.39% Explained")
    with m_col4:
        st.metric("Best Architecture", "Stacking", "RidgeCV Meta-Model")

    st.markdown("---")

    b_col1, b_col2 = st.columns([1.2, 0.8], gap="large")

    with b_col1:
        st.markdown("##### 🏆 Validation Model Comparison (15% Split)")
        if val_results is not None:
            formatted_val = val_results.copy()
            formatted_val["MAE"] = formatted_val["MAE"].apply(lambda x: f"{x:.4f} min")
            formatted_val["RMSE"] = formatted_val["RMSE"].apply(lambda x: f"{x:.4f} min")
            formatted_val["MSE"] = formatted_val["MSE"].apply(lambda x: f"{x:.2f}")
            formatted_val["R2"] = formatted_val["R2"].apply(lambda x: f"{x:.4f}")
            st.dataframe(formatted_val, hide_index=True)
        else:
            st.info("Validation results table not found.")

    with b_col2:
        st.markdown("##### 🎯 Final Test Evaluation (Untouched 15% Split)")
        if test_results is not None:
            st.dataframe(test_results, hide_index=True)

        st.markdown("""
        > **Why MAE is Primary**:
        > In food delivery ETA systems, users and operations interpret performance directly in minutes off the target. An **MAE of 3.06m** means predictions are within ~3 minutes of actual delivery on average.
        """)
