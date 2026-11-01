"""Streamlit web application for interactive Sales Prediction and Budget Optimization."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.config import (
    BEST_MODEL_PATH,
    METRICS_CSV_PATH,
    RAW_DATA_PATH,
    FEATURE_RANGES,
)
from src.predict import predict_sales

st.set_page_config(
    page_title="Sales Prediction & Media Mix Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2b5c8f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .header-style {
        color: #1f2937;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_metrics():
    return pd.read_csv(METRICS_CSV_PATH)


@st.cache_resource
def get_model():
    return joblib.load(BEST_MODEL_PATH)


df_raw = load_data()
metrics_df = load_metrics()
model = get_model()

# Sidebar Navigation
st.sidebar.title("📈 Sales AI Dashboard")
st.sidebar.markdown("**CodSoft Data Science Internship**")
st.sidebar.markdown("*Task 4: Sales Prediction using Python*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Dataset Explorer",
        "🤖 Model Comparison & Evaluation",
        "🔮 Sales Prediction Tool",
        "💡 What-If Scenario Simulator",
    ],
)

# ----------------- PAGE 1: DATASET EXPLORER -----------------
if page == "📊 Dataset Explorer":
    st.title("📊 Advertising Dataset Overview & Filters")
    st.markdown(
        "Explore the historical advertising campaign expenditures across **TV**, **Radio**, "
        "and **Newspaper** channels and their relationship with final **Sales**."
    )

    # Sidebar interactive filters
    st.sidebar.subheader("Filter Campaigns by Spend ($k)")
    tv_range = st.sidebar.slider(
        "TV Spend ($k)",
        float(df_raw["TV"].min()),
        float(df_raw["TV"].max()),
        (float(df_raw["TV"].min()), float(df_raw["TV"].max())),
    )
    radio_range = st.sidebar.slider(
        "Radio Spend ($k)",
        float(df_raw["Radio"].min()),
        float(df_raw["Radio"].max()),
        (float(df_raw["Radio"].min()), float(df_raw["Radio"].max())),
    )
    newspaper_range = st.sidebar.slider(
        "Newspaper Spend ($k)",
        float(df_raw["Newspaper"].min()),
        float(df_raw["Newspaper"].max()),
        (float(df_raw["Newspaper"].min()), float(df_raw["Newspaper"].max())),
    )

    filtered_df = df_raw[
        (df_raw["TV"].between(tv_range[0], tv_range[1]))
        & (df_raw["Radio"].between(radio_range[0], radio_range[1]))
        & (df_raw["Newspaper"].between(newspaper_range[0], newspaper_range[1]))
    ].copy()
    filtered_df["Total_Spend"] = (
        filtered_df["TV"] + filtered_df["Radio"] + filtered_df["Newspaper"]
    )

    # High-level KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campaigns Displayed", f"{len(filtered_df)} / {len(df_raw)}")
    c2.metric("Avg TV Spend", f"${filtered_df['TV'].mean():.1f}k")
    c3.metric("Avg Radio Spend", f"${filtered_df['Radio'].mean():.1f}k")
    c4.metric("Avg Sales", f"${filtered_df['Sales'].mean():.1f}k units")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Scatter Plot: Channel Spend vs Sales")
        selected_channel = st.selectbox(
            "Select Channel to Plot against Sales:", ["TV", "Radio", "Newspaper", "Total_Spend"]
        )
        try:
            fig_scatter = px.scatter(
                filtered_df,
                x=selected_channel,
                y="Sales",
                trendline="ols",
                color="Sales",
                color_continuous_scale="Viridis",
                title=f"{selected_channel} Spend vs. Sales Volume (with Trendline)",
            )
        except Exception:
            fig_scatter = px.scatter(
                filtered_df,
                x=selected_channel,
                y="Sales",
                color="Sales",
                color_continuous_scale="Viridis",
                title=f"{selected_channel} Spend vs. Sales Volume",
            )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_right:
        st.subheader("Channel Budget Composition")
        sums = {
            "TV": filtered_df["TV"].sum(),
            "Radio": filtered_df["Radio"].sum(),
            "Newspaper": filtered_df["Newspaper"].sum(),
        }
        fig_pie = px.pie(
            values=list(sums.values()),
            names=list(sums.keys()),
            color=list(sums.keys()),
            color_discrete_map={"TV": "#e74c3c", "Radio": "#3498db", "Newspaper": "#2ecc71"},
            hole=0.4,
            title="Total Spend Allocation Share Across Channels",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Campaign Dataset Table")
    st.dataframe(filtered_df.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv_bytes,
        file_name="filtered_advertising_sales.csv",
        mime="text/csv",
    )

# ----------------- PAGE 2: MODEL COMPARISON -----------------
elif page == "🤖 Model Comparison & Evaluation":
    st.title("🤖 Machine Learning Model Benchmarking")
    st.markdown(
        "Evaluation results across multiple algorithms evaluated on 5-fold cross validation "
        "and held-out test data (random_state=42)."
    )

    # Bar chart of RMSE
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("Test RMSE Comparison (Lower is Better)")
        chart_df = metrics_df[metrics_df["Model"] != "Baseline (Mean)"]
        fig_bars = px.bar(
            chart_df,
            x="Model",
            y="Test_RMSE",
            color="Test_RMSE",
            color_continuous_scale="Tealgrn_r",
            text_auto=".3f",
            title="Model Comparison by Held-Out Test RMSE",
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    with col2:
        st.subheader("Test R² Variance Explained (Higher is Better)")
        fig_r2 = px.bar(
            chart_df,
            x="Model",
            y="Test_R2",
            color="Test_R2",
            color_continuous_scale="Viridis",
            text_auto=".3f",
            title="Model Comparison by Test R² Score",
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Comprehensive Performance Metrics Table")
    st.dataframe(
        metrics_df.style.format({
            "Test_R2": "{:.4f}",
            "Test_RMSE": "{:.4f}",
            "Test_MAE": "{:.4f}",
            "Train_R2": "{:.4f}",
            "Train_RMSE": "{:.4f}",
            "CV_R2_Score": "{:.4f}",
            "CV_RMSE": "{:.4f}",
        }).highlight_min(subset=["Test_RMSE", "Test_MAE"], color="#d4edda")
        .highlight_max(subset=["Test_R2", "Train_R2"], color="#d4edda"),
        use_container_width=True,
    )

    csv_metrics = metrics_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Metrics Summary (CSV)",
        data=csv_metrics,
        file_name="model_metrics.csv",
        mime="text/csv",
    )

# ----------------- PAGE 3: PREDICTION TOOL -----------------
elif page == "🔮 Sales Prediction Tool":
    st.title("🔮 Interactive Sales Forecaster")
    st.markdown("Enter advertising allocations across channels to predict sales units ($k).")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        st.subheader("Marketing Budget Inputs ($ in Thousands)")
        tv_input = st.number_input(
            "TV Budget ($k)",
            min_value=0.0,
            max_value=1000.0,
            value=150.0,
            step=5.0,
            help="Typical training range: $0.7k to $296.4k",
        )
        radio_input = st.number_input(
            "Radio Budget ($k)",
            min_value=0.0,
            max_value=500.0,
            value=25.0,
            step=2.5,
            help="Typical training range: $0.0k to $49.6k",
        )
        newspaper_input = st.number_input(
            "Newspaper Budget ($k)",
            min_value=0.0,
            max_value=500.0,
            value=20.0,
            step=2.5,
            help="Typical training range: $0.3k to $114.0k",
        )

        predict_button = st.button("🚀 Calculate Predicted Sales", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Forecast Output")
        try:
            res = predict_sales(tv_input, radio_input, newspaper_input)
            pred = res["predicted_sales"]
            total_b = tv_input + radio_input + newspaper_input

            st.metric("Predicted Sales", f"{pred:,.2f}k units", delta=f"${pred:,.2f}k value")
            st.info(f"Total Campaign Investment: **${total_b:,.2f}k**")

            if res["warnings"]:
                for w in res["warnings"]:
                    st.warning(w)

            # Composition donut
            fig_alloc = go.Figure(
                data=[
                    go.Pie(
                        labels=["TV", "Radio", "Newspaper"],
                        values=[tv_input, radio_input, newspaper_input],
                        hole=0.45,
                        marker_colors=["#e74c3c", "#3498db", "#2ecc71"],
                    )
                ]
            )
            fig_alloc.update_layout(title="Proposed Budget Allocation", margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_alloc, use_container_width=True)

            pred_df = pd.DataFrame([{
                "TV_Spend": tv_input,
                "Radio_Spend": radio_input,
                "Newspaper_Spend": newspaper_input,
                "Total_Spend": total_b,
                "Predicted_Sales": pred,
            }])
            st.download_button(
                label="📥 Download Prediction (CSV)",
                data=pred_df.to_csv(index=False).encode("utf-8"),
                file_name="sales_prediction_result.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ----------------- PAGE 4: WHAT-IF SCENARIO SIMULATOR -----------------
elif page == "💡 What-If Scenario Simulator":
    st.title("💡 What-If Sensitivity Simulator")
    st.markdown(
        "> [!IMPORTANT]\n"
        "> **Notice**: Predictions displayed here represent **statistical model estimates** "
        "conditioned on historical data associations, and **not** a guaranteed promise of future sales return."
    )

    st.subheader("Base Budget Profile")
    b1, b2, b3 = st.columns(3)
    base_tv = b1.slider("Base TV Spend ($k)", 10.0, 300.0, 150.0)
    base_radio = b2.slider("Base Radio Spend ($k)", 0.0, 50.0, 25.0)
    base_np = b3.slider("Base Newspaper Spend ($k)", 0.0, 100.0, 20.0)

    base_pred = predict_sales(base_tv, base_radio, base_np)["predicted_sales"]
    st.metric("Base Forecasted Sales", f"{base_pred:,.2f}k units")

    st.divider()
    st.subheader("Vary One Channel While Holding Others Constant")
    target_channel = st.selectbox("Select Channel to Vary:", ["TV", "Radio", "Newspaper"])

    sweep_range = np.linspace(0, 300 if target_channel == "TV" else (50 if target_channel == "Radio" else 100), 40)
    sweep_results = []

    for val in sweep_range:
        cur_tv = val if target_channel == "TV" else base_tv
        cur_radio = val if target_channel == "Radio" else base_radio
        cur_np = val if target_channel == "Newspaper" else base_np
        pred_val = predict_sales(cur_tv, cur_radio, cur_np)["predicted_sales"]
        sweep_results.append({
            f"{target_channel}_Spend": val,
            "Forecasted_Sales": pred_val,
        })

    sweep_df = pd.DataFrame(sweep_results)
    fig_sweep = px.line(
        sweep_df,
        x=f"{target_channel}_Spend",
        y="Forecasted_Sales",
        title=f"Sensitivity Curve: Impact of Changing {target_channel} Spend on Sales",
        markers=True,
    )
    fig_sweep.add_vline(
        x=base_tv if target_channel == "TV" else (base_radio if target_channel == "Radio" else base_np),
        line_dash="dash",
        line_color="red",
        annotation_text="Current Base Value",
    )
    st.plotly_chart(fig_sweep, use_container_width=True)
