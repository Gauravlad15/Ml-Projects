import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Models & Encoders ─────────────────────────────────────────────────────
xgb_model = joblib.load("xgb_model.pkl")
lr_model   = joblib.load("linear_model.pkl")
country_le = joblib.load("Country_labelencoder.pkl")
status_le  = joblib.load("Status_labelencoder.pkl")
trained_columns = list(xgb_model.get_booster().feature_names)

# ── Load Data ──────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("Life_expectancy_cleaned (1).csv")
df_raw.columns = df_raw.columns.str.strip()

df_encoded = pd.read_csv("Life_expectancy_cleaned.csv")
df_encoded.columns = df_encoded.columns.str.strip()

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Life Expectancy App", layout="wide")
st.title("🌍 Life Expectancy Prediction App")

tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Overview", "📈 EDA", "🎯 Model Metrics", "🔮 Predict"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📋 Dataset Overview")
    st.write(f"**Total Rows:** {df_raw.shape[0]}  |  **Total Columns:** {df_raw.shape[1]}")

    st.markdown("#### 📄 Full Dataset")
    st.dataframe(df_raw, use_container_width=True, height=500)

    st.markdown("#### 📊 Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)

    st.markdown("#### ❓ Missing Values")
    missing = df_raw.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing = missing[missing["Missing Count"] > 0]
    if missing.empty:
        st.success("✅ No missing values found!")
    else:
        st.dataframe(missing, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Exploratory Data Analysis")

    # Plot 1 — Life Expectancy Distribution
    st.markdown("#### Distribution of Life Expectancy")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df_raw, x="Life Expectancy", kde=True, ax=ax1, color="steelblue")
    ax1.set_title("Count of Life Expectancy", fontsize=14)
    ax1.set_xlabel("Life Expectancy")
    ax1.set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

    st.markdown("---")

    # Plot 2 — Year vs GDP
    st.markdown("#### Year & Country Wise GDP")
    data_sample = df_raw.sample(1000, random_state=42)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=data_sample, x="Year", y="GDP", ax=ax2, alpha=0.6, color="coral")
    ax2.set_title("Years & GDP", fontsize=14)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("GDP")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")

    # Plot 3 — Correlation Heatmap
    st.markdown("#### Feature Correlation Heatmap")
    x_corr = df_encoded.drop("Life Expectancy", axis=1)
    fig3, ax3 = plt.subplots(figsize=(16, 10))
    sns.heatmap(
        x_corr.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax3,
        fmt=".1f",
        annot_kws={"size": 8},
        linewidths=0.5
    )
    ax3.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🎯 Model Performance Metrics")

    # Exact values from notebook
    metrics = pd.DataFrame({
        "Model":    ["Linear Regression", "Linear Regression", "XGBRegressor", "XGBRegressor"],
        "Split":    ["Train",             "Test",              "Train",         "Test"],
        "R2 Score": [0.5173,              0.4864,              0.99,             0.9522],
        "MSE":      [44.3258,             44.6985,             0.1,             4.163]
    })

    st.dataframe(metrics, use_container_width=True)

    st.markdown("---")

    # R2 Score Chart
    st.markdown("#### R2 Score Comparison")
    r2_chart = alt.Chart(metrics).mark_bar().encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("R2 Score:Q", scale=alt.Scale(domain=[0, 1]), title="R2 Score"),
        color=alt.Color("Split:N", scale=alt.Scale(scheme="set2")),
        xOffset="Split:N",
        tooltip=["Model", "Split", "R2 Score"]
    ).properties(width=400, height=350, title="R2 Score — Train vs Train")
    st.altair_chart(r2_chart, use_container_width=True)

    st.markdown("---")

    # MSE Chart
    st.markdown("#### MSE Comparison")
    mse_chart = alt.Chart(metrics).mark_bar().encode(
        x=alt.X("Model:N", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("MSE:Q", title="Mean Squared Error"),
        color=alt.Color("Split:N", scale=alt.Scale(scheme="set2")),
        xOffset="Split:N",
        tooltip=["Model", "Split", "MSE"]
    ).properties(width=400, height=350, title="MSE — Train vs Train")
    st.altair_chart(mse_chart, use_container_width=True)

    st.markdown("---")

    # Train Accuracy Comparison (same as notebook)
    st.markdown("#### Overall Train Accuracy Comparison")
    acc_df = pd.DataFrame({
        "Models":   ["Linear Regression", "XGBRegressor"],
        "Accuracy": [0.5173,              0.99]
    })
    acc_chart = alt.Chart(acc_df).mark_bar().encode(
        x=alt.X("Models:N", 
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Accuracy:Q", 
                scale=alt.Scale(domain=[0, 1]), 
                title="R2 Accuracy"),
        color=alt.Color("Models:N", 
                        scale=alt.Scale(scheme="tableau10")),
        tooltip=["Models", "Accuracy"]
    ).properties(width=400, height=350, title="Models Accuracy Comparison")
    st.altair_chart(acc_chart, use_container_width=True)

    st.info("💡 XGBRegressor: Test R2 = 0.95 | Linear Regression: Test R2 = 0.48 (trained on single feature: Schooling)")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTION
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔮 Predict Life Expectancy")

    input_data = {}
    skip_cols    = {"Country", "Status", "Life Expectancy"}
    numeric_cols = [col for col in trained_columns if col not in skip_cols]
    half         = len(numeric_cols) // 2

    col1, col2 = st.columns(2)

    with col1:
        selected_country      = st.selectbox("Select Country", options=country_le.classes_)
        selected_status       = st.selectbox("Select Status",  options=status_le.classes_)
        input_data["Country"] = int(country_le.transform([selected_country])[0])
        input_data["Status"]  = int(status_le.transform([selected_status])[0])

        for i, col in enumerate(numeric_cols[:half]):
            col_data       = df_raw[col] if col in df_raw.columns else pd.Series([0])
            input_data[col] = st.number_input(
                label=f"{col}",
                min_value=float(col_data.min()),
                max_value=float(col_data.max()),
                value=float(col_data.median()),
                key=f"num1_{i}"
            )

    with col2:
        for i, col in enumerate(numeric_cols[half:]):
            col_data       = df_raw[col] if col in df_raw.columns else pd.Series([0])
            input_data[col] = st.number_input(
                label=f"{col}",
                min_value=float(col_data.min()),
                max_value=float(col_data.max()),
                value=float(col_data.median()),
                key=f"num2_{i}"
            )

    if st.button("🔮 Predict", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=trained_columns, fill_value=0)

            # XGBoost — all features
            pred_xgb = xgb_model.predict(input_df)[0]

            # Linear Regression — Schooling only, no scaler
            input_lr = input_df[["Schooling"]].values.reshape(-1, 1)
            pred_lr  = lr_model.predict(input_lr)[0]

            st.success("✅ Prediction Complete!")

            perf = pd.DataFrame({
                "Model": ["XGBRegressor", "Linear Regression"],
                "Predicted Life Expectancy (years)": [round(float(pred_xgb), 2), round(float(pred_lr), 2)]
            })
            st.dataframe(perf, use_container_width=True)

            chart = alt.Chart(perf).mark_bar().encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Predicted Life Expectancy (years):Q", scale=alt.Scale(zero=False)),
                color=alt.Color("Model:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=["Model", "Predicted Life Expectancy (years)"]
            ).properties(height=350, title="Model Prediction Comparison")
            st.altair_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
