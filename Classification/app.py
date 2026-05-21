import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        width: 100%;
        border: none;
    }
    .stButton>button:hover { background-color: #1a252f; }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .churn { background-color: #ffe0e0; color: #c0392b; border: 2px solid #c0392b; }
    .no-churn { background-color: #e0ffe0; color: #27ae60; border: 2px solid #27ae60; }
</style>
""", unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    log = joblib.load('logistic_model.pkl')
    rm  = joblib.load('random_forest.pkl')
    xgb = joblib.load('xgb.pkl')
    ss  = joblib.load('scaler.pkl')
    return log, rm, xgb, ss

log, rm, xgb, ss = load_models()

# ── Title ─────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction")
st.markdown("Fill in the customer details below and select a model to predict churn.")
st.markdown("---")

# ── Layout: 3 Columns ─────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Partner", ["Yes", "No"])
    dependents      = st.selectbox("Dependents", ["Yes", "No"])
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

with col2:
    st.subheader("🌐 Internet & Services")
    internet_service  = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support      = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv      = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies  = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Account & Billing")
    tenure           = st.slider("Tenure (months)", 0, 72, 12)
    contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment          = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
    total_charges    = st.number_input("Total Charges ($)", 0.0, 10000.0, 500.0)

st.markdown("---")

# ── Model Selection ───────────────────────────────────────────
model_choice = st.selectbox("🤖 Select Model", [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
])

# ── Encoding Helper ───────────────────────────────────────────
def yes_no(val):
    return 1 if val == "Yes" else 0

def encode_three(val, options):
    return options.index(val)

# ── Predict Button ────────────────────────────────────────────
if st.button("🔍 Predict Churn"):

    # Encode inputs
    gender_enc    = 1 if gender == "Male" else 0
    senior_enc    = yes_no(senior_citizen)
    partner_enc   = yes_no(partner)
    depend_enc    = yes_no(dependents)
    phone_enc     = yes_no(phone_service)
    multiline_enc = encode_three(multiple_lines, ["No", "Yes", "No phone service"])
    internet_enc  = encode_three(internet_service, ["DSL", "Fiber optic", "No"])
    sec_enc       = encode_three(online_security, ["No", "Yes", "No internet service"])
    backup_enc    = encode_three(online_backup, ["No", "Yes", "No internet service"])
    device_enc    = encode_three(device_protection, ["No", "Yes", "No internet service"])
    tech_enc      = encode_three(tech_support, ["No", "Yes", "No internet service"])
    tv_enc        = encode_three(streaming_tv, ["No", "Yes", "No internet service"])
    movies_enc    = encode_three(streaming_movies, ["No", "Yes", "No internet service"])
    contract_enc  = encode_three(contract, ["Month-to-month", "One year", "Two year"])
    paper_enc     = yes_no(paperless)
    payment_enc   = encode_three(payment, [
                        "Bank transfer (automatic)", "Credit card (automatic)",
                        "Electronic check", "Mailed check"])

    try:
        # Exact column order matching training data:
        # ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        #  'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        #  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        #  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        #  'MonthlyCharges', 'TotalCharges']

        input_df = pd.DataFrame([[
            gender_enc,      # gender
            senior_enc,      # SeniorCitizen
            partner_enc,     # Partner
            depend_enc,      # Dependents
            tenure,          # tenure ← 5th
            phone_enc,       # PhoneService
            multiline_enc,   # MultipleLines
            internet_enc,    # InternetService
            sec_enc,         # OnlineSecurity
            backup_enc,      # OnlineBackup
            device_enc,      # DeviceProtection
            tech_enc,        # TechSupport
            tv_enc,          # StreamingTV
            movies_enc,      # StreamingMovies
            contract_enc,    # Contract
            paper_enc,       # PaperlessBilling
            payment_enc,     # PaymentMethod
            monthly_charges, # MonthlyCharges
            total_charges    # TotalCharges
        ]], columns=[
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ])

        # Scale numerical columns
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[num_cols] = ss.transform(input_df[num_cols])

        # Select model
        if model_choice == "Logistic Regression":
            model = log
        elif model_choice == "Random Forest":
            model = rm
        else:
            model = xgb

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1] * 100

        # Show result
        if prediction == 1:
            st.markdown(f"""
            <div class="result-box churn">
                ⚠️ Customer is likely to CHURN<br>
                <small>Churn Probability: {probability:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box no-churn">
                ✅ Customer will NOT Churn<br>
                <small>Churn Probability: {probability:.1f}%</small>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Make sure all .pkl files are in the same folder as app.py")