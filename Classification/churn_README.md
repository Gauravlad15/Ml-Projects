# 📊 Customer Churn Prediction

A machine learning project to predict customer churn using the Telco Customer Churn dataset. The project includes end-to-end pipeline from data cleaning to deployment.

---

## 🎯 Problem Statement

Customer churn is when a customer stops doing business with a company. Predicting churn early helps businesses take action to retain customers. This project predicts whether a customer will churn based on their account details and usage patterns.

---

## 📁 Dataset

- **Source:** IBM Telco Customer Churn Dataset
- **Rows:** 7043
- **Features:** 19
- **Target:** Churn (Yes/No)

---

## 🔄 Project Flow

```
Data Loading → Data Cleaning → EDA → Encoding → 
Train-Test Split → Scaling → SMOTE → Modeling → 
Evaluation → Feature Importance → Deployment
```

---

## ⚙️ Key Steps

### 1. Data Cleaning
- Dropped `customerID` (irrelevant feature)
- Fixed `TotalCharges` — converted from object to float
- Detected and handled 11 hidden null values (blank spaces)

### 2. Exploratory Data Analysis
- Churn distribution — 73% No Churn, 27% Churn
- Churn vs Contract Type — Month-to-month churns most
- Churn vs Tenure — New customers churn more
- Churn vs Monthly Charges — Higher charges = more churn
- Correlation Heatmap

### 3. Preprocessing
- Label Encoding for categorical columns
- Train-Test Split (80:20)
- Standard Scaling on numerical columns
- SMOTE applied on training data only to handle class imbalance (73:27)

### 4. Models Used

| Model | Accuracy |
|-------|----------|
| Logistic Regression | 76% |
| Random Forest | 78% |
| XGBoost | 77% |

### 5. Evaluation Metrics
- Accuracy Score
- Classification Report (Precision, Recall, F1 Score)
- Confusion Matrix

### 6. Feature Importance
Top features identified:
- Tenure
- Monthly Charges
- Contract Type

---

## 💡 Business Insights

- Month-to-month customers churn 3x more — offer long-term contract discounts
- New customers (tenure < 12 months) are at highest risk — improve onboarding
- High monthly charges drive churn — review pricing strategy
- Fiber optic users churn more despite paying premium — improve service quality

---

## 🚀 Deployment

Built an interactive web app using **Streamlit**:
- Input customer details
- Select model (LR / RF / XGBoost)
- Get real-time churn prediction with probability %

---

## 🛠️ Tech Stack

```
Python | Pandas | NumPy | Scikit-learn | XGBoost
Matplotlib | Seaborn | Streamlit | Joblib
```

---

## 📂 Project Structure

```
├── MLClassPro.ipynb      ← Main notebook
├── app.py                ← Streamlit app
├── logistic_model.pkl    ← Saved LR model
├── random_forest.pkl     ← Saved RF model
├── xgb.pkl               ← Saved XGBoost model
├── scaler.pkl            ← Saved scaler
└── README.md
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install streamlit scikit-learn xgboost pandas numpy joblib

# Run app
streamlit run app.py
```
