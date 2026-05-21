# 🏥 Life Expectancy Prediction

A machine learning project to predict life expectancy of countries using the WHO Life Expectancy dataset. The project includes end-to-end pipeline from data cleaning to deployment.

---

## 🎯 Problem Statement

Life expectancy varies significantly across countries due to factors like GDP, healthcare, education and lifestyle. This project predicts life expectancy based on these factors and identifies key drivers.

---

## 📁 Dataset

- **Source:** WHO Life Expectancy Dataset
- **Features:** 22
- **Countries:** 193
- **Target:** Life Expectancy (years)

---

## 🔄 Project Flow

```
Data Loading → Data Cleaning → Outlier Treatment → EDA → 
Preprocessing → Train-Test Split → Scaling → Modeling → 
Evaluation → Feature Importance → Deployment
```

---

## ⚙️ Key Steps

### 1. Data Cleaning
- Handled missing values using Median imputation
- Fixed column dtypes

### 2. Outlier Treatment
- Detected outliers using Boxplot visualization
- Removed outliers using IQR method (1.5 * IQR)
- Verified removal with before/after boxplots

### 3. Exploratory Data Analysis
- Life Expectancy distribution by country status
- GDP vs Life Expectancy correlation
- Adult Mortality vs Life Expectancy
- Schooling vs Life Expectancy
- Correlation Heatmap

### 4. Models Used

| Model | Metric |
|-------|--------|
| Linear Regression | Baseline |
| XGBoost | Best Performance |

### 5. Evaluation Metrics
- Mean Squared Error (MSE)
- R2 Score
- Model Comparison Bar Chart

### 6. Feature Importance
Top features identified:
- Adult Mortality
- GDP
- Schooling (Education)

---

## 💡 Business Insights

- Higher GDP strongly correlates with better life expectancy
- Adult Mortality is the most important predictor
- Education investment improves life expectancy significantly
- Developing countries average 60-65 years vs Developed 75+ years

---

## 🚀 Deployment

Built an interactive web app using **Streamlit**:
- Input country details
- Get real-time life expectancy prediction

---

## 🛠️ Tech Stack

```
Python | Pandas | NumPy | Scikit-learn | XGBoost
Matplotlib | Seaborn | Streamlit | Joblib
```

---

## 📂 Project Structure

```
├── LifeExpectancy.ipynb  ← Main notebook
├── app.py                ← Streamlit app
├── model.pkl             ← Saved model
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
