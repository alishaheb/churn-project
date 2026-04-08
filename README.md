# churn-project
# Customer Churn Prediction (End-to-End ML Project)

## 📌 Overview

This project builds an end-to-end machine learning pipeline to predict customer churn for a telecom company. The goal is to identify customers at risk of leaving and provide actionable business insights.

Unlike basic ML notebooks, this project covers the full lifecycle:

* Data cleaning and preprocessing
* Feature engineering
* Model training and evaluation
* Explainability (SHAP)
* Deployment-ready API

---

## 📊 Dataset

* **Source:** Telco Customer Churn Dataset (Kaggle)
* **Size:** 7,043 customers, 21 features
* **Target Variable:** `Churn` (Yes/No)

---

## ⚙️ Project Structure

```
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── api.py
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── encoders.pkl
├── outputs/
│   ├── eda_churn_drivers.png
│   ├── model_evaluation.png
│   ├── shap_summary.png
│   ├── shap_bar.png
│   └── shap_waterfall.png
├── requirements.txt
└── README.md
```

---

## 🔍 Key Steps

### 1. Data Cleaning

* Handled missing values (e.g., `TotalCharges`)
* Removed non-informative columns (`customerID`)
* Checked class imbalance (27% churn)

### 2. Feature Engineering

Created meaningful features:

* `ChargePerTenure`
* `NumServices`
* `HasProtection`
* `TenureGroup`
* `AvgMonthlySpend`

These features capture real business behavior and improve model performance.

### 3. Handling Imbalanced Data

* Applied **SMOTE** on training data only
* Avoided data leakage

### 4. Model Training

Trained and compared:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost

Used **Stratified 5-Fold Cross-Validation**

### 5. Evaluation Metrics

* F1 Score
* AUC-ROC
* Confusion Matrix

Accuracy was avoided due to class imbalance.

---

## 📈 Results

* Best model selected based on AUC-ROC
* Strong performance in identifying churners (high recall)

---

## 🧠 Model Explainability (SHAP)

Used SHAP to explain predictions:

* **Summary Plot:** global feature importance
* **Bar Plot:** average impact of features
* **Waterfall Plot:** explains individual predictions

### Key Insights:

* Month-to-month contracts have highest churn
* Low tenure customers are high risk
* High spending relative to tenure increases churn probability

---

## 🚀 Deployment (FastAPI)

The model is wrapped in a REST API:

### Endpoint:

```
POST /predict
```

### Input:

Customer features (JSON)

### Output:

* Churn probability
* Prediction (Yes/No)
* Risk level (Low/Medium/High)

### Run locally:

```
uvicorn src.api:app --reload
```

Access interactive docs:

```
http://127.0.0.1:8000/docs
```

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* SHAP
* FastAPI
* Matplotlib / Seaborn

---

## 📌 Key Takeaways

* Feature engineering has the biggest impact on performance
* Handling class imbalance is critical in real-world problems
* Explainability is essential for business trust
* Deployment separates projects from real-world solutions

---

## 📎 How to Run

1. Clone the repo:

```
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run training:

```
python src/train.py
```

4. Start API:

```
uvicorn src.api:app --reload
```

---

---
