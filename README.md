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

## 👤 Author

Your Name
LinkedIn: your-link
GitHub: your-link

---


## 📊 Exploratory Data Analysis (EDA)

The following visualizations highlight key patterns in customer churn behavior:

![EDA Churn Drivers](https://github.com/alishaheb/churn-project/blob/main/eda_churn_drivers.png)

### Key Insights:

* **Month-to-month contracts** show the highest churn rate
* **Low tenure customers** are significantly more likely to leave
* **Higher monthly charges** correlate with increased churn
* **Fiber optic users** tend to churn more than DSL users
* **Electronic check payment method** has the highest churn

These insights guided the feature engineering and modeling strategy.

---

## 📈 Model Evaluation

The performance of multiple models was evaluated using ROC curves and confusion matrices:

![Model Evaluation](https://github.com/alishaheb/churn-project/blob/main/model_evaluation.png)

### Key Observations:

* Tree-based models (Random Forest, XGBoost) outperform linear models
* AUC-ROC was used as the primary metric due to class imbalance
* The selected model achieves strong recall for churned customers
* Minimizing **False Negatives** is critical (missing a churner is costly)

---

## 🧠 Model Explainability (SHAP)

### Feature Importance (Global)

![SHAP Summary](https://github.com/alishaheb/churn-project/blob/main/shap_summary.png)

* **Contract type** is the most influential feature
* **Tenure** strongly impacts churn probability
* **ChargePerTenure** (engineered feature) is a key risk indicator

---

### Feature Impact (Average)

![SHAP Bar](https://github.com/alishaheb/churn-project/blob/main/shap_bar.png)

* Confirms which features consistently influence predictions
* Helps prioritize business actions (e.g., targeting high-risk segments)

---

### Individual Prediction Explanation

![SHAP Waterfall](https://github.com/alishaheb/churn-project/blob/main/shap_waterfall.png)

This plot explains **why a specific customer is predicted to churn**:

* Red features → push toward churn
* Blue features → push away from churn

This allows full transparency into model decisions, making the solution usable in real business scenarios.

---

## 💡 Business Value

This project goes beyond prediction:

* Identifies **high-risk customers early**
* Enables **targeted retention strategies**
* Provides **explainable insights** for decision-makers

---
