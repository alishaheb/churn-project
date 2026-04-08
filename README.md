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
