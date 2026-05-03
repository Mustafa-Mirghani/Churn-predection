# 📉 Customer Churn Prediction — Telecom Industry

![Python](https://img.shields.io/badge/Python-3.10-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📋 Project Summary

Built an end-to-end binary classification pipeline to predict customer churn for a telecom company using the Telco Customer Churn dataset. Performed rigorous statistical feature analysis, trained and compared three machine learning models, and applied threshold optimization to maximize churner detection. The final model achieves **91% Recall** — identifying 9 out of every 10 customers at risk of churning — enabling the business to proactively intervene before losing customers.

---

## 💼 Business Context

Customer churn is one of the most critical challenges in the telecom industry. Acquiring a new customer costs **5x more** than retaining an existing one. With a churn rate of 36% in this dataset, the business is losing over one-third of its customer base. This model enables the retention team to:

- Identify at-risk customers **before** they leave
- Prioritize outreach based on churn probability scores
- Design targeted retention strategies for high-risk segments
- Reduce revenue loss through proactive intervention

---

## 📊 Dataset Overview

| Field | Detail |
|---|---|
| **Source** | Kaggle — Telco Customer Churn |
| **Size** | 7,043 rows × 21 features |
| **Target Variable** | Churn (Binary — Yes/No) |
| **Class Distribution** | 64% No Churn / 36% Churn |
| **Feature Types** | 2 Numerical, 17 Categorical |

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Statistical feature selection using **Chi-Square tests** and **Cramér's V** for all categorical features
- **T-Tests** for numerical features to identify significant predictors
- Visualizations — grouped bar charts, boxplots, KDE plots, correlation heatmap
- Dropped statistically insignificant features: `Gender`, `PhoneService`, `MultipleLines`

### 2. Data Preprocessing
- Dropped `TotalCharges` due to multicollinearity with `tenure` and `MonthlyCharges`
- Reclassified `SeniorCitizen` from numerical to binary categorical
- Label Encoding for binary features, One Hot Encoding for multi-class features
- **RobustScaler** applied to numerical features — chosen for robustness to outliers
- Class imbalance handled using two approaches: **SMOTE** and **Class Weights**
- 80/20 stratified train/test split with `random_state=42`

### 3. Modeling
- Trained and compared **3 models × 2 imbalance approaches = 6 configurations**
- Models: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with **GridSearchCV** (5-fold cross validation)
- Primary evaluation metric: **ROC-AUC** and **F2-Score** (Recall-weighted)

### 4. Optimization
- **Threshold Optimization** — moved decision threshold from 0.5 to 0.32 to maximize Recall
- **Feature Engineering** — created 3 domain-driven features:
  - `LoyaltyRatio` = tenure / MonthlyCharges
  - `ServicesCount` = sum of 6 add-on service subscriptions
  - `IsNewCustomer` = 1 if tenure < 6 months

---

## 🔍 Key EDA Findings

- **Contract Type** is the strongest predictor of churn (Cramér's V = 0.41) — month-to-month customers churn **3x more** than two-year contract customers
- **Fiber Optic** customers show the highest churn rate despite using the premium service — likely due to higher cost awareness and market comparison behavior
- **New customers** (tenure < 6 months) represent the highest churn risk segment — most churners leave within the first year
- **High monthly charges** significantly increase churn probability — churners pay on average $15 more per month than loyal customers
- Customers with **no add-on services** (Online Security, Tech Support) are significantly more likely to churn — fewer services means lower switching costs

---

## 🤖 Model Performance

### All Models Comparison

| Model | Approach | ROC-AUC | Recall | Precision | F2-Score |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.8279 | 0.76 | 0.50 | 0.6877 |
| **Logistic Regression** | **Class Weights** | **0.8431** | **0.78** | **0.51** | **0.7064** |
| Random Forest | SMOTE | 0.8117 | 0.63 | 0.54 | 0.6065 |
| Random Forest | Class Weights | 0.8227 | 0.49 | 0.61 | 0.5100 |
| XGBoost | SMOTE | 0.8349 | 0.76 | 0.52 | 0.6979 |
| XGBoost | Class Weights | 0.8417 | 0.53 | 0.63 | 0.5491 |

### Final Model Results
**Model:** Logistic Regression + Class Weights + Threshold Optimization (0.32) + Feature Engineering

```
ROC-AUC:   0.846
Recall:    0.909   ← Catches 91% of real churners
Precision: 0.438
F2-Score:  0.748
```

### Confusion Matrix (Test Set)
|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 600 ✅ | 435 ❌ |
| **Actual: Churn** | 34 ❌ | 340 ✅ |

> Out of 374 real churners — the model successfully identified **340** before they left.

---

## 🏆 Top Predictive Features

| Rank | Feature | Effect on Churn |
|---|---|---|
| 1 | LoyaltyRatio (tenure/MonthlyCharges) | Higher ratio → Lower churn |
| 2 | Tenure | Longer tenure → Lower churn |
| 3 | Contract_Two Year | Two-year contract → Lower churn |
| 4 | PaymentMethod_Electronic Check | Electronic check → Higher churn |
| 5 | InternetService_Fiber Optic | Fiber optic → Higher churn |

---

## 💡 Business Recommendations

1. **Incentivize contract migration** — Offer discounts or free service upgrades to month-to-month customers who transition to annual contracts. This directly targets the highest churn risk segment.

2. **New customer onboarding program** — Provide free premium add-ons (Online Security, Tech Support) for the first 6 months. Data shows new customers are most vulnerable — increasing switching costs early reduces early churn significantly.

3. **Fiber Optic retention strategy** — Launch quarterly satisfaction surveys and personalized loyalty offers for Fiber Optic customers. Their market awareness makes proactive engagement critical.

4. **Pricing audit for high-charge customers** — Sales team to proactively contact customers on suboptimal plans with personalized recommendations. Right-sizing plans reduces cost-driven churn without blanket discounts.

5. **Add-on service awareness campaign** — Targeted email or in-app campaigns introducing unsubscribed customers to Online Security and TechSupport with free 30-day trials. More services = higher switching costs = lower churn.

---

## 🛠️ Tech Stack

```
Python          3.10
pandas          2.0
numpy           1.24
scikit-learn    1.3
imbalanced-learn 0.11  (SMOTE)
xgboost         1.7
matplotlib      3.7
seaborn         0.12
scipy           1.11
```

---

## ▶️ How To Run

```bash
# 1. Clone the repository
git clone https://github.com/Mustafa-Mirghani/Churn-predection

# 2. Navigate to project directory
cd churn-predection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter End-to-End Churn predicition project.ipynb

# 5. Run all cells
```

> **Note:** Dataset available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Download and place in `data/` folder before running.

---

## 📈 Future Improvements

- [ ] Test ensemble stacking — combining Logistic Regression and XGBoost predictions
- [ ] Collect more recent data to retrain model periodically
- [ ] Explore deep learning approach with larger dataset

---

## 👤 Author

Mustafa Mirghani
🐙 [GitHub](https://github.com/Mustafa-Mirghani)

---

*This project was built as part of a data science portfolio. Feedback and suggestions are welcome.*

