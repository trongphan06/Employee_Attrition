# 📊 Employee Attrition Analysis & Prediction

Analysis and prediction of employee attrition risk based on the **Employee Attrition Analytics Dataset 2026** (15,000 employees, 32 columns).

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🔍 Key Insights

- **Overall attrition rate:** 32.3% Yes / 67.7% No — nearly 1 in 3 employees in the dataset belong to the "left" group.
- **By department:** `HR` has the highest attrition rate (33.3%), `Marketing` the lowest (31.4%). The gap between departments is only about ~2 percentage points — Department is not a strongly discriminating factor.
- **Tenure:** `HR` has the highest average tenure (4.6 years), `Marketing` the lowest (4.35 years). Notably, `IT` (35 years) and `Sales` (34 years) are the only two departments with employees who have stayed **over 30 years**.
- **Salary by job level:** Average income rises steadily and clearly with `Job_Level` — from ~$5.7K (level 1) to ~$14.6K (level 5), a ~2.6x gap between the lowest and highest levels.
- **Outliers:** Several numerical variables (`Commute_Time_Minutes`, `Absence_Days`...) have data points beyond the typical IQR threshold — worth noting if feeding into outlier-sensitive models (Logistic Regression).

---

## 📈 Analysis Results

**From the correlation chart:**
- Aside from `Attrition_Risk_Score` (a pre-existing field, r = 0.60), the two strongest correlates of Attrition are **Overtime** (r = +0.23) and **Job_Satisfaction** (r = −0.20) — heavy overtime and low job satisfaction are the clearest warning signals.
- Working-condition factors (`Work_Life_Balance`, `Distance_From_Home`, `Commute_Time_Minutes`) all show moderate correlation (|r| ≈ 0.10–0.15).
- Demographic traits (`Age`, `Number_Of_Projects`, `Team_Size`) show **almost no correlation** with Attrition (|r| < 0.02) → day-to-day work experience drives attrition far more than personal characteristics.

**From the models:**

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **AdaBoostClassifier** ⭐ | **0.748** | **0.665** | 0.446 | **0.534** | **0.777** |
| RandomForestClassifier | 0.733 | 0.633 | 0.412 | 0.499 | 0.753 |
| LogisticRegression | 0.723 | 0.628 | 0.351 | 0.450 | 0.733 |

- **AdaBoostClassifier** delivers the best overall performance, edging out Random Forest and Logistic Regression on both Accuracy and ROC-AUC.
- While overall Accuracy is decent (~75%), **Recall for the "will leave" class is only 35–45%** — the model misses more than half of the employees who actually leave. This aligns with the moderate-to-low correlations found during EDA: the current features aren't strong enough to fully separate the two classes.

---

## 🤖 Advanced Predictive Modeling

- **Preprocessing:** `StandardScaler` for numerical features combined with `OneHotEncoder` for categorical features via `ColumnTransformer`.
- **Data split:** 70/30 train/test, `stratify`-ed on `Attrition` to preserve class ratio across both sets.
- **Screening benchmark:** Used `LazyPredict` to run ~20 classification algorithms simultaneously (AdaBoost, LightGBM, Random Forest, XGBoost, Logistic Regression...) to shortlist promising candidates before going deeper.
- **Detailed training:** Three models were fully tuned and evaluated with Confusion Matrix + Classification Report: `AdaBoostClassifier`, `RandomForestClassifier`, `LogisticRegression`.
- **Model explainability:** Extracted Feature Importance from AdaBoost and Random Forest to identify which features contribute most to the prediction.

---

## ⚠️ Limitations & Future Work

- **Low Recall** on the Attrition = 1 class → the current model is suitable for spotting general trends but not yet reliable enough for "early warning" on specific individuals. Potential fixes:
  - Address class imbalance (`SMOTE`, `class_weight='balanced'`).
  - Tune the classification threshold instead of using the default 0.5.
- **Categorical variables not yet included** (`Department`, `Job_Role`, `Business_Travel`) in the training features — adding them could boost predictive power, especially since `Department` showed a slight difference in attrition during EDA.
- Try **GridSearch/Optuna** to tune hyperparameters for AdaBoost, LightGBM, and XGBoost — currently run with default parameters only.
- Add **SHAP values** to explain each feature's contribution at the individual prediction level, rather than relying solely on aggregate Feature Importance.
- Consider removing `Attrition_Risk_Score` from the training features (if this value is computed *after* attrition is already known) to avoid the risk of **data leakage**.
