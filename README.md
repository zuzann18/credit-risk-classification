
# Loan Default Prediction: End-to-End ML Pipeline

This project provides a comprehensive machine learning solution for predicting loan defaults, using the HMEQ (Home Equity) dataset. The goal is to help banks and financial institutions automate, optimize, and explain credit risk decisions, replacing manual, error-prone processes with robust, interpretable ML models.

---

## Project Overview

- **Business Problem:**  
  Loan defaults threaten bank profitability and stability. The manual approval process is slow and can introduce bias. Automating this decision with data science ensures fairness, transparency, and efficiency.
  
- **Objective:**  
  Build and interpret a classification model to identify loan applicants at risk of default, providing actionable insights for both business users and regulators.

---

## Features

- **Data Preprocessing:**  
  - Imputation of missing values  
  - Outlier detection and treatment  
  - Feature encoding and scaling  
  - Handling class imbalance (SMOTE)

- **Exploratory Data Analysis (EDA):**  
  - Univariate, bivariate, and multivariate analysis  
  - Visualization of feature distributions, relationships, and correlations

- **Model Development:**  
  - Logistic Regression (baseline, interpretable)
  - Random Forest (ensemble, non-linear)
  - XGBoost (state-of-the-art boosting)
  - Hyperparameter tuning with GridSearchCV

- **Evaluation:**  
  - Cross-validated model comparison (ROC-AUC, F1, precision, recall, confusion matrices)
  - Statistical tests to ensure robust model selection

- **Interpretability:**  
  - SHAP (SHapley Additive exPlanations) for feature importance and transparent decisions
  - Business-oriented recommendations for credit policy

- **Deployment-Readiness:**  
  - Save/load full preprocessing + model pipeline  
  - Notebook-based documentation suitable for stakeholders

---

## Results

- **Best Model:** XGBoost (ROC-AUC ≈ 0.96, high F1-score, best at reducing false negatives)
- **Key Features:** Debt-to-Income Ratio, Age of Oldest Credit Line, Delinquencies, Major Derogatory Reports
- **Interpretability:** SHAP analysis supports explainable AI for regulatory compliance and customer transparency
### F1 Macro Score
 ![image](https://github.com/user-attachments/assets/1f3b8673-f63c-4f38-bd2e-6327dc3f93c2)

### Recommendation: XGBoost over Random Forest

* **F1 Score (Default Class)**: XGBoost = 0.82 vs. RF = 0.76
* **ROC AUC**: XGBoost = 0.961 vs. RF = 0.952
* **Cross-Validation**: XGBoost shows slightly better macro F1 across folds
* **Statistical Tests**: No significant difference, but McNemar's test favors XGBoost
* **Interpretability**: Both support SHAP; XGBoost integrates better
* **Efficiency**: XGBoost (`tree_method='hist'`) is faster and scales better

**Conclusion**: Both models are strong, but **XGBoost is preferred** for deployment due to better performance and generalization.
- ## Evaluation of the Best Model: ROC-AUC and Confusion Matrix
 ![image](https://github.com/user-attachments/assets/92ae3655-44db-495e-ae61-157afc902031)
 ![image](https://github.com/user-attachments/assets/fb13f404-343d-447c-8bae-818c1d14e964)

  ##  Most Predictive Variables of Loan Default
- Based on SHAP value analysis and confirmed by reduced-feature model performance:

- **DEBTINC** *(Debt-to-Income Ratio)* – Strongest predictor of default risk.
- **DELINQ** *(Number of Delinquent Credit Lines)* – High delinquency counts are consistently associated with default.
- **CLAGE** *(Age of Oldest Credit Line)* – Shorter histories are riskier.
- **DEROG** *(Number of Major Derogatory Reports)* – Critical marker for prior credit issues.
- **MORTDUE** and **VALUE** – High financial obligations and low asset coverage signal increased risk.

These features alone retained strong model performance when used with a tuned XGBoost classifier.

**Generalizability**:
  - Validated with stratified k-fold cross-validation and robust ROC AUC stability.
  - Reduced-feature models maintain effectiveness, even with fewer inputs – a sign of good generalization.



