
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
    
### **Data Imputation Techniques**
##  IterativeImputer (MICE), Random sampling

- MICE (Multiple Imputation by Chained Equations) was considered to *improve the reliability of missing value treatment by modeling each feature with missing data as a function of the others.*

- *Implementation*: IterativeImputer(estimator=BayesianRidge, max_iter=10, random_state=42)

- Applied to: LOAN, MORTDUE, VALUE, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC

### Handling Missing Values for Nominal Unordered Variables (JOB, REASON) - Random Sampling

- For nominal unordered categorical variables such as REASON and JOB, missing values were imputed using random sampling based on observed probability distributions. This preserved the original distribution of categories and prevented over-representation of the mode.



### **Statistical Evaluation and Model Comparison**

To ensure model selection is not only based on performance metrics but also on statistical robustness, the following methodologies were implemented:

* **Cross-Validation**:
  Used **Stratified K-Fold (k=5)** cross-validation to maintain class balance across folds. The **F1 Macro score** was used as the optimization metric to balance performance across both default and non-default classes.

* **Model Selection Metrics**:

  * **ROC-AUC**: Assesses overall classification ability across thresholds.
  * **F1 Score**: Evaluated globally (macro) and specifically for the minority class (`BAD = 1`), to balance precision and recall.
  * **Precision & Recall**: Critical for business impact—especially minimizing **false negatives**.

* **Statistical Significance Testing**:

  * **Wilcoxon Signed-Rank Test** and **Student’s t-Test**: For pairwise comparison of model scores across folds.
  * **Kruskal-Wallis Test**: To test performance differences across all models in a non-parametric framework.
  * **McNemar’s Test**: To compare error distributions between two classifiers.
  * **Cochran’s Q Test**: To assess consistency of classifier predictions across multiple models.

*Outcome*: These tests confirmed that while XGBoost and Random Forest performed similarly, **XGBoost demonstrated marginally better generalization** and model consistency. The **difference was statistically supported by McNemar’s and Cochran’s Q tests**, validating its recommendation for deployment.

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
 ## Evaluation of the Best Model: ROC-AUC and Confusion Matrix
 ![image](https://github.com/user-attachments/assets/92ae3655-44db-495e-ae61-157afc902031)

 Here’s a ready-to-paste section you can add to your **README.md** (under "Evaluation of the Best Model" or "Results") to describe and interpret the ROC curve and confusion matrix for **XGBoost**:

---

## Evaluation of the Best Model – XGBoost (ROC Curve & Confusion Matrix)

![ROC and Confusion Matrix](./548aa545-faad-4f41-b8fa-5170ae271630.png)

### **Left: ROC Curve**

* The **ROC Curve** (Receiver Operating Characteristic) shows the trade-off between the **true positive rate** (recall) and the **false positive rate** at various threshold settings.
* The **AUC (Area Under Curve)** is **0.97**, indicating excellent model performance.
* AUC values close to 1.0 suggest that the model effectively distinguishes between defaulters and non-defaulters.
* The ROC curve rapidly rises to the top-left corner, which reflects high sensitivity and low false positive rates.

> **Interpretation**: The XGBoost model has strong discriminative power and maintains high recall while minimizing false alarms.

---

### **Right: Confusion Matrix**

* The confusion matrix provides a breakdown of predictions on the test set:

  * **True Negatives (TN)**: 1409
  * **False Positives (FP)**: 22
  * **False Negatives (FN)**: 59
  * **True Positives (TP)**: 298

* **Recall (Sensitivity)** for defaulters:

  $$
  \frac{TP}{TP + FN} = \frac{298}{298 + 59} \approx 0.83
  $$

* **Precision** for defaulters:  
  $\frac{TP}{TP + FP} = \frac{298}{298 + 22} \approx 0.93$






> **Interpretation**:
> The model correctly identifies the majority of defaulters while keeping the false positive rate low. This balance is critical in loan approval scenarios, where minimizing false negatives reduces financial risk.

---

Let me know if you'd like to add the raw metric values (like F1-score, accuracy) below the matrix or include a threshold analysis.

  ##  Most Predictive Variables of Loan Default
 ![image](https://github.com/user-attachments/assets/fb13f404-343d-447c-8bae-818c1d14e964)

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



