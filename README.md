# 🏦 Credit-Score-Classification

A full end-to-end machine learning pipeline for classifying credit scores into three categories — **Poor**, **Standard**, and **Good** — using structured financial and behavioral data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Models & Hyperparameter Tuning](#models--hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Feature Importance & SHAP](#feature-importance--shap)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Outputs](#outputs)

---

## 📌 Overview

This project tackles a **multi-class classification problem** to predict an individual's credit score category based on financial behavior and demographics. The pipeline covers everything from raw data ingestion to model explainability.

**Target Classes:**

| Class    | Description                        |
|----------|------------------------------------|
| Poor     | High-risk credit profile           |
| Standard | Average credit profile             |
| Good     | Low-risk, healthy credit profile   |

---

## 📂 Dataset

| Property        | Details                        |
|-----------------|--------------------------------|
| File            | `Credit Score.csv`             |
| Delimiter       | `,`                            |
| Target Column   | `Credit_Score`                 |
| Key Features    | Age, Annual Income, Debt, Payment Behavior, Credit History, etc. |

**Columns Dropped (non-informative identifiers):**
`ID`, `Customer_ID`, `Month`, `Name`, `SSN`

---

## 🗂 Project Structure

```
credit-score-classification/
│
├── Credit Score.csv               # Raw dataset
├── credit_score_notebook.ipynb    # Main notebook
│
└── outputs/
    ├── LGBM_credit_score.joblib   # Saved best model
    ├── feature_importance.csv     # Top feature importances
    └── predictions.csv            # Actual vs Predicted on test set
```

---

## 🔧 Pipeline Walkthrough

### 1. 🧹 Data Preprocessing & Cleaning

- **Type Casting**: Strips non-numeric characters and coerces columns like `Age`, `Annual_Income`, `Outstanding_Debt`, etc. to numeric.
- **Credit History Age**: Parsed from string format (e.g., `"3 Years 4 Months"`) into total months.
- **Noisy Values Replaced**: `'_______'` → `Unknown`, `'_'` → `NaN`, `'!@9#%8'` → `NaN`, `'NM'` → `NaN`.
- **Outlier Filtering**: Hard rules applied (e.g., Age 18–80, Interest Rate ≤ 100, Credit History ≤ Age × 12).
- **Winsorization**: 1st–99th percentile clipping on 11 numerical columns to suppress extreme outliers.

### 2. 📊 Exploratory Data Analysis (EDA)

- **Target Distribution**: Class balance check with imbalance ratio.
- **KDE Plots**: Distribution per class for 8 key numerical features.
- **Correlation Heatmap**: Pairwise correlation among all numerical features.
- **Stacked Bar Charts**: Relationship between categorical features and credit score.
- **Kruskal-Wallis Test**: Statistical significance of each feature across classes.

### 3. ⚙️ Feature Engineering (Custom Transformer: `CFE`)

11 domain-informed features are created inside a scikit-learn compatible custom transformer:

| Feature              | Formula / Logic                                    |
|----------------------|----------------------------------------------------|
| `Debt_to_Income`     | `Outstanding_Debt / (Annual_Income + 1)`           |
| `EMI_to_Salary`      | `Total_EMI_per_month / (Monthly_Inhand_Salary + 1)`|
| `Debt_per_Loan`      | `Outstanding_Debt / (Num_of_Loan + 1)`             |
| `Credit_Risk_Score`  | `Credit_Utilization_Ratio × Interest_Rate`         |
| `Delay_per_History`  | `Num_of_Delayed_Payment / (Credit_History_Age + 1)`|
| `Delay_Ratio`        | `Delay_from_due_date / (Credit_History_Age + 1)`   |
| `Investment_Rate`    | `Amount_invested_monthly / (Monthly_Inhand_Salary + 1)` |
| `Savings_after_EMI`  | `Monthly_Balance - Total_EMI_per_month`            |
| `Inquiry_per_History`| `Num_Credit_Inquiries / (Credit_History_Age + 1)`  |
| `Loan_Diversity`     | `Type_of_Loan_Count / (Num_of_Loan + 1)`           |
| `Income_Stability`   | `(Monthly_Inhand_Salary × 12) / (Annual_Income + 1)` |

---

## 🤖 Models & Hyperparameter Tuning

Five models are benchmarked, each wrapped in a dedicated scikit-learn `Pipeline`:

| Model                    | Preprocessing Strategy                        |
|--------------------------|-----------------------------------------------|
| `RandomForestClassifier` | Impute → Scale → OneHotEncode                 |
| `XGBClassifier`          | Impute → Scale → OneHotEncode                 |
| `LGBMClassifier`         | Impute → Scale → OneHotEncode                 |
| `CatBoostClassifier`     | Impute only (native categorical support)      |
| `HistGradientBoosting`   | Impute → OrdinalEncode (missing value native) |

**Tuning Strategy:**

- **Method**: `RandomizedSearchCV` with 30 iterations per model
- **Cross-Validation**: `StratifiedKFold` (5 folds, shuffled)
- **Scoring Metric**: `roc_auc_ovr` (One-vs-Rest AUC)
- **Parallelization**: `n_jobs=-1`

---

## 📈 Evaluation

The best model is selected based on **ROC AUC (OVR)** on the test set and evaluated using:

- Accuracy, Precision, Recall, F1 Macro
- ROC AUC (multi-class OVR)
- Classification Report (per-class breakdown)
- Confusion Matrix
- Error Pattern Analysis (which classes get misclassified into which)

---

## 🔍 Feature Importance & SHAP

### Feature Importance
Top 10 features from the best model's `feature_importances_` attribute, normalized and visualized as a horizontal bar chart.

### SHAP Analysis
- **Explainer**: `shap.TreeExplainer` (falls back to `KernelExplainer` if needed)
- **Sample**: 500 random test samples
- **Plots**:
  - **Beeswarm (dot)**: Feature impact distribution for the `Poor` class
  - **Bar**: Mean absolute SHAP values (top 10 features) for the `Poor` class

---

## 📦 Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost lightgbm catboost shap joblib
```

> Python 3.8+ recommended.

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/werrenedbert06/Credit-Score-Classification.git
   cd Credit-Score-Classification
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place the dataset**
   
   Ensure `Credit Score.csv` is in the root directory.

4. **Run the notebook**
   ```bash
   jupyter notebook credit_score_notebook.ipynb
   ```

---

## 💾 Outputs

After running the full notebook, the following files will be saved to the `outputs/` directory:

| File                        | Description                                  |
|-----------------------------|----------------------------------------------|
| `LGBM_credit_score.joblib`  | Best trained model (serialized with joblib)  |
| `feature_importance.csv`    | Top 10 normalized feature importance scores  |
| `predictions.csv`           | Test set predictions vs actual labels        |

---

## 🧠 Key Design Decisions

- **Custom Transformer (`CFE`)**: Integrates feature engineering directly into the scikit-learn pipeline, preventing data leakage.
- **Separate Pipelines per Model**: Each model has a tailored preprocessing strategy (e.g., CatBoost handles categoricals natively).
- **Stratified CV**: Ensures each fold maintains class distribution, critical for imbalanced targets.
- **Winsorization over Dropping**: Preserves data volume while controlling extreme outlier influence.

---

## 📄 License

This project is for educational and portfolio purposes. Feel free to fork and adapt.
