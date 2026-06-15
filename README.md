🏦 Credit Score Classification

A comprehensive end-to-end machine learning project for classifying credit scores into Poor, Standard, and Good categories based on financial and behavioral data.
📋 Table of Contents

    Overview

    Dataset

    Project Structure

    Pipeline Walkthrough

    Models & Hyperparameter Tuning

    Evaluation

    Feature Importance Analysis

    Requirements

    How to Run

📌 Overview

This project solves a multi-class classification problem to predict an individual's credit score category using financial behavior, demographics, and credit history. The pipeline covers data ingestion, cleaning, EDA, preprocessing, modeling, and evaluation.

Target Classes:

    Poor (0): High-risk credit profile

    Standard (1): Average credit profile

    Good (2): Low-risk, healthy credit profile

📂 Dataset
Property	Details
File	data_C.csv
Delimiter	,
Rows	25,000
Columns (raw)	29
Target Column	Credit_Score

Key Features:

    Demographics: Age, Occupation, Annual Income, Monthly Inhand Salary

    Financial: Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan

    Behavioral: Delay_from_due_date, Num_of_Delayed_Payment, Payment_Behaviour

    Credit History: Credit_History_Age, Credit_Mix, Payment_of_Min_Amount

    Debt & Balance: Outstanding_Debt, Credit_Utilization_Ratio, Monthly_Balance

Columns Dropped (non-informative identifiers):
Unnamed: 0, ID, Customer_ID, Name, SSN
🗂 Project Structure
text

credit-score-classification/
│
├── data_C.csv              # Raw dataset
└── notebook.ipynb          # Main analysis notebook (all code included)

🔧 Pipeline Walkthrough
1. 🧹 Data Cleaning & Type Casting

    Type Casting: Strips non-numeric characters from columns like Age, Annual_Income, Num_of_Loan, Num_of_Delayed_Payment, Outstanding_Debt, Changed_Credit_Limit, Amount_invested_monthly and converts to numeric.

    Credit History Parsing: Converts string format (e.g., "17 Years and 11 Months") into total months.

    Noisy Value Replacement:

        '_______' → NaN (Occupation)

        '_' → NaN (Credit_Mix)

        'NM' → NaN (Payment_of_Min_Amount)

        '!@9#%8' → NaN (Payment_Behaviour)

    Type_of_Loan: Keeps only the first loan type, replaces missing with NaN.

    Outlier Handling: Hard caps applied (Age 18–100, Num_Bank_Accounts 0–20, Interest_Rate 0–100, etc.) and winsorization using 1st–99th percentiles.

2. 📊 Exploratory Data Analysis (EDA)

    Target Distribution:

        Standard: 13,282 (53.1%)

        Poor: 7,268 (29.1%)

        Good: 4,450 (17.8%)

        Imbalance detected → class_weight='balanced' used in modeling

    Correlation Analysis (Spearman):

        Positive correlation with Credit Score: Credit_History_Age_Months, Monthly_Inhand_Salary, Monthly_Balance

        Negative correlation: Outstanding_Debt, Interest_Rate, Num_of_Delayed_Payment

        Near-zero correlation: Credit_Utilization_Ratio, Total_EMI_per_month

    Distribution Checks: Histogram + Boxplot for all numerical features revealed right-skewed distributions with outliers → Median imputation chosen for robustness.

3. ⚙️ Preprocessing

Preprocessing Pipeline (using ColumnTransformer):
Feature Type	Columns	Transformations
Numerical	17 columns	SimpleImputer(strategy='median') → StandardScaler()
Ordinal	Credit_Mix, Payment_of_Min_Amount	SimpleImputer(strategy='most_frequent') → OrdinalEncoder(categories=[['Bad','Standard','Good'], ['No','Yes']], handle_unknown='use_encoded_value', unknown_value=-1)
Nominal	Month, Occupation, Type_of_Loan, Payment_Behaviour	SimpleImputer(strategy='most_frequent') → OneHotEncoder(handle_unknown='ignore', sparse_output=False)

Train-Test Split: 80% train, 20% test, stratified by target.
🤖 Models & Hyperparameter Tuning

Three models were benchmarked using 5-fold cross-validation with F1 Macro scoring (due to class imbalance).
Random Forest (Best Model)
Experiment	Parameters	CV F1 Mean
RF-5	n_estimators=200, criterion='entropy', class_weight='balanced'	0.6981
RF-3	n_estimators=300, criterion='gini'	0.6967
RF-2	n_estimators=200, criterion='gini'	0.6945
XGBoost
Experiment	Parameters	CV F1 Mean
XGB-4	n_estimators=100, max_depth=8, learning_rate=0.1	0.6868
XGB-2	n_estimators=200, max_depth=6, learning_rate=0.1	0.6859
HistGradientBoosting
Experiment	Parameters	CV F1 Mean
HGB-4	max_iter=200, learning_rate=0.05	0.6838

Best Model: Random Forest (RF-5) with CV F1 Macro = 0.6981
📈 Evaluation (Test Set)

Classification Report (RF-5):
Class	Precision	Recall	F1-score	Support
Poor	0.75	0.72	0.73	1,454
Standard	0.76	0.77	0.77	2,656
Good	0.62	0.62	0.62	890

    Accuracy: 0.73

    Macro F1: 0.71

    Weighted F1: 0.73

Confusion Matrix Insights:

    Poor → misclassified as Standard (328), rarely as Good

    Good → often misclassified as Standard (234)

    Standard → most stable predictions

🔍 Feature Importance Analysis

Top features by importance (Random Forest):

    Outstanding_Debt

    Annual_Income

    Interest_Rate

    Credit_History_Age_Months

    Num_of_Loan

    Age

    Num_of_Delayed_Payment

    Monthly_Inhand_Salary

    Credit_Mix (encoded)

    Amount_invested_monthly

Key Insights:

    High Outstanding_Debt and Interest_Rate push predictions toward Poor

    Low Annual_Income and Monthly_Inhand_Salary correlate with Poor classification

    Short credit history (Credit_History_Age_Months) contributes to Poor prediction

📦 Requirements
bash

pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost

    Python 3.8+ recommended.

▶️ How to Run

    Clone the repository
    bash

    git clone https://github.com/yourusername/credit-score-classification.git
    cd credit-score-classification

    Install dependencies
    bash

    pip install -r requirements.txt

    Place the dataset
    Ensure data_C.csv is in the root directory.

    Run the notebook
    bash

    jupyter notebook notebook.ipynb

📄 License

This project is for educational and portfolio purposes. Feel free to fork and adapt.
