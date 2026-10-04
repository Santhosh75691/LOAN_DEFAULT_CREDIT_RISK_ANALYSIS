# ============================================================
# LOAN DEFAULT & CREDIT RISK ANALYSIS
# COMPLETE BEGINNER PROJECT
# ============================================================
# This code:
# 1. Loads your original CSV
# 2. Cleans the data
# 3. Automatically creates a CLEANED CSV
# 4. Performs KPI and risk analysis
# 5. Creates charts
# 6. Builds a Logistic Regression model
#
# Keep this Python file and the original CSV in the same folder.
# ============================================================

# Install libraries once if needed:
# pip install pandas numpy matplotlib seaborn scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# STEP 1: LOAD ORIGINAL DATA
# ============================================================

INPUT_FILE = "loan_default_credit_risk_dataset.csv"
CLEANED_FILE = "loan_default_credit_risk_dataset_cleaned.csv"

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("LOAN DEFAULT & CREDIT RISK ANALYSIS")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nOriginal columns:")
print(df.columns.tolist())

# ============================================================
# STEP 2: DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Remove duplicate rows
duplicate_count = df.duplicated().sum()
df = df.drop_duplicates().copy()

print("Duplicate rows removed:", duplicate_count)

# Clean column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# Numeric columns
numeric_cols = [
    "Age",
    "Employment_Years",
    "Annual_Income",
    "Credit_Score",
    "Num_Credit_Lines",
    "Num_Open_Accounts",
    "Late_Payments",
    "Debt_to_Income",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "EMI",
    "Default_Flag"
]

# Convert numeric columns to numbers
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Clean text columns
text_cols = df.select_dtypes(include="object").columns

for col in text_cols:
    df[col] = df[col].astype("string").str.strip()

# Check missing values before fixing
print("\nMissing values before fixing:")
print(df.isnull().sum())

# Fill numeric missing values with median
for col in numeric_cols:
    if col in df.columns and df[col].isna().any():
        df[col] = df[col].fillna(df[col].median())

# Fill text missing values with mode
for col in text_cols:
    if df[col].isna().any():
        mode = df[col].mode(dropna=True)
        if len(mode) > 0:
            df[col] = df[col].fillna(mode.iloc[0])
        else:
            df[col] = df[col].fillna("Unknown")

# Make Default_Flag 0/1
df["Default_Flag"] = df["Default_Flag"].round().astype(int)
df["Default_Flag"] = df["Default_Flag"].clip(0, 1)

# Recreate Default_Status from Default_Flag
df["Default_Status"] = df["Default_Flag"].map({
    0: "Paid",
    1: "Default"
})

# ============================================================
# STEP 3: CREATE CREDIT RISK BANDS
# ============================================================

def credit_band(score):
    if score < 580:
        return "Poor"
    elif score < 670:
        return "Fair"
    elif score < 740:
        return "Good"
    else:
        return "Very Good/Excellent"

def dti_band(dti):
    if dti < 0.20:
        return "Low DTI"
    elif dti < 0.40:
        return "Medium DTI"
    elif dti < 0.60:
        return "High DTI"
    else:
        return "Very High DTI"

df["Credit_Risk_Band"] = df["Credit_Score"].apply(credit_band)
df["DTI_Risk_Band"] = df["Debt_to_Income"].apply(dti_band)

# ============================================================
# STEP 4: SAVE CLEANED DATASET
# ============================================================

df.to_csv(CLEANED_FILE, index=False)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())

print("\nCleaned dataset shape:")
print(df.shape)

print("\nCLEANED DATASET SAVED AS:")
print(CLEANED_FILE)

# ============================================================
# STEP 5: KEY KPIs
# ============================================================

total_loans = len(df)
total_loan_amount = df["Loan_Amount"].sum()
average_loan_amount = df["Loan_Amount"].mean()
average_credit_score = df["Credit_Score"].mean()
defaulted_loans = df["Default_Flag"].sum()
default_rate = df["Default_Flag"].mean() * 100
average_interest_rate = df["Interest_Rate"].mean()

print("\n" + "=" * 70)
print("KEY BUSINESS KPIs")
print("=" * 70)

print("Total Loans           :", total_loans)
print("Total Loan Amount     : ₹{:,.0f}".format(total_loan_amount))
print("Average Loan Amount   : ₹{:,.2f}".format(average_loan_amount))
print("Average Credit Score  : {:.2f}".format(average_credit_score))
print("Defaulted Loans       :", defaulted_loans)
print("Default Rate          : {:.2f}%".format(default_rate))
print("Average Interest Rate : {:.2f}%".format(average_interest_rate))

# ============================================================
# STEP 6: DEFAULT RATE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("DEFAULT RATE ANALYSIS")
print("=" * 70)

credit_default = (
    df.groupby("Credit_Risk_Band")["Default_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

purpose_default = (
    df.groupby("Loan_Purpose")["Default_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

location_default = (
    df.groupby("Location")["Default_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

employment_default = (
    df.groupby("Employment_Type")["Default_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

dti_default = (
    df.groupby("DTI_Risk_Band")["Default_Flag"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

print("\nDefault Rate by Credit Risk Band:")
print(credit_default)

print("\nDefault Rate by Loan Purpose:")
print(purpose_default)

print("\nDefault Rate by Location:")
print(location_default)

print("\nDefault Rate by Employment Type:")
print(employment_default)

print("\nDefault Rate by DTI Risk Band:")
print(dti_default)

# ============================================================
# STEP 7: VISUALIZATIONS
# ============================================================

sns.set_theme(style="whitegrid")

# 1. Default Status
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Default_Status")
plt.title("Loan Default Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Loans")
plt.tight_layout()
plt.show()

# 2. Default Rate by Credit Risk Band
risk_order = [
    "Poor",
    "Fair",
    "Good",
    "Very Good/Excellent"
]

plt.figure(figsize=(9, 5))
plot_order = [x for x in risk_order if x in credit_default.index]

sns.barplot(
    x=plot_order,
    y=[credit_default[x] for x in plot_order]
)

plt.title("Default Rate by Credit Risk Band")
plt.xlabel("Credit Risk Band")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 3. Default Rate by Loan Purpose
plt.figure(figsize=(9, 5))

sns.barplot(
    x=purpose_default.index,
    y=purpose_default.values
)

plt.title("Default Rate by Loan Purpose")
plt.xlabel("Loan Purpose")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=25)
plt.tight_layout()
plt.show()

# 4. Default Rate by Location
plt.figure(figsize=(8, 5))

sns.barplot(
    x=location_default.index,
    y=location_default.values
)

plt.title("Default Rate by Location")
plt.xlabel("Location")
plt.ylabel("Default Rate (%)")
plt.tight_layout()
plt.show()

# 5. Credit Score Distribution
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Credit_Score",
    hue="Default_Status",
    bins=25,
    kde=True
)

plt.title("Credit Score Distribution by Default Status")
plt.xlabel("Credit Score")
plt.ylabel("Number of Loans")
plt.tight_layout()
plt.show()

# 6. Income vs Loan Amount
plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Loan_Amount",
    hue="Default_Status",
    alpha=0.7
)

plt.title("Annual Income vs Loan Amount")
plt.xlabel("Annual Income")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.show()

# 7. DTI vs Default Status
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Default_Status",
    y="Debt_to_Income"
)

plt.title("Debt-to-Income Ratio by Default Status")
plt.xlabel("Loan Status")
plt.ylabel("Debt-to-Income Ratio")
plt.tight_layout()
plt.show()

# 8. Late Payments vs Default
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Default_Status",
    y="Late_Payments"
)

plt.title("Late Payments by Default Status")
plt.xlabel("Loan Status")
plt.ylabel("Number of Late Payments")
plt.tight_layout()
plt.show()

# 9. Loan Amount Distribution
plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="Loan_Amount",
    bins=30,
    kde=True
)

plt.title("Loan Amount Distribution")
plt.xlabel("Loan Amount")
plt.ylabel("Number of Loans")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 8: CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    "Age",
    "Employment_Years",
    "Annual_Income",
    "Credit_Score",
    "Num_Credit_Lines",
    "Num_Open_Accounts",
    "Late_Payments",
    "Debt_to_Income",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "EMI",
    "Default_Flag"
]

plt.figure(figsize=(12, 9))

sns.heatmap(
    df[correlation_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ============================================================
# STEP 9: MACHINE LEARNING
# ============================================================

print("\n" + "=" * 70)
print("MACHINE LEARNING - LOGISTIC REGRESSION")
print("=" * 70)

features = [
    "Age",
    "Employment_Years",
    "Annual_Income",
    "Home_Ownership",
    "Employment_Type",
    "Location",
    "Credit_Score",
    "Num_Credit_Lines",
    "Num_Open_Accounts",
    "Late_Payments",
    "Debt_to_Income",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate",
    "Loan_Purpose",
    "Application_Channel",
    "Education"
]

X = df[features]
y = df["Default_Flag"]

numeric_features = [
    "Age",
    "Employment_Years",
    "Annual_Income",
    "Credit_Score",
    "Num_Credit_Lines",
    "Num_Open_Accounts",
    "Late_Payments",
    "Debt_to_Income",
    "Loan_Amount",
    "Loan_Term_Months",
    "Interest_Rate"
]

categorical_features = [
    "Home_Ownership",
    "Employment_Type",
    "Location",
    "Loan_Purpose",
    "Application_Channel",
    "Education"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

probability = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, probability)

print("\nMODEL RESULTS")
print("Accuracy : {:.2%}".format(accuracy))
print("ROC-AUC  : {:.3f}".format(auc))

print("\nClassification Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

# ============================================================
# STEP 10: FINAL BUSINESS INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL BUSINESS INSIGHTS")
print("=" * 70)

print(
    "1. Overall loan default rate: {:.2f}%".format(default_rate)
)

print(
    "2. Highest-risk credit band:",
    credit_default.idxmax()
)

print(
    "3. Highest-risk loan purpose:",
    purpose_default.idxmax()
)

print(
    "4. Highest-risk location:",
    location_default.idxmax()
)

print(
    "5. Highest-risk employment type:",
    employment_default.idxmax()
)

print(
    "6. Highest-risk DTI band:",
    dti_default.idxmax()
)

print("\nCleaned CSV created successfully:")
print(CLEANED_FILE)

print("\nPROJECT COMPLETED SUCCESSFULLY!")
