"""
Customer Churn - Exploratory Data Analysis (EDA)
Run: python eda.py
Generates PNG charts in the 'eda_charts' folder for your report.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/churn.csv"
OUTPUT_DIR = "eda_charts"

os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------
# 1. Overall churn rate (class imbalance)
# ---------------------------------------------------------
plt.figure(figsize=(5, 4))
df["Churn"].value_counts().plot(kind="bar", color=["#4C72B0", "#DD8452"])
plt.title("Overall Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/1_churn_distribution.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 2. Churn rate by contract type
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/2_churn_by_contract.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 3. Tenure distribution vs churn
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
             palette=["#4C72B0", "#DD8452"], bins=30)
plt.title("Tenure Distribution by Churn")
plt.xlabel("Tenure (months)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/3_tenure_vs_churn.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 4. Monthly charges vs churn
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#4C72B0", "#DD8452"])
plt.title("Monthly Charges by Churn Status")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/4_monthlycharges_vs_churn.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 5. Churn rate by Internet Service type
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="InternetService", hue="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn by Internet Service Type")
plt.xlabel("Internet Service")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/5_churn_by_internet_service.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 6. Churn rate by Payment Method
# ---------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.countplot(data=df, y="PaymentMethod", hue="Churn", palette=["#4C72B0", "#DD8452"])
plt.title("Churn by Payment Method")
plt.xlabel("Count")
plt.ylabel("Payment Method")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/6_churn_by_payment_method.png", dpi=150)
plt.close()

# ---------------------------------------------------------
# 7. Correlation heatmap (numeric features)
# ---------------------------------------------------------
numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()
numeric_df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

plt.figure(figsize=(5, 4))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/7_correlation_heatmap.png", dpi=150)
plt.close()

print(f"\nAll charts saved to the '{OUTPUT_DIR}/' folder:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    print(f" - {f}")
