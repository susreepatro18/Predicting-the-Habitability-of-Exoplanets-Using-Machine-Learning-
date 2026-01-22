import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Configuration
# -------------------------------
INPUT_FILE = os.path.join("outputs", "merged_dataset.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📌 Module 2: Data Cleaning & Feature Engineering\n")

# -------------------------------
# Step 1: Load merged dataset
# -------------------------------
df = pd.read_csv(INPUT_FILE)
print("Dataset loaded:", df.shape)

# -------------------------------
# Step 2: Handle missing values
# -------------------------------
print("\nHandling missing values...")

numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# Fill numerical columns with median
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Fill categorical columns with mode
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values handled.")

# -------------------------------
# Step 3: Outlier handling (IQR method)
# -------------------------------
print("\nHandling outliers using IQR method...")

for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

print("Outliers capped.")

# -------------------------------
# Step 4: Encode categorical features (One-Hot Encoding)
# -------------------------------
print("\nEncoding categorical features (st_spectype)...")

df = pd.get_dummies(df, columns=["st_spectype"], drop_first=True)

print("Categorical encoding completed.")

# -------------------------------
# Step 5: Habitability Score Index (HSI)
# -------------------------------
print("\nCreating Habitability Score Index...")

df["habitability_score"] = (
    (1 / (1 + abs(df["pl_rade"] - 1))) * 0.25 +
    (1 / (1 + abs(df["pl_eqt"] - 288))) * 0.25 +
    (1 / (1 + abs(df["pl_insol"] - 1))) * 0.25 +
    (1 / (1 + abs(df["pl_orbeccen"]))) * 0.25
)

print("Habitability Score Index created.")

# -------------------------------
# Step 6: Stellar Compatibility Index (SCI)
# -------------------------------
print("\nCreating Stellar Compatibility Index...")

df["stellar_compatibility"] = (
    (1 / (1 + abs(df["st_teff"] - 5778))) * 0.6 +
    (1 / (1 + abs(df["st_mass"] - 1))) * 0.4
)

print("Stellar Compatibility Index created.")

# -------------------------------
# Step 7: Normalize numerical features
# -------------------------------
print("\nNormalizing numerical features...")

scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("Normalization completed.")

# -------------------------------
# Step 8: Data validation using statistics
# -------------------------------
print("\nSaving descriptive statistics...")

stats_file = os.path.join(OUTPUT_DIR, "module2_statistics.txt")
with open(stats_file, "w") as f:
    f.write(str(df.describe()))

print("Statistics saved.")

# -------------------------------
# Step 9: Data validation using visualization
# -------------------------------
print("\nGenerating validation visualizations...")

# Missing values heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "missing_values_heatmap.png"))
plt.close()

# Distribution of Habitability Score
plt.figure(figsize=(6, 4))
sns.histplot(df["habitability_score"], bins=30, kde=True)
plt.title("Habitability Score Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "habitability_score_distribution.png"))
plt.close()

print("Visualizations saved.")

# -------------------------------
# Step 10: Save cleaned dataset
# -------------------------------
cleaned_file = os.path.join(OUTPUT_DIR, "cleaned_feature_engineered_dataset.csv")
df.to_csv(cleaned_file, index=False)

print("\nCleaned dataset saved to:", cleaned_file)
print("\n✅ Module 2: Data Cleaning & Feature Engineering COMPLETED SUCCESSFULLY")
