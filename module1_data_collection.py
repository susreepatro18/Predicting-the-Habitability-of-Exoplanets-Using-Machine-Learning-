import pandas as pd
import os

# -------------------------------
# Configuration
# -------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

FILE_1 = os.path.join(DATA_DIR, "nasa_exoplanets.csv")
FILE_2 = os.path.join(DATA_DIR, "exoplanets_dataset.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Step 1: Load datasets
# -------------------------------
print("Loading datasets...\n")

# NASA dataset contains metadata lines starting with '#'
df1 = pd.read_csv(
    FILE_1,
    comment="#",
    engine="python"
)

df2 = pd.read_csv(FILE_2)

print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)

# -------------------------------
# Step 2: Select common features
# -------------------------------
COMMON_FEATURES = [
    'pl_orbper',
    'pl_rade',
    'pl_bmasse',
    'pl_orbeccen',
    'pl_insol',
    'pl_eqt',
    'st_teff',
    'st_mass',
    'st_spectype'
]

df1_selected = df1[COMMON_FEATURES]
df2_selected = df2[COMMON_FEATURES]

# -------------------------------
# Step 3: Merge datasets (row-wise)
# -------------------------------
merged_df = pd.concat(
    [df1_selected, df2_selected],
    ignore_index=True
)

print("\nMerged dataset shape (before duplicates):", merged_df.shape)

# -------------------------------
# Step 4: Remove duplicate rows
# -------------------------------
before = merged_df.shape[0]
merged_df.drop_duplicates(inplace=True)
after = merged_df.shape[0]

print(f"Removed {before - after} duplicate rows")

# -------------------------------
# Step 5: Data validation
# -------------------------------
summary = []
summary.append(f"Final Dataset Shape: {merged_df.shape}\n")

summary.append("Column-wise Missing Values:\n")
summary.append(str(merged_df.isnull().sum()))
summary.append("\n")

summary.append("Basic Statistics (Numerical Columns):\n")
summary.append(str(merged_df.describe()))

with open(os.path.join(OUTPUT_DIR, "data_summary.txt"), "w") as f:
    f.write("\n".join(summary))

# -------------------------------
# Step 6: Save merged dataset
# -------------------------------
merged_file_path = os.path.join(OUTPUT_DIR, "merged_dataset.csv")
merged_df.to_csv(merged_file_path, index=False)

print("\nMerged dataset saved to:", merged_file_path)
print("Data summary saved to outputs/data_summary.txt")

print("\n✅ Module 1: Data Collection & Management COMPLETED SUCCESSFULLY")
