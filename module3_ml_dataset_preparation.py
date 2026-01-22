import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------------
# Configuration
# -------------------------------
INPUT_FILE = os.path.join("outputs", "cleaned_feature_engineered_dataset.csv")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n📌 Module 3: Machine Learning Dataset Preparation\n")

# -------------------------------
# Step 1: Load cleaned dataset
# -------------------------------
df = pd.read_csv(INPUT_FILE)
print("Dataset loaded:", df.shape)

# -------------------------------
# Step 2: Define target variable
# -------------------------------
print("\nDefining target variable (Habitability Class)...")

# Binary classification based on habitability score
df["habitability_class"] = np.where(
    df["habitability_score"] >= df["habitability_score"].median(),
    1,  # Habitable
    0   # Non-Habitable
)

print("Target variable created.")

# -------------------------------
# Step 3: Feature selection based on domain relevance
# -------------------------------
print("\nSelecting relevant features...")

target = "habitability_class"

# Drop non-ML columns
drop_cols = [
    target,
    "habitability_score"  # Used only for labeling
]

X = df.drop(columns=drop_cols)
y = df[target]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# -------------------------------
# Step 4: Identify numerical & categorical features
# -------------------------------
numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["uint8", "bool"]).columns.tolist()

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

# -------------------------------
# Step 5: Preprocessing pipelines
# -------------------------------
print("\nCreating preprocessing pipelines...")

numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("feature_selection", SelectKBest(score_func=f_classif, k=15))
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

print("Pipelines created.")

# -------------------------------
# Step 6: Train-test split (80:20)
# -------------------------------
print("\nSplitting dataset (80:20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# -------------------------------
# Step 7: Apply preprocessing
# -------------------------------
print("\nApplying preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)

# -------------------------------
# Step 8: Save prepared datasets
# -------------------------------
print("\nSaving prepared datasets...")

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train_processed)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test_processed)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train.values)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test.values)

print("Datasets saved.")

print("\n✅ Module 3: Machine Learning Dataset Preparation COMPLETED SUCCESSFULLY")
