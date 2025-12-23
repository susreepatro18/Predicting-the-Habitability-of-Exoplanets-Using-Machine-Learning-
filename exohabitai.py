# =========================================================
# ExoHabitAI: Predicting Habitability of Exoplanets
# Modules 1 to 4 (NASA Exoplanet CSV – FIXED VERSION)
# =========================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# MODULE 1: DATA COLLECTION & MANAGEMENT
# ---------------------------------------------------------

print("\nModule 1: Data Collection & Management")

FILE_NAME = "nasa_exoplanets.csv"

df = pd.read_csv(
    FILE_NAME,
    comment="#",      # Ignore NASA metadata lines
    sep=",",
    engine="python"   # Handles complex CSV safely
)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())
print(df.columns.tolist())

# ---------------------------------------------------------
# MODULE 2: DATA CLEANING & FEATURE ENGINEERING
# ---------------------------------------------------------

print("\nModule 2: Data Cleaning & Feature Engineering")

# Required features for habitability analysis
required_features = [
    'pl_rade',       # Planet radius (Earth radius)
    'pl_bmasse',     # Planet mass (Earth mass)
    'pl_orbper',     # Orbital period (days)
    'pl_eqt',        # Equilibrium temperature (K)
    'pl_insol',      # Insolation flux
    'pl_orbeccen',   # Orbital eccentricity
    'st_teff',       # Stellar effective temperature
    'st_rad',        # Stellar radius
    'st_mass',       # Stellar mass
    'st_met',        # Stellar metallicity
    'st_spectype'    # Stellar spectral type (categorical)
]

# Select only available columns (prevents KeyError)
available_features = [col for col in required_features if col in df.columns]

df_model = df[available_features].copy()

print("Selected features:", available_features)

# ------------------------------
# Habitability Score (Rule-Based)
# ------------------------------
def habitability_score(row):
    score = 0

    if pd.notna(row.get('pl_rade')) and 0.5 <= row['pl_rade'] <= 2.0:
        score += 1
    if pd.notna(row.get('pl_eqt')) and 200 <= row['pl_eqt'] <= 350:
        score += 1
    if pd.notna(row.get('pl_insol')) and 0.5 <= row['pl_insol'] <= 2.0:
        score += 1
    if pd.notna(row.get('pl_orbeccen')) and row['pl_orbeccen'] <= 0.3:
        score += 1

    return score

df_model['habitability_score'] = df_model.apply(habitability_score, axis=1)

# Binary classification target
df_model['habitable'] = np.where(df_model['habitability_score'] >= 3, 1, 0)

print("Target variable created (habitable / non-habitable)")


# ---------------------------------------------------------
# MODULE 3: MACHINE LEARNING DATASET PREPARATION
# ---------------------------------------------------------

print("\nModule 3: ML Dataset Preparation")

X = df_model.drop(['habitability_score', 'habitable'], axis=1)
y = df_model['habitable']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = ['st_spectype'] if 'st_spectype' in X.columns else []

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed")

# ---------------------------------------------------------
# MODULE 4: AI MODEL FOR HABITABILITY PREDICTION
# ---------------------------------------------------------

print("\nModule 4: Model Training & Evaluation")

# ------------------
# Random Forest Model
# ------------------
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    ))
])

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print(classification_report(y_test, y_pred_rf))

# ------------------------
# Logistic Regression Model
# ------------------------
lr_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
y_prob_lr = lr_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_lr))
print(classification_report(y_test, y_pred_lr))

# ---------------------------------------------------------
# RANK EXOPLANETS BY PREDICTED HABITABILITY
# ---------------------------------------------------------

df_ranked = df[['pl_name']].copy()
df_ranked['Habitability_Probability'] = rf_model.predict_proba(X)[:, 1]

df_ranked = df_ranked.sort_values(
    by='Habitability_Probability',
    ascending=False
)

print("\nTop 10 Potentially Habitable Exoplanets:")
print(df_ranked.head(10))

import joblib

joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(X_test, "X_test.pkl")
joblib.dump(y_test, "y_test.pkl")

print("Model and test data saved successfully")
