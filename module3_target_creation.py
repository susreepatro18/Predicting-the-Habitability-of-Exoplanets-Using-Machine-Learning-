import pandas as pd
import numpy as np

# Load merged dataset
df = pd.read_csv("outputs/merged_dataset.csv")

# -----------------------------
# HABITABILITY LOGIC
# -----------------------------
df["habitability"] = np.where(
    (df["pl_eqt"].between(180, 300)) &
    (df["pl_rade"] <= 2.0) &
    (df["pl_insol"].between(0.25, 2.0)),
    1,
    0
)

print("\nHabitability Distribution:")
print(df["habitability"].value_counts())

# Save updated dataset
df.to_csv("outputs/merged_with_target.csv", index=False)

print("\n✅ Target column created successfully")
