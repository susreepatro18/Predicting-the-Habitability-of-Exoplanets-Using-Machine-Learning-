import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load saved objects
rf_model = joblib.load("rf_model.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")

# Predict probabilities
y_probs = rf_model.predict_proba(X_test)[:, 1]

# ROC calculation
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC–AUC Curve for Exoplanet Habitability Model")
plt.legend()
plt.show()
