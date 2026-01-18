from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import json
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder="static", template_folder="templates")

CORS(app)

print("🔥 Backend running (prediction + ranking) 🔥")

# =============================
# LOAD MODEL
# =============================
rf_classifier = joblib.load("models/rf_classifier.pkl")

# =============================
# LOAD FEATURES
# =============================
with open("features/features_classification.json", "r") as f:
    FEATURES = json.load(f)

# =============================
# PAGES
# =============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/visualization")
def visualization():
    return render_template("visualization.html")

# =============================
# PREDICT API
# =============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON input"}), 400

        radius = float(data.get("planet_radius", 0))
        mass = float(data.get("planet_mass", 0))
        temperature = float(data.get("surface_temperature", 0))

        density = mass / (radius ** 3) if radius > 0 else 0

        feature_vector = []
        for f in FEATURES:
            if f == "planet_radius":
                feature_vector.append(radius)
            elif f == "planet_mass":
                feature_vector.append(mass)
            elif f == "density":
                feature_vector.append(density)
            elif f == "surface_temperature":
                feature_vector.append(temperature)
            else:
                feature_vector.append(0)

        X = np.array(feature_vector).reshape(1, -1)

        ml_prob = float(rf_classifier.predict_proba(X)[0][1])

        temp_score = 1 if 250 <= temperature <= 320 else max(0, 1 - abs(temperature - 288) / 500)
        density_score = max(0, 1 - abs(density - 1))

        habitability_score = (
            0.5 * ml_prob +
            0.25 * temp_score +
            0.25 * density_score
        )

        habitability_class = "Habitable" if habitability_score >= 0.5 else "Non-Habitable"

        return jsonify({
            "habitability_class": habitability_class,
            "habitability_score": round(habitability_score, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================
# RANKING API
# =============================
@app.route("/ranking", methods=["GET"])
def ranking():
    try:
        df = pd.read_csv("data/top_10_habitable_exoplanets.csv")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =============================
# RUN
# =============================
if __name__ == "__main__":
    app.run(debug=True)
