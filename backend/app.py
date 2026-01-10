from flask import Flask, jsonify, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return jsonify({"message": "ExoHabitAI API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = [
        data["planet_mass_earth"],
        data["orbital_period_days"],
        data["orbit_distance_au"],
        data["star_temperature_k"],
        data["star_radius_solar"]
    ]

    features_scaled = scaler.transform([features])
    prob = model.predict_proba(features_scaled)[0][1]

    return jsonify({
        "habitability_score": round(prob, 4),
        "habitable": int(prob >= 0.5)
    })
import sqlite3

@app.route("/add_planet", methods=["POST"])
def add_planet():
    data = request.json

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO exoplanets (name, score) VALUES (?, ?)",
        (data["name"], data["score"])
    )

    conn.commit()
    conn.close()

    return jsonify({"message": "Planet saved successfully"})
@app.route("/ranking", methods=["GET"])
def ranking():
    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute(
        "SELECT name, score FROM exoplanets ORDER BY score DESC LIMIT 10"
    )

    rows = cur.fetchall()
    conn.close()

    return jsonify([
        {"name": r[0], "score": r[1]} for r in rows
    ])

@app.route("/routes")
def routes():
    return jsonify([str(r) for r in app.url_map.iter_rules()])

if __name__ == "__main__":
    app.run(debug=True)
