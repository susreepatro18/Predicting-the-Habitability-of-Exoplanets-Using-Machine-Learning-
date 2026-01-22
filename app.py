from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import joblib
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")

import seaborn as sns
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from openpyxl import Workbook


app = Flask(__name__)

API_KEY = "SECRET123"
model = joblib.load("model/habitability_model.pkl")

# 🔴 MUST MATCH MODEL TRAINING FEATURES
FEATURES = [
    "pl_orbper",
    "pl_orbeccen",
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_rad",
    "st_mass",
    "st_lum",
    "sy_dist",
    "st_spectype",
    "discoverymethod"
]

# ---------------- DB ---------------- #

def get_db():
    return sqlite3.connect("database.db", check_same_thread=False)

def check_key(req):
    return req.headers.get("x-api-key") == API_KEY

# ---------------- ROUTES ---------------- #

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/predict-page")
def predict_page():
    return render_template("index.html")

# ---------------- PREDICT ---------------- #

@app.route("/predict", methods=["POST"])
def predict():
    if not check_key(request):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    # ✅ AUTO-FILL REQUIRED FEATURES
    X = pd.DataFrame([{
        "pl_orbper": data.get("pl_orbper", 365),
        "pl_orbeccen": data.get("pl_orbeccen", 0.016),
        "pl_rade": data["pl_rade"],
        "pl_bmasse": data["pl_bmasse"],
        "pl_eqt": data["pl_eqt"],
        "pl_insol": data.get("pl_insol", 1.0),
        "st_teff": data["st_teff"],
        "st_rad": data["st_rad"],
        "st_mass": data["st_mass"],
        "st_lum": data["st_lum"],
        "sy_dist": data["sy_dist"],
        "st_spectype": data.get("st_spectype", "G2V"),
        "discoverymethod": data.get("discoverymethod", "Transit")
    }])[FEATURES]

    score = float(model.predict_proba(X)[0][1])

    return jsonify({
        "habitability_score": round(score, 3),
        "habitability_prediction": 1 if score >= 0.4 else 0
    })

# ---------------- STORE ---------------- #

@app.route("/store", methods=["POST"])
def store():
    if not check_key(request):
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json

    # Prepare model input
    X = pd.DataFrame([data])[FEATURES]
    score = float(model.predict_proba(X)[0][1])

    con = get_db()
    cur = con.cursor()

    cur.execute("""
        INSERT INTO exoplanets (
            planet_name,
            pl_orbper, pl_orbeccen, pl_rade, pl_bmasse, pl_eqt, pl_insol,
            st_teff, st_rad, st_mass, st_lum,
            sy_dist,
            habitability_score
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        data["planet_name"],
        data["pl_orbper"], data["pl_orbeccen"], data["pl_rade"],
        data["pl_bmasse"], data["pl_eqt"], data["pl_insol"],
        data["st_teff"], data["st_rad"], data["st_mass"], data["st_lum"],
        data["sy_dist"],
        score
    ))

    con.commit()
    con.close()

    return jsonify({"status": "stored"})

# ---------------- RANKING ---------------- #

@app.route("/ranking")
def ranking():
    if not check_key(request):
        return jsonify({"error": "Unauthorized"}), 401

    con = get_db()
    df = pd.read_sql("""
        SELECT planet_name, habitability_score
        FROM exoplanets
        ORDER BY habitability_score DESC
    """, con)
    con.close()

    df["rank"] = df.index + 1
    return jsonify(df.to_dict(orient="records"))

#---------------dashboard--------------------#
@app.route("/dashboard")
def dashboard():
    con = get_db()
    df = pd.read_sql("""
        SELECT planet_name, habitability_score
        FROM exoplanets
        ORDER BY habitability_score DESC
    """, con)
    con.close()

    df["rank"] = df.index + 1
    ranking = df.to_dict(orient="records")

    return render_template(
        "dashboard.html",
        ranking=ranking
    )

# ---------------- DASHBOARD ANALYTICS ---------------- #

@app.route("/generate-dashboard")
def generate_dashboard():
    con = get_db()
    df = pd.read_sql("SELECT * FROM exoplanets", con)
    con.close()

    if df.empty:
        return "No data available for dashboard"

    os.makedirs("static/plots", exist_ok=True)

    # =============================
    # 1️⃣ Habitability Score Distribution
    # =============================
    plt.figure(figsize=(6,4))
    sns.histplot(df["habitability_score"], bins=10, kde=True)
    plt.title("Habitability Score Distribution")
    plt.xlabel("Habitability Score")
    plt.ylabel("Number of Planets")
    plt.tight_layout()
    plt.savefig("static/plots/score_distribution.png")
    plt.close()

    # =============================
    # 2️⃣ Top 10 Habitable Planets
    # =============================
    top = df.sort_values("habitability_score", ascending=False).head(10)

    plt.figure(figsize=(6,4))
    sns.barplot(
        x=top["habitability_score"],
        y=top["planet_name"]
    )
    plt.title("Top 10 Habitable Exoplanets")
    plt.xlabel("Habitability Score")
    plt.ylabel("Planet Name")
    plt.tight_layout()
    plt.savefig("static/plots/top_planets.png")
    plt.close()

    # =============================
    # 3️⃣ Correlation Heatmap (WORKING)
    # =============================
    numeric_df = df.drop(columns=["id", "planet_name"])

    plt.figure(figsize=(10,7))
    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        linewidths=0.5
    )

    plt.title("Star–Planet Parameter Correlation with Habitability")
    plt.tight_layout()
    plt.savefig("static/plots/correlation_heatmap.png")
    plt.close()

    return "Dashboard generated successfully"

@app.route("/export/pdf")
def export_pdf():
    con = get_db()
    df = pd.read_sql("""
        SELECT planet_name, habitability_score
        FROM exoplanets
        ORDER BY habitability_score DESC
        LIMIT 10
    """, con)
    con.close()

    file_path = "top_exoplanets.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(
        "Top Candidate Habitable Exoplanets", styles["Title"]
    ))

    table_data = [["Planet Name", "Habitability Score"]]
    for _, row in df.iterrows():
        table_data.append([
            row["planet_name"],
            f"{row['habitability_score']:.3f}"
        ])

    table = Table(table_data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.cyan),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))

    elements.append(table)
    doc.build(elements)

    return send_file(file_path, as_attachment=True)

@app.route("/export/excel")
def export_excel():
    con = get_db()
    df = pd.read_sql("""
        SELECT planet_name, habitability_score
        FROM exoplanets
        ORDER BY habitability_score DESC
        LIMIT 10
    """, con)
    con.close()

    file_path = "top_exoplanets.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "Top Exoplanets"

    ws.append(["Planet Name", "Habitability Score"])

    for _, row in df.iterrows():
        ws.append([
            row["planet_name"],
            round(row["habitability_score"], 3)
        ])

    wb.save(file_path)
    return send_file(file_path, as_attachment=True)


# ---------------- RUN ---------------- #

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
