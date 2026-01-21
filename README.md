# ExoHabitAI – Exoplanet Habitability Prediction System

ExoHabitAI is a machine learning-powered web application that predicts the habitability potential of exoplanets based on key planetary and stellar parameters.

It integrates a trained ML model, RESTful backend APIs, SQLite database, and an interactive frontend dashboard to deliver scientifically meaningful predictions in a clean, user-friendly interface.

Designed as a complete end-to-end project — ideal for final-year submissions, internships, academic evaluations, and applied AI / data science portfolios.

## Project Objectives

- Predict exoplanet habitability using supervised machine learning
- Provide real-time predictions through REST APIs
- Store and rank predicted planets in a persistent database
- Visualize model insights, feature importance, and data distributions
- Export analytical reports in PDF and Excel formats

## Key Features

### Habitability Prediction
- Random Forest Classifier
- Habitability probability score (0.0 – 1.0)
- Binary classification: Habitable / Non-habitable

### Interactive Dashboard
- Responsive, modern UI (Bootstrap 5 + Chart.js)
- Dynamic charts for input parameters
- Confidence indicators and classification labels

### Planet Ranking & Storage
- SQLite database for saving predictions
- Ranked list of exoplanets sorted by habitability score

### Analytics & Visualizations
- Feature importance bar chart
- Habitability score distribution histogram
- Correlation heatmap between planetary/stellar parameters

### Report Export
- PDF summary report of top-ranked planets
- Excel (.xlsx) export of prediction data

## Machine Learning Model

**Algorithm**  
Random Forest Classifier (scikit-learn)

**Preprocessing**  
StandardScaler

**Input Features**  
- Planet Mass (Earth masses)  
- Orbital Period (days)  
- Orbit Distance / Semi-major Axis (AU)  
- Star Effective Temperature (Kelvin)  
- Star Radius (Solar radii)

**Outputs**  
- Habitability probability (float 0–1)  
- Binary habitability class label

## Technology Stack

### Frontend
- HTML5 · CSS3 · JavaScript (ES6+)  
- Bootstrap 5  
- Chart.js

### Backend & API
- Python 3.8+  
- Flask  
- Flask-CORS  
- SQLite

### Data Science & Machine Learning
- NumPy · Pandas  
- scikit-learn  
- Matplotlib · Seaborn  
- Joblib (model & scaler serialization)

## Project Structure

- `app.py`                        → Main Flask application  
- `model/`  
  - `model.joblib`                → Trained Random Forest model  
  - `scaler.joblib`               → Fitted StandardScaler  
- `static/`  
  - `css/`                        → Custom styles  
  - `js/`                         → JavaScript & Chart.js scripts  
  - `images/`                     → Icons, logos, screenshots, etc.  
- `templates/`  
  - `index.html`                  → Main dashboard page  
- `notebooks/`  
  - `model_training.ipynb`        → Model development and evaluation notebook  
- `database/`  
  - `planets.db`                  → SQLite database  
- `requirements.txt`              → List of Python dependencies  
- `README.md`                     → This documentation file

## How to Run (Quick Start)

1. Clone the repository
2. Install dependencies  
   ```bash
   pip install -r requirements.txt
3. Run the Flask app
   ```bash
   python app.py
5. Open:  http://127.0.0.1:5000/

## Future Improvements (Optional Ideas)

Add more features (stellar metallicity, planet radius, eccentricity…)
Try advanced models: LightGBM, Neural Networks
Implement user authentication
Deploy on cloud (Render, Railway, Vercel + Flask backend)
Add exoplanet data fetching from NASA Exoplanet Archive API

## License
This project is developed for academic and educational purposes.
