from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")

EXPECTED_FEATURES = [
    "temperature_C","DO_mg_per_L","pH","turbidity","BOD","conductivity_uS_cm",
    "ammonia_ppm","nitrate_mg_L","hardness_mg_L_CaCO3","COD",
    "CO2_ppm","alkalinity_mg_L","activity_level"
]

@app.route("/", methods=["GET", "POST"])
def dashboard():
    prediction = None
    if request.method == "POST":
        # Collect input values from form and convert to float
        X = [float(request.form[f]) for f in EXPECTED_FEATURES]
        X_scaled = scaler.transform([X])
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0,1]
        # Determine risk level color
        if prob >= 0.7:
            risk_color = "red"
            risk_text = "High Risk"
        elif prob >= 0.4:
            risk_color = "orange"
            risk_text = "Moderate Risk"
        else:
            risk_color = "green"
            risk_text = "Low Risk"
        prediction = {
            "label": int(pred),
            "prob": float(prob),
            "risk_color": risk_color,
            "risk_text": risk_text
        }
    return render_template("dashboard.html", features=EXPECTED_FEATURES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
