import pandas as pd
import joblib

# Load the trained model and scaler
model = joblib.load("rf_model.joblib")
scaler = joblib.load("scaler.joblib")

# Example fish-farm data (you can adjust values)
sample = {
    "Temperature": [27.5],
    "DO": [6.8],
    "pH": [7.4],
    "Turbidity": [2.1],
    "BOD": [3.0],
    "Conductivity": [500],
    "Ammonia": [0.4],
    "Nitrate": [20],
    "Hardness": [150],
    "COD": [25],
    "CO2": [2.8],
    "Alkalinity": [200],
    "Activity_Level": [0.8],
    "No_of_Disease_Occurrences": [0],
    
}

df = pd.DataFrame(sample)

# Scale and predict
scaled = scaler.transform(df)
prediction = model.predict(scaled)[0]

if prediction == 1:
    print("⚠️  Fish are at HIGH risk of disease!")
else:
    print("✅ Fish are healthy and safe.")
