# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "simulated_fish_farm_dataset.csv"

def load_and_prepare(csv_file):

    df = pd.read_csv(csv_file)

    print("Columns:", df.columns.tolist())

    # Create binary label
    if "Status" in df.columns:
        df["label"] = (df["Status"].astype(str) == "Unhealthy").astype(int)

    elif "Disease_Risk" in df.columns:
        df["label"] = (df["Disease_Risk"] > 0.5).astype(int)

    else:
        raise ValueError("No Status or Disease_Risk column found")

    # Correct feature names from dataset
    features = [
        "Temperature",
        "DO",
        "pH",
        "Turbidity",
        "BOD",
        "Conductivity",
        "Ammonia",
        "Nitrate",
        "Hardness",
        "COD",
        "CO2",
        "Alkalinity",
        "Activity_Level"
    ]

    X = df[features].fillna(0)

    y = df["label"]

    return X, y, features


def train_and_save(X, y, features):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Accuracy
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance
    importance = model.feature_importances_

    sns.barplot(x=importance, y=features)

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig("feature_importances.png")

    print("\nSaved feature_importances.png")

    # Save model and scaler
    joblib.dump(model, "rf_model.joblib")

    joblib.dump(scaler, "scaler.joblib")

    print("\nSaved rf_model.joblib")

    print("Saved scaler.joblib")


if __name__ == "__main__":

    X, y, features = load_and_prepare(CSV_FILE)

    train_and_save(X, y, features)