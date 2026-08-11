# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "noisy_fish_dataset.csv"

def load_and_prepare(csv_file):
    df = pd.read_csv(csv_file)
    print("Columns in dataset:", df.columns.tolist())

    # Create binary label from Status
    df["label"] = (df["Status"].astype(str) == "Unhealthy").astype(int)

    # Drop identifiers and leakage features
    drop_cols = ["Sample_ID", "Status", "Disease_Risk"]
    features = [c for c in df.columns if c not in drop_cols + ["label"]]

    X = df[features].fillna(method="ffill").fillna(method="bfill").fillna(0)
    y = df["label"].values

    print(f"✅ Selected {len(features)} features for training.")
    return X, y, features

def train_models(X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC(kernel="rbf", probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n🏆 {name} Accuracy: {acc:.3f}")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.3f}")

    # Save model and scaler
    joblib.dump(best_model, "best_model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    print("💾 Saved best_model.joblib and scaler.joblib")

    # Plot feature importances if Random Forest is best
    if best_model_name == "Random Forest":
        importances = best_model.feature_importances_
        fi = pd.Series(importances, index=features).sort_values(ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(x=fi.values, y=fi.index)
        plt.title("Feature Importances (Random Forest)")
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        print("📊 Saved feature_importances.png")

if __name__ == "__main__":
    X, y, features = load_and_prepare(CSV_FILE)
    train_models(X, y, features)
