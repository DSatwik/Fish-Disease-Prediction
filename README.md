# Fish Disease Prediction Using Machine Learning

## Project Overview

This project is a machine learning-based system for the early prediction of fish diseases using water quality parameters and fish activity information.

The system analyzes environmental and behavioral parameters such as temperature, dissolved oxygen, pH, turbidity, ammonia, nitrate, conductivity, and activity level to predict the possible health condition of fish.

## Project Title

**A Machine Learning System for Early Prediction of Fish Diseases Using Multimodal Data**

## Objectives

- Predict fish diseases at an early stage.
- Analyze water quality parameters affecting fish health.
- Use machine learning to classify fish health conditions.
- Provide a simple interface for entering parameters and obtaining predictions.
- Support fish farmers in monitoring fish health.

## Technologies Used

- Python
- Machine Learning
- Scikit-learn
- Pandas
- NumPy
- Flask
- Joblib
- HTML
- CSS

## Machine Learning

The project uses a machine learning classification model trained using fish farm and water quality data.

Important input parameters include:

- Temperature
- Dissolved Oxygen (DO)
- pH
- Turbidity
- BOD
- Conductivity
- Ammonia
- Nitrate
- Hardness
- Water Quality Index (WQI)
- COD
- CO2
- Alkalinity
- Fish Activity Level

## Project Structure

```text
fish-disease-project/
│
├── templates/
│   └── HTML template files
│
├── app.py
├── predict.py
├── train.py
├── noisy_train.py
│
├── best_model.joblib
├── rf_model.joblib
├── scaler.joblib
│
├── noisy_fish_dataset.xlsx
├── simulated_fish_farm_dataset.xlsx
│
└── feature_importances.png
