import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Custom regression accuracy metric
def regression_accuracy(y_true, y_pred, tolerance=0.1):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    valid = y_true != 0
    accuracy = np.mean(np.abs((y_pred[valid] - y_true[valid]) / y_true[valid]) <= tolerance)
    return accuracy

# Load dataset
df = pd.read_csv("synthetic_high_corr.csv")

# Features and targets
x = df[['Vehicle_Type', 'Age', 'Engine_Size', 'Load_Capacity']]
y1 = df["Mileage"]
y2 = df["Repair_Count"]
y3 = df["Fuel_Consumption"]

# One-hot encode Vehicle_Type
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
vehicle_encoded = ohe.fit_transform(x[['Vehicle_Type']])
vehicle_encoded_df = pd.DataFrame(vehicle_encoded, columns=ohe.get_feature_names_out(['Vehicle_Type']))

# Combine encoded and numeric features
x_encoded = pd.concat([vehicle_encoded_df, x.drop(columns=['Vehicle_Type']).reset_index(drop=True)], axis=1)

# Split data (same split across all targets)
x_train, x_test, y_train_df, y_test_df = train_test_split(
    x_encoded,
    df[['Mileage', 'Repair_Count', 'Fuel_Consumption']],
    test_size=0.2,
    random_state=42
)

# Initialize Gradient Boosting models
model1 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model2 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
model3 = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)

# Fit models
model1.fit(x_train, y_train_df["Mileage"])
model2.fit(x_train, y_train_df["Repair_Count"])
model3.fit(x_train, y_train_df["Fuel_Consumption"])

# Predictions
y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)

# Evaluation function
def evaluate_model(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc = regression_accuracy(y_test, y_pred, tolerance=0.1)

    print(f"\n=== {name} ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy (±10% tolerance): {acc * 100:.2f}%")

# Evaluate all models
evaluate_model("Mileage Model (GB)", y_test_df["Mileage"], y_pred1)
evaluate_model("Repair Count Model (GB)", y_test_df["Repair_Count"], y_pred2)
evaluate_model("Fuel Consumption Model (GB)", y_test_df["Fuel_Consumption"], y_pred3)

# --- Predict on new vehicle input ---
new_vehicle = {
    "Vehicle_Type": "SUV",
    "Age": 7,
    "Engine_Size": 2.0,
    "Load_Capacity": 2000
}

# Convert input to DataFrame
input_df = pd.DataFrame([new_vehicle])

# Encode Vehicle_Type
vehicle_encoded = ohe.transform(input_df[["Vehicle_Type"]])
vehicle_encoded_df = pd.DataFrame(vehicle_encoded, columns=ohe.get_feature_names_out(["Vehicle_Type"]))

# Combine with numeric features
input_processed = pd.concat([
    vehicle_encoded_df,
    input_df.drop(columns=["Vehicle_Type"]).reset_index(drop=True)
], axis=1)

# Predict for new vehicle
mileage_pred = model1.predict(input_processed)[0]
repair_pred = model2.predict(input_processed)[0]
fuel_pred = model3.predict(input_processed)[0]

# --- Print Predictions ---
print("\n=== Prediction for New Vehicle ===")
print(f"Vehicle Type : {new_vehicle['Vehicle_Type']}")
print(f"Age : {new_vehicle['Age']} years")
print(f"Engine Size : {new_vehicle['Engine_Size']} L")
print(f"Load Capacity : {new_vehicle['Load_Capacity']} kg")

print("\n--- Predicted Outputs ---")
print(f"Mileage : {mileage_pred:.2f} km/l")
print(f"Repair Count : {repair_pred:.2f} per year")
print(f"Fuel Consumption : {fuel_pred:.2f} L/100km")