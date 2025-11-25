import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

def regression_accuracy(y_true, y_pred, tolerance=0.1):
    """
    Returns the percentage of predictions within ±tolerance of the true value.
    Example: tolerance=0.1 means within ±10%.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    valid = y_true != 0
    accuracy = np.mean(np.abs((y_pred[valid] - y_true[valid]) / y_true[valid]) <= tolerance)
    return accuracy


df=pd.read_csv("synthetic_high_corr.csv")



x=df[['Vehicle_Type','Age','Engine_Size','Load_Capacity']]
y1=df["Mileage"]
y2=df["Repair_Count"]
y3=df["Fuel_Consumption"]

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform Vehicle_Type column only
vehicle_encoded = ohe.fit_transform(x[['Vehicle_Type']])

# Convert to DataFrame with proper column names
vehicle_encoded_df = pd.DataFrame(vehicle_encoded, columns=ohe.get_feature_names_out(['Vehicle_Type']))






# Replace Vehicle_Type column with encoded columns
x_encoded = pd.concat([vehicle_encoded_df, x.drop(columns=['Vehicle_Type']).reset_index(drop=True)], axis=1)


x_train1,x_test1,y_train1,y_test1=train_test_split(x_encoded,y1, test_size=0.2, random_state=42)
x_train2,x_test2,y_train2,y_test2=train_test_split(x_encoded,y2, test_size=0.2, random_state=42)
x_train3,x_test3,y_train3,y_test3=train_test_split(x_encoded,y3,test_size=0.2, random_state=42)

model1=RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(x_train1,y_train1)
model2=RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(x_train2,y_train2)
model3=RandomForestRegressor(n_estimators=100, random_state=42)
model3.fit(x_train3,y_train3)

y_pred1=model1.predict(x_test1)
y_pred2=model2.predict(x_test2)
y_pred3=model3.predict(x_test3)

# prediction1=model1.predict(x_test1)
# prediction2=model2.predict(x_test2)
# prediction3=model3.predict(x_test3)

r_squared1=r2_score(y_test1, y_pred1)
r_squared2=r2_score(y_test2,y_pred2)
r_squared3=r2_score(y_test3,y_pred3)






def evaluate_model(name, y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    acc = regression_accuracy(y_test, y_pred, tolerance=0.1)
    print(f"\n=== {name} ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Accuracy (±10% tolerance): {acc*100:.2f}%")

evaluate_model("Mileage Model", y_test1, y_pred1)
evaluate_model("Repair Count Model", y_test2, y_pred2)
evaluate_model("Fuel Consumption Model", y_test3, y_pred3)


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

# Predict
mileage_pred = model1.predict(input_processed)[0]
repair_pred = model2.predict(input_processed)[0]
fuel_pred = model3.predict(input_processed)[0]

# --- Print Predictions ---
print("\n=== Prediction for New Vehicle ===")
print(f"Vehicle Type     : {new_vehicle['Vehicle_Type']}")
print(f"Age              : {new_vehicle['Age']} years")
print(f"Engine Size      : {new_vehicle['Engine_Size']} L")
print(f"Load Capacity    : {new_vehicle['Load_Capacity']} kg")

print("\n--- Predicted Outputs ---")
print(f"Mileage           : {mileage_pred:.2f} km/l")
print(f"Repair Count      : {repair_pred:.2f} per year")
print(f"Fuel Consumption  : {fuel_pred:.2f} L/100km")






# print(f"MSE for target one:{mean_squared_error(y_test1,prediction1  )}")
# print(f"R-squared value for target one: {r_squared1}")

# print(f"MSE for target two: {mean_squared_error(y_test2,prediction2)}")
# print(f"R-squared value for target two: {r_squared2}")

# print(f"MSE for target three: {mean_squared_error(y_test3, prediction3)}")
# print(f"R-squared value for target two: {r_squared3}")