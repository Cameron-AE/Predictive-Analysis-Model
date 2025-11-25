import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error


def regression_accuracy(y_true,y_pred, tolerance=1):
    y_true,y_pred=np.array(y_true), np.array(y_pred)
    valid=y_true!=0
    accuracy=np.mean(np.abs((y_pred[valid]-y_true[valid])/y_true[valid])<=tolerance)   # calculates fraction of predictions where relative error is below the set tollerance

    return accuracy

def evaluate_model(name,y_test,y_pred):
    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    acc=regression_accuracy(y_test,y_pred, tolerance=0.1)
    print()
    print(f"==={name}===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")    #Mean Absolute Error
    print(f"RMSE: {rmse:.2f}")  #Root Mean Square error
    print(f"Accuracy(±10%): {acc*100:.2f}%")


df=pd.read_csv("synthetic_high_corr.csv")


x=df[["Vehicle_Type", "Age", "Engine_Size", "Load_Capacity"]]
y1=df["Mileage"]
y2=df["Repair_Count"]
y3=df["Fuel_Consumption"]



    # One-hot encode Vehicle_Type
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
vehicle_encoded = ohe.fit_transform(x[['Vehicle_Type']])
vehicle_encoded_df = pd.DataFrame(vehicle_encoded, columns=ohe.get_feature_names_out(['Vehicle_Type']))
x_encoded = pd.concat([vehicle_encoded_df, x.drop(columns=['Vehicle_Type']).reset_index(drop=True)], axis=1)


x_train1,x_test1,y_train1,y_test1=train_test_split(x_encoded,y1, test_size=0.2, random_state=42)
x_train2,x_test2,y_train2,y_test2=train_test_split(x_encoded,y2,test_size=0.2, random_state=42 )
x_train3,x_test3,y_train3,y_test3=train_test_split(x_encoded,y3, test_size=0.2, random_state=42)


model1=LinearRegression()
model1.fit(x_train1,y_train1)

model2=LinearRegression()
model2.fit(x_train2,y_train2)

model3=LinearRegression()
model3.fit(x_train3,y_train3)

y_pred1=model1.predict(x_test1)
y_pred2=model2.predict(x_test2)
y_pred3=model3.predict(x_test3)


evaluate_model("Mileage Model(Linear)",y_test1,y_pred1)
evaluate_model("Repair Count(Linear)",y_test2,y_pred2)
evaluate_model("Fuel Consumption(Linear)",y_test3,y_pred3)


new_vehicle={
    "Vehicle_Type":"Car",
    "Age":5,
    "Engine_Size":2.0,
    "Load_Capacity":1500
    
}

input_df = pd.DataFrame([new_vehicle])
vehicle_encoded = ohe.transform(input_df[["Vehicle_Type"]])
vehicle_encoded_df = pd.DataFrame(vehicle_encoded, columns=ohe.get_feature_names_out(["Vehicle_Type"]))
input_processed = pd.concat([
    vehicle_encoded_df,
    input_df.drop(columns=["Vehicle_Type"]).reset_index(drop=True)
], axis=1)


mileage_pred=model1.predict(input_processed)[0]
repair_count_pred=model2.predict(input_processed)[0]
fuel_consumption_pred=model3.predict(input_processed)[0]

print("\n=== Prediction for New Vehicle ===")
print(f"Vehicle Type     : {new_vehicle['Vehicle_Type']}")
print(f"Age              : {new_vehicle['Age']} years")
print(f"Engine Size      : {new_vehicle['Engine_Size']} L")
print(f"Load Capacity    : {new_vehicle['Load_Capacity']} kg")

print("\n--- Predicted Outputs (Linear Model) ---")
print(f"Mileage           : {mileage_pred:.2f} km/l")
print(f"Repair Count      : {repair_count_pred:.2f} per year")
print(f"Fuel Consumption  : {fuel_consumption_pred:.2f} L/100km")



# # Save models
# with open('model_mileage.pkl', 'wb') as f:
#     pickle.dump(model1, f)

# with open('model_repair_count.pkl', 'wb') as f:
#     pickle.dump(model2, f)

# with open('model_fuel_consumption.pkl', 'wb') as f:
#     pickle.dump(model3, f)

# # Save encoder
# with open('vehicle_ohe.pkl', 'wb') as f:
#     pickle.dump(ohe, f)