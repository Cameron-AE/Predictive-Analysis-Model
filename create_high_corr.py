import pandas as pd
import numpy as np

np.random.seed(42)

n = 300
vehicle_types = ["Truck", "Sedan", "SUV", "Van"]

data = []

for i in range(1, n + 1):
    vtype = np.random.choice(vehicle_types)

    if vtype == "Truck":
        engine = np.random.uniform(3.0, 5.0)
        load = np.random.uniform(5000, 15000)
    elif vtype == "SUV":
        engine = np.random.uniform(2.0, 3.5)
        load = np.random.uniform(1500, 3000)
    elif vtype == "Van":
        engine = np.random.uniform(2.0, 3.0)
        load = np.random.uniform(2000, 4000)
    else:  # Sedan
        engine = np.random.uniform(1.2, 2.5)
        load = np.random.uniform(800, 1500)

    age = np.random.uniform(1, 15)

    # tightly correlated synthetic targets
    mileage = 35 - 0.8 * age - 2.5 * engine - 0.002 * load + np.random.normal(0, 0.5)
    fuel_cons = 3 + 0.5 * engine + 0.05 * age + 0.0005 * load + np.random.normal(0, 0.2)
    repair_count = 0.3 * age + 0.0002 * load + np.random.normal(0, 0.2)

    data.append([
        i, vtype, mileage, fuel_cons, repair_count, age, engine, load
    ])

df = pd.DataFrame(
    data,
    columns=[
        "Fleet_ID", "Vehicle_Type", "Mileage", "Fuel_Consumption",
        "Repair_Count", "Age", "Engine_Size", "Load_Capacity"
    ]
)

# save to csv
df.to_csv("synthetic_high_corr.csv", index=False)
print("synthetic_fleet_data_correlated.csv created successfully!")
print(df.head())