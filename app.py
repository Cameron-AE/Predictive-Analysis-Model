from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load models and encoder on startup
with open('model_mileage.pkl', 'rb') as f:
    model_mileage = pickle.load(f)

with open('model_repair_count.pkl', 'rb') as f:
    model_repair = pickle.load(f)

with open('model_fuel_consumption.pkl', 'rb') as f:
    model_fuel = pickle.load(f)

with open('vehicle_ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get form data
        vehicle_type = request.form['vehicle_type']
        age = float(request.form['age'])
        engine_size = float(request.form['engine_size'])
        load_capacity = float(request.form['load_capacity'])

        # Prepare input dataframe
        input_df = pd.DataFrame([{
            'Vehicle_Type': vehicle_type,
            'Age': age,
            'Engine_Size': engine_size,
            'Load_Capacity': load_capacity
        }])

        # One-hot encode Vehicle_Type
        vehicle_encoded = ohe.transform(input_df[['Vehicle_Type']])
        vehicle_encoded_df = pd.DataFrame(vehicle_encoded, columns=ohe.get_feature_names_out(['Vehicle_Type']))

        # Combine with other features
        input_processed = pd.concat([vehicle_encoded_df, input_df.drop(columns=['Vehicle_Type'])], axis=1)

        # Predict
        mileage_pred = model_mileage.predict(input_processed)[0]
        repair_pred = model_repair.predict(input_processed)[0]
        fuel_pred = model_fuel.predict(input_processed)[0]

        return render_template('result.html',
                               mileage=round(mileage_pred, 2),
                               repair=round(repair_pred, 2),
                               fuel=round(fuel_pred, 2),
                               vehicle_type=vehicle_type,
                               age=age,
                               engine_size=engine_size,
                               load_capacity=load_capacity)

    # GET request: show form
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)