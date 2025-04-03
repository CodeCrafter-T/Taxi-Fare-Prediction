from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Define the preprocessing function FIRST
def preprocess_input(df):
    df = df.copy()
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Extract datetime features
    df['pickup_datetime_year'] = df['pickup_datetime'].dt.year
    df['pickup_datetime_month'] = df['pickup_datetime'].dt.month
    df['pickup_datetime_day'] = df['pickup_datetime'].dt.day
    df['pickup_datetime_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_datetime_hour'] = df['pickup_datetime'].dt.hour

    # Haversine function
    def haversine_np(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # Earth radius in km

    # Landmark coordinates
    landmarks = {
        'jfk': (-73.7781, 40.6413),
        'lga': (-73.8740, 40.7769),
        'ewr': (-74.1745, 40.6895),
        'met': (-73.9632, 40.7794),
        'wtc': (-74.0099, 40.7126)
    }

    # Calculate distances
    df['trip_distance'] = haversine_np(df['pickup_longitude'], df['pickup_latitude'],
                                      df['dropoff_longitude'], df['dropoff_latitude'])
    
    for name, (lon, lat) in landmarks.items():
        df[f'{name}_drop_distance'] = haversine_np(lon, lat, 
                                                 df['dropoff_longitude'], 
                                                 df['dropoff_latitude'])

    # Select final features
    return df[['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
              'passenger_count', 'pickup_datetime_year', 'pickup_datetime_month',
              'pickup_datetime_day', 'pickup_datetime_weekday', 'pickup_datetime_hour',
              'trip_distance', 'jfk_drop_distance', 'lga_drop_distance', 
              'ewr_drop_distance', 'met_drop_distance', 'wtc_drop_distance']]

# Load model and preprocessor
with open("models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        'pickup_datetime': request.form['pickup_datetime'],
        'pickup_longitude': float(request.form['pickup_longitude']),
        'pickup_latitude': float(request.form['pickup_latitude']),
        'dropoff_longitude': float(request.form['dropoff_longitude']),
        'dropoff_latitude': float(request.form['dropoff_latitude']),
        'passenger_count': float(request.form['passenger_count'])
    }
    
    df = pd.DataFrame(data, index=[0])
    processed = preprocess_input(df)
    prediction = model.predict(processed)[0]
    
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)