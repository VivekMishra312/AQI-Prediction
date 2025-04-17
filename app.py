from flask import Flask, jsonify
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from model import LSTMModel
import joblib  # for loading scaler objects

app = Flask(__name__)

# Load scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Load trained model
input_size = 4  # hour, day, month, weekday
hidden_size = 256
output_size = 9  # Number of pollutants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModel(input_size, hidden_size, output_size)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.to(device)
model.eval()


def generate_next_7_days_features():
    today = datetime.today()
    features = []

    for i in range(7):
        date = today + timedelta(days=i)
        features.append([
            12,  # Assume midday (12 PM)
            date.day,
            date.month,
            date.weekday()
        ])

    return np.array(features)


@app.route('/predict', methods=['GET'])
def predict():
    features = generate_next_7_days_features()
    features_scaled = scaler_X.transform(features)

    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).unsqueeze(1).to(device)
    with torch.no_grad():
        predictions = model(features_tensor).cpu().numpy()
    
    predictions_inverse = scaler_y.inverse_transform(predictions)
    response = []

    pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone']
    start_date = datetime.today()

    for i in range(7):
        prediction_dict = {'date': (start_date + timedelta(days=i)).strftime("%Y-%m-%d")}
        for j, pollutant in enumerate(pollutants):
            prediction_dict[pollutant] = float(round(predictions_inverse[i][j], 2))
        response.append(prediction_dict)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
