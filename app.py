from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# 1. Model directory setup (robust, works if you run from project root)
model_dir = os.path.join(os.path.dirname(__file__), "model")

# 2. Define your LSTMModel class FIRST
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

# 3. Now instantiate scalers and model
scaler_X = joblib.load(os.path.join(model_dir, 'scaler_X.pkl'))
scaler_y = joblib.load(os.path.join(model_dir, 'scaler_y.pkl'))
input_dim = 3  # hour, weekday, junction

model = LSTMModel(input_dim, hidden_dim=64)
model_path = os.path.join(model_dir, 'traffic_lstm.pth')
model.load_state_dict(torch.load(model_path))
model.eval()
lookback = 6
last_inputs = []

# 4. Serve frontend from correct folder
frontend_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))

app = Flask(__name__)

@app.route('/')
def serve_index():
    return send_from_directory(frontend_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(frontend_folder, path)

@app.route('/predict', methods=['POST'])
def predict():
    global last_inputs
    inputs = request.json
    # Accept only non-negative values; notify for invalid input
    hour = float(inputs.get('hour', 0))
    weekday = float(inputs.get('weekday', 0))
    junction = int(inputs.get('Junction', 0))
    if hour < 0 or weekday < 0 or junction < 0:
        return jsonify({'error': 'Negative values not allowed for prediction.'}), 400
    current_X = np.array([[hour, weekday, junction]])
    current_X_scaled = scaler_X.transform(current_X)
    if len(last_inputs) < lookback - 1:
        history = [current_X_scaled] * (lookback - 1)
    else:
        history = last_inputs[-(lookback-1):]
    X_input = np.vstack(history + [current_X_scaled])
    X_input_torch = torch.tensor(X_input.reshape(1, lookback, input_dim), dtype=torch.float32)
    with torch.no_grad():
        y_pred_scaled = model(X_input_torch).numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]
    last_inputs.append(current_X_scaled)
    return jsonify({'predicted_traffic': float(y_pred)})

if __name__ == '__main__':
    app.run(debug=True)
