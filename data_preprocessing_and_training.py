import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import joblib
import random

# Set deterministic behavior for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Load and preprocess data
data = pd.read_csv('data/traffic_data.csv')
print(data.columns)  # For quick verification

# Basic data cleaning
original_len = len(data)
data = data.dropna(subset=['DateTime', 'Junction', 'Vehicles'])
print(f"Dropped {original_len - len(data)} rows due to NA in key columns.")

# Filter negative or invalid values
neg = (data['Vehicles'] < 0) | (data['Junction'] < 0)
if neg.any():
    print(f"Warning: Dropping {neg.sum()} rows with negative values.")
    data = data[~neg]

# Extract time features
data['hour'] = pd.to_datetime(data['DateTime']).dt.hour
data['weekday'] = pd.to_datetime(data['DateTime']).dt.dayofweek

# Ensure all features exist
feature_cols = ['hour', 'weekday', 'Junction']
for col in feature_cols:
    if col not in data.columns:
        raise ValueError(f"Missing expected column: {col}")

features = feature_cols
target = 'Vehicles'

X = data[features].values
y = data[target].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

def create_sequences(X, y, lookback=6):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i-lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

lookback = 6
X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)

split = int(0.8 * len(X_seq))
X_train, y_train = X_seq[:split], y_seq[:split]
X_test, y_test = X_seq[split:], y_seq[split:]

X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).flatten()
X_test_torch = torch.tensor(X_test, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).flatten()

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])

input_dim = X_train.shape[2]
model = LSTMModel(input_dim, hidden_dim=64)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 30
for epoch in range(epochs):
    model.train()
    optim.zero_grad()
    output = model(X_train_torch)
    loss = criterion(output.flatten(), y_train_torch)
    loss.backward()
    optim.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Save model and scalers
joblib.dump(scaler_X, 'backend/model/scaler_X.pkl')
joblib.dump(scaler_y, 'backend/model/scaler_y.pkl')
torch.save(model.state_dict(), 'backend/model/traffic_lstm.pth')

print("Training complete. Model and scalers saved.")
