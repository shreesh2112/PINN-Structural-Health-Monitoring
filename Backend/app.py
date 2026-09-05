from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

#PINN Architecture
class BridgePINN(nn.Module):
    def __init__(self):
        super(BridgePINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40), nn.Tanh(),
            nn.Linear(40, 40), nn.Tanh(),
            nn.Linear(40, 1)
        )
    def forward(self, x):
        return x * self.net(x)

# Load data and constants
data = pd.read_csv('D:\\truss_sensor_data.csv')
x_max = float(data['position_x'].max())
u_max = float(data['deflection_measured'].abs().max())

model = BridgePINN()
try:
    model.load_state_dict(torch.load('bridge_pinn.pth', map_location=torch.device('cpu')))
    model.eval()
    has_weights = True
except FileNotFoundError:
    has_weights = False

@app.get("/api/data")
def get_dashboard_data():
    x_eval = np.linspace(0, x_max, 100)
    
    if has_weights:
        with torch.no_grad():
            x_norm = torch.tensor(x_eval / x_max, dtype=torch.float32).view(-1, 1)
            y_pred = (model(x_norm).numpy().flatten() * u_max).tolist()
    else:
        y_pred = ((1e6 * x_eval) / (0.01 * 210e9)).tolist()

    return {
        "sensor_x": data['position_x'].tolist(),
        "sensor_y": data['deflection_measured'].tolist(),
        "pinn_x": x_eval.tolist(),
        "pinn_y": y_pred,
        "max_deflection": max(y_pred),
        "total_sensors": len(data)
    }

@app.get("/api/predict")
def predict_point(x: float):
    if x < 0 or x > x_max:
        return {"error": f"Position out of bounds (0-{x_max}m)"}
    
    if has_weights:
        with torch.no_grad():
            x_norm = torch.tensor([[x / x_max]], dtype=torch.float32)
            pred = float(model(x_norm).item() * u_max)
    else:
        pred = float((1e6 * x) / (0.01 * 210e9))
        
    return {"position": x, "deflection": pred}