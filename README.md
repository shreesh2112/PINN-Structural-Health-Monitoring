# Real-Time Bridge Structural Health Monitoring using PINNs 

An end-to-end cyber-physical digital twin platform for non-destructive Structural Health Monitoring (SHM) and real-time deflection profile reconstruction of truss bridges. 

This platform pairs a **Physics-Informed Neural Network (PINN)** with an asynchronous **FastAPI** backend server and a responsive **HTML5/Chart.js** dashboard.

---

## Key Technical Features

* **Hard Boundary Constraint Enforcement:** Integrates spatial boundary conditions directly into the neural network architecture ($u(0) = 0$), guaranteeing physical validity even under noisy input streams.
* **Elasticity Physics Integration:** Uses PyTorch automatic differentiation (`autograd`) to penalize deviations from beam elasticity equations alongside data MSE loss.
* **Automatic Dimension Normalization:** Dynamically maps spatial variables ($x_{\text{max}}$, $u_{\text{max}}$) into a stable $[0, 1]$ non-dimensional optimization space during training.
* **Low-Latency Asynchronous Serving:** Exposes RESTful endpoints via FastAPI/Uvicorn to process continuous spatial queries and batch updates.
* **Interactive UI/UX:** Real-time deflection profile charting, instant spatial coordinate spot-checking, and automatic anomaly status indicators.

---

## System Architecture

```text
┌─────────────────────────┐      ┌──────────────────────────┐      ┌─────────────────────────┐
│ Dynamic Field Sensors   │ ───► │ PINN Core Engine         │ ───► │ FastAPI Backend API     │
│ (Noisy Strain Gauges)   │      │ (Hard/Soft Constraints)  │      │ (Data/Prediction Route) │
└─────────────────────────┘      └──────────────────────────┘      └────────────┬────────────┘
                                                                                │
                                                                                ▼
                                                                   ┌─────────────────────────┐
                                                                   │ Client Web Dashboard    │
                                                                   │ (Chart.js / Single Page)│
                                                                   └─────────────────────────┘
