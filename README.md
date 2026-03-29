# 🌍 Pollution Aware Vehical Routing(Delhi)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Production_Ready-009688.svg)
![WebSockets](https://img.shields.io/badge/WebSockets-Real_Time-yellow.svg)
![Machine Learning](https://img.shields.io/badge/XGBoost-Predictive_Model-orange.svg)
![Leaflet.js](https://img.shields.io/badge/Leaflet.js-Interactive_Maps-lightgreen.svg)

## 📌 Overview
The **Delhi AQI Route Optimizer** is a multi-objective spatial routing engine designed to minimize human exposure to hyper-local air pollution. Unlike standard GPS systems that optimize purely for time or distance, this engine calculates optimal paths using a custom A* algorithm that dynamically reacts to **live wind physics**, **predictive pollution baselines**, and the **biological respiratory rates** of different transport modes.

This project was engineered to solve a real-world problem in New Delhi, demonstrating complex graph mathematics, real-time telemetry, and full-stack integration.

---

## 🚀 Core Architecture & Features

### 1. 🫁 Multi-Modal Biological Multipliers (Non-Linear Graph Weights)
The A* search algorithm does not treat all users equally. It dynamically applies exponential penalties to the graph edges based on the user's transportation mode and minute-ventilation (respiratory rate).
* **🚗 Car Mode ($Exponent = 5$):** Baseline penalty for enclosed cabin routing.
* **🚶 Pedestrian Mode ($Exponent = 7$):** Increased penalty accounting for prolonged street-level exposure time.
* **🚴 Cyclist Mode ($Exponent = 10$):** Massive exponential penalty accounting for high cardiovascular respiration and deep PM2.5 inhalation, violently routing cyclists away from toxic hotspots.

### 2. 🌬️ Live Wind Vector Physics Engine
Integrates with the **OpenWeather API** to fetch real-time wind speed and directional headings. The backend calculates a dynamic directional multiplier using cosine similarities to predict where pollution from static hotspots is actively drifting, shifting the graph weights in real-time before the route is drawn.

### 3. 📡 Reactive Telemetry & Dynamic Rerouting (WebSockets)
Features a continuous evaluation loop for moving users. The client sends a telemetry pulse every 1.5 seconds via **WebSockets**. The backend recalculates the world-state using the live timestamp and current coordinates. If a sudden wind shift or pollution spike creates an alternative path that is at least **15% safer** than the remaining route, the server broadcasts a `REROUTE` payload, triggering an instant UI path shift.

### 4. 🧠 Predictive AQI Baseline (XGBoost)
Instead of relying strictly on fragile, high-latency live sensor scraping, the routing engine utilizes an **XGBoost Machine Learning model** (trained on historical Delhi sensor data). It predicts a highly stable city-wide pollution baseline based on the current hour and day of the week, ensuring zero-latency route generation.

---

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn, WebSockets
* **Spatial/Math:** OSMnx, NetworkX, Pandas
* **Machine Learning:** Scikit-Learn, XGBoost, Joblib
* **Frontend:** Vanilla JavaScript, HTML5, CSS3, Leaflet.js
* **External APIs:** Nominatim (Smart Geocoding), OpenWeather API (Wind Physics)

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_USERNAME/aqi_routing.git](https://github.com/YOUR_USERNAME/aqi_routing.git)
cd aqi_routing