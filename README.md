# 🌍 Hyper-Local Air Quality & Route Recommendation System

**Academic Level:** 4th Semester Engineering Project | MNIT Jaipur  
**Domain:** Spatial Data Science, Machine Learning, Full-Stack Web Development  

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Modern-009688.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange.svg)
![Leaflet](https://img.shields.io/badge/Leaflet.js-Interactive_Map-brightgreen.svg)

## 📌 Overview
The **Eco-Route Optimizer** is a full-stack spatial routing engine that determines the optimal path between two geographic coordinates not just by physical distance, but by minimizing user exposure to particulate matter and air pollution. 

Unlike standard navigation apps that rely on shortest-path algorithms, this system employs a custom Multi-Objective A* algorithm, predicting temporal baseline pollution using Machine Learning, and spatially distributing it across a real-time OpenStreetMap graph to generate dynamic "Eco-Routes."

---

## 🚀 Key Features

* **🧠 Dynamic Temporal Forecasting:** An XGBoost Regressor trained on historical Delhi AQI data predicts the baseline city pollution based on the current hour and day, eliminating external API latency.
* **🗺️ Spatial Interpolation:** Utilizes an Inverse Distance Weighting (IDW) algorithm to blend data from anchor stations, creating a hyper-local pollution gradient across thousands of individual street edges.
* **🚦 Multi-Objective Routing:** Triggers exponential spatial penalties ($Cost = Distance \times (AQI/100)^\alpha$) to dynamically alter street lengths, allowing the algorithm to actively bypass severe pollution hotspots.
* **⚡ REST API Backend:** Powered by FastAPI with server-side caching, ensuring sub-second route calculation and graph traversal.
* **📱 Interactive Web Dashboard:** A responsive frontend built with Leaflet.js that allows users to drop pins, visualize the graph, and compare the Standard Route vs. the Extreme Eco-Route in real-time.
* **🔄 Automated ETL Pipeline:** Integrated `.bat` scripting to automate the fetching of fresh web data for continuous ML model retraining.

---

## 🏗️ System Architecture & Tech Stack

### 1. Backend & Geospatial Engine
* **Framework:** FastAPI
* **Graph Extraction:** OSMnx (OpenStreetMap NetworkX)
* **Routing Mathematics:** NetworkX (Dijkstra / A-Star)

### 2. Machine Learning Pipeline
* **Model:** XGBoost Regressor (100 Estimators, Depth 5)
* **Data Processing:** Pandas, Scikit-Learn (Train/Test Split, MAE Evaluation)
* **Serialization:** Joblib

### 3. Frontend Visualization
* **Mapping Library:** Leaflet.js
* **Routing UI:** Leaflet Routing Machine
* **Languages:** HTML5, CSS3, Vanilla JavaScript

---

## 📂 Repository Structure

```text
├── backend/
│   ├── main.py                           # FastAPI REST server & routing engine
│   └── requirements.txt                  # Python dependencies
├── ml_model/
│   ├── train.py                          # ML training pipeline and feature engineering
│   ├── xgboost_aqi_model.pkl             # Serialized XGBoost predictor
│   ├── fetch_data.bat                    # Automated ETL pipeline script
│   └── New Delhi January dataset hourly.csv # Historical training data
├── frontend/
│   ├── index.html                        # Main web dashboard
│   ├── style.css                         # UI styling
│   └── app.js                            # API integration and Leaflet map logic
└── README.md