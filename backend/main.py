import osmnx as ox
import networkx as nx
import random
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Delhi AQI Routing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD THE AI BRAIN ---
try:
    aqi_model = joblib.load("../ml_model/xgboost_aqi_model.pkl")
    print("✅ XGBoost ML Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    aqi_model = None


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered AQI Routing Backend!"}

@app.get("/api/get_route")
def get_real_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float):
    try:
        print(f"Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
        
        north = max(start_lat, end_lat) + 0.01
        south = min(start_lat, end_lat) - 0.01
        east = max(start_lon, end_lon) + 0.01
        west = min(start_lon, end_lon) - 0.01

        print("Downloading street network from OpenStreetMap...")
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')

        print("Asking AI for current pollution levels...")
        now = datetime.now()
        current_hour = now.hour
        current_day = now.weekday()

        if aqi_model:
            input_data = pd.DataFrame({'Hour': [current_hour], 'DayOfWeek': [current_day]})
            predicted_base_aqi = aqi_model.predict(input_data)[0]
        else:
            predicted_base_aqi = 200

        print(f"🤖 AI Predicts current Delhi Base AQI is: {predicted_base_aqi:.2f}")

        for u, v, key, data in G.edges(keys=True, data=True):
            hyper_local_aqi = predicted_base_aqi * random.uniform(0.8, 1.2)
            data['hyper_local_aqi'] = hyper_local_aqi 
            aqi_penalty = hyper_local_aqi / 50.0 
            data['aqi_adjusted_length'] = data['length'] * aqi_penalty

        orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        dest_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        print("Calculating TD-A* Eco-Friendly path...")
        route = nx.shortest_path(G, orig_node, dest_node, weight='aqi_adjusted_length')

        # --- CALCULATE THE SPECIFIC ROUTE AQI ---
        route_aqis = []
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                first_key = list(edge_data.keys())[0]
                route_aqis.append(edge_data[first_key]['hyper_local_aqi'])
                
        avg_route_aqi = 0
        if route_aqis:
            avg_route_aqi = sum(route_aqis) / len(route_aqis)
            print(f"🌿 Average AQI of this specific ECO-ROUTE: {avg_route_aqi:.2f}")

        path_coords = []
        for node in route:
            lat = G.nodes[node]['y']
            lon = G.nodes[node]['x']
            path_coords.append([lat, lon])

        print("Eco-Route successfully sent to the map!")
        
        # --- LOOK AT THE LAST LINE OF THIS BLOCK ---
        return {
            "status": "success",
            "message": "AI Eco street route calculated!",
            "path_coordinates": path_coords,
            "average_aqi": float(round(avg_route_aqi, 2))  # <--- WRAP IT IN float()
        }
        
    except Exception as e:
        print(f"Routing error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "path_coordinates": []
        }