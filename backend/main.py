import os
import osmnx as ox
import networkx as nx
import joblib
import pandas as pd
import math
import requests # <-- ADDED FOR OPENWEATHER API
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

app = FastAPI(title="Delhi AQI Routing API")

# --- WEBSOCKET CONNECTION MANAGER ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/map")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = "graph_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

RAM_CACHE = {}

try:
    aqi_model = joblib.load("../ml_model/xgboost_aqi_model.pkl")
    print("✅ XGBoost ML Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Model not found. Using baseline AQI. ({e})")
    aqi_model = None

class TelemetryData(BaseModel):
    current_lat: float
    current_lon: float
    end_lat: float
    end_lon: float
    current_route_cost: float 
    mode: str = "car"  # <-- NEW: Dynamic endpoint now accepts mode 

# ==============================================================
# LIVE OPENWEATHER PHYSICS ENGINE
# ==============================================================
def get_live_wind_data():
    try:
        # YOUR LIVE OPENWEATHER API KEY
        API_KEY = "e08ed3f2fca35361d9656b15ca10cc82" 
        
        # Central Delhi Coordinates
        url = f"https://api.openweathermap.org/data/2.5/weather?lat=28.6139&lon=77.2090&appid={API_KEY}&units=metric"
        
        # 3-second timeout so the routing doesn't freeze if the internet drops
        res = requests.get(url, timeout=3) 
        data = res.json()
        
        # OpenWeather returns meters/second. Multiply by 3.6 to get km/h.
        speed_kmh = data['wind']['speed'] * 3.6
        direction_deg = data['wind']['deg']
        
        print(f"🌍 LIVE WIND ACQUIRED: {round(speed_kmh, 1)} km/h at {direction_deg}°")
        return {"speed_kmh": round(speed_kmh, 1), "direction_deg": direction_deg}
        
    except Exception as e:
        print(f"⚠️ Weather API Offline. Using Fallback. ({e})")
        return {"speed_kmh": 15.0, "direction_deg": 315.0} # Fallback to NW Wind

def calculate_directional_weight(station_lat, station_lon, street_lat, street_lon, dist, wind_speed, wind_dir):
    dy = street_lat - station_lat
    dx = street_lon - station_lon
    street_angle = (math.degrees(math.atan2(dy, dx)) + 360) % 360
    angle_diff = abs((street_angle - wind_dir + 180) % 360 - 180)
    wind_effect = math.cos(math.radians(angle_diff))
    
    directional_multiplier = max(0.25, 1.0 + (wind_speed * 0.05 * wind_effect))
    base_weight = 1.0 / (dist ** 2)
    return base_weight * directional_multiplier

# ==============================================================
# DYNAMIC ROUTING ENDPOINT (WEBSOCKETS)
# ==============================================================
@app.post("/api/v1/dynamic_route")
async def dynamic_recalculate(data: TelemetryData):
    try:
        if "delhi" not in RAM_CACHE:
            return {"status": "error", "message": "Map not loaded."}
        G = RAM_CACHE["delhi"]

        now = datetime.now()
        if aqi_model:
            input_data = pd.DataFrame({'Hour': [now.hour], 'DayOfWeek': [now.weekday()]})
            new_baseline_aqi = float(aqi_model.predict(input_data)[0])
        else:
            new_baseline_aqi = 200.0

        stations = [
            {"name": "Anand Vihar", "lat": 28.6469, "lon": 77.3159, "aqi": new_baseline_aqi * 1.4}, 
            {"name": "RK Puram", "lat": 28.5632, "lon": 77.1869, "aqi": new_baseline_aqi * 1.05},
            {"name": "Punjabi Bagh", "lat": 28.6740, "lon": 77.1320, "aqi": new_baseline_aqi * 1.1},
            {"name": "ITO", "lat": 28.6284, "lon": 77.2405, "aqi": new_baseline_aqi * 1.2},
            {"name": "Dwarka", "lat": 28.5791, "lon": 77.0753, "aqi": new_baseline_aqi * 0.7}   
        ]

        # GRAB LIVE WIND
        wind_data = get_live_wind_data()

        for u, v, key, edge_data in G.edges(keys=True, data=True):
            u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
            v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
            street_lat = (u_y + v_y) / 2.0
            street_lon = (u_x + v_x) / 2.0

            numerator, denominator = 0, 0
            for station in stations:
                dist = math.sqrt((street_lat - station['lat'])**2 + (street_lon - station['lon'])**2) + 0.0001
                weight = calculate_directional_weight(
                    station['lat'], station['lon'], street_lat, street_lon, dist, wind_data['speed_kmh'], wind_data['direction_deg']
                )
                numerator += weight * station['aqi']
                denominator += weight
            
            hyper_local_aqi = numerator / denominator
            raw_length = edge_data.get('length', 1.0)
            edge_length = sum(raw_length) if isinstance(raw_length, list) else float(raw_length)

            # --- THE EXPONENTIAL BIOLOGICAL FIX ---
            mode_exponent = 5.0
            if data.mode == "bike":
                mode_exponent = 10.0  # Massive exponential penalty for cyclists
            elif data.mode == "walk":
                mode_exponent = 7.0   # Medium penalty for pedestrians

            G[u][v][key]['hyper_local_aqi'] = hyper_local_aqi 
            G[u][v][key]['length_balanced'] = edge_length * ((hyper_local_aqi / 50.0) ** 2)
            
            # Apply the mode directly to the power curve!
            G[u][v][key]['length_extreme'] = edge_length * ((hyper_local_aqi / 50.0) ** mode_exponent)
            # --------------------------------------
        current_node = ox.distance.nearest_nodes(G, X=data.current_lon, Y=data.current_lat)
        end_node = ox.distance.nearest_nodes(G, X=data.end_lon, Y=data.end_lat)

        new_route = nx.shortest_path(G, current_node, end_node, weight='length_extreme')
        
        new_route_cost = 0
        for i in range(len(new_route) - 1):
            u, v = new_route[i], new_route[i+1]
            edge_info = G.get_edge_data(u, v)
            if edge_info:
                first_key = list(edge_info.keys())[0]
                new_route_cost += edge_info[first_key].get('length_extreme', 1.0)

        THRESHOLD = 0.85 

        if new_route_cost <= (data.current_route_cost * THRESHOLD):
            new_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in new_route]
            await manager.broadcast({"type": "REROUTE", "coords": new_coords})
            return {
                "status": "REROUTE_TRIGGERED",
                "message": "Significant AQI drop detected. Rerouting.",
                "new_baseline": float(new_baseline_aqi),
                "route_coords": new_coords,
                "cost": float(new_route_cost),
                "wind_data": wind_data # <-- SENT TO FRONTEND
            }
        else:
            snapped_lat = G.nodes[current_node]['y']
            snapped_lon = G.nodes[current_node]['x']
            await manager.broadcast({
                "type": "LOCATION_UPDATE", 
                "current_location": [snapped_lat, snapped_lon]
            })
            return {
                "status": "ROUTE_STABLE",
                "message": "Current route is still optimal.",
                "new_baseline": float(new_baseline_aqi),
                "route_coords": None, 
                "cost": float(data.current_route_cost),
                "wind_data": wind_data # <-- SENT TO FRONTEND
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# ==============================================================
# STATIC ROUTING ENDPOINT (INITIAL LOAD)
# ==============================================================
@app.get("/api/get_route")
def get_real_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float, mode: str= "car"):
    try:
        print(f"\n--- NEW ROUTE REQUEST ---")
        
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
        cache_path = os.path.join(CACHE_DIR, "delhi_master_graph.pkl")

        if "delhi" in RAM_CACHE:
            print("🚀 Reading city from RAM...")
            G = RAM_CACHE["delhi"]
        elif os.path.exists(cache_path):
            print("⚡ Reading city from Hard Drive...")
            G = joblib.load(cache_path)
            RAM_CACHE["delhi"] = G
        else:
            print("🐢 MEGA SLOW LOAD: Downloading map of Delhi...")
            G = ox.graph_from_point((28.6139, 77.2090), dist=20000, network_type='drive')
            joblib.dump(G, cache_path)
            RAM_CACHE["delhi"] = G

        now = datetime.now()
        if aqi_model:
            input_data = pd.DataFrame({'Hour': [now.hour], 'DayOfWeek': [now.weekday()]})
            predicted_base_aqi = float(aqi_model.predict(input_data)[0])
        else:
            predicted_base_aqi = 200.0

        stations = [
            {"name": "Anand Vihar", "lat": 28.6469, "lon": 77.3159, "aqi": predicted_base_aqi * 1.4}, 
            {"name": "RK Puram", "lat": 28.5632, "lon": 77.1869, "aqi": predicted_base_aqi * 1.05},
            {"name": "Punjabi Bagh", "lat": 28.6740, "lon": 77.1320, "aqi": predicted_base_aqi * 1.1},
            {"name": "ITO", "lat": 28.6284, "lon": 77.2405, "aqi": predicted_base_aqi * 1.2},
            {"name": "Dwarka", "lat": 28.5791, "lon": 77.0753, "aqi": predicted_base_aqi * 0.7}   
        ]

        # GRAB LIVE WIND
        wind_data = get_live_wind_data()

        for u, v, key, edge_data in G.edges(keys=True, data=True):
            u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
            v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
            street_lat = (u_y + v_y) / 2.0
            street_lon = (u_x + v_x) / 2.0

            numerator, denominator = 0, 0
            for station in stations:
                dist = math.sqrt((street_lat - station['lat'])**2 + (street_lon - station['lon'])**2) + 0.0001
                weight = calculate_directional_weight(
                    station['lat'], station['lon'], street_lat, street_lon, dist, wind_data['speed_kmh'], wind_data['direction_deg']
                )
                numerator += weight * station['aqi']
                denominator += weight
            
            hyper_local_aqi = numerator / denominator
            raw_length = edge_data.get('length', 1.0)
            edge_length = sum(raw_length) if isinstance(raw_length, list) else float(raw_length)

            # --- THE EXPONENTIAL BIOLOGICAL FIX ---
            mode_exponent = 5.0
            if mode == "bike":
                mode_exponent = 10.0  # Massive exponential penalty for cyclists
            elif mode == "walk":
                mode_exponent = 7.0   # Medium penalty for pedestrians

            G[u][v][key]['hyper_local_aqi'] = hyper_local_aqi 
            G[u][v][key]['length_balanced'] = edge_length * ((hyper_local_aqi / 50.0) ** 2)
            
            # Apply the mode directly to the power curve!
            G[u][v][key]['length_extreme'] = edge_length * ((hyper_local_aqi / 50.0) ** mode_exponent)
            # --------------------------------------
        orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        dest_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
        balanced_route = nx.shortest_path(G, orig_node, dest_node, weight='length_balanced')
        extreme_route = nx.shortest_path(G, orig_node, dest_node, weight='length_extreme')

        def get_route_aqi(route_nodes):
            total_pollution, total_distance = 0, 0
            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    length_raw = edge_data[first_key].get('length', 1.0)
                    length = sum(length_raw) if isinstance(length_raw, list) else float(length_raw)
                    aqi = edge_data[first_key].get('hyper_local_aqi', 100.0)
                    total_pollution += (aqi * length)
                    total_distance += length
            return total_pollution / total_distance if total_distance > 0 else 0

        def get_route_distance_km(route_nodes):
            total_meters = 0
            for i in range(len(route_nodes) - 1):
                u, v = route_nodes[i], route_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    length_raw = edge_data[first_key].get('length', 1.0)
                    total_meters += sum(length_raw) if isinstance(length_raw, list) else float(length_raw)
            return total_meters / 1000.0

        return {
            "status": "success",
            "message": "3-Way Comparison routes calculated!",
            "stations": stations,
            "wind_data": wind_data, # <-- SENT TO FRONTEND
            
            "shortest_path_coords": [[G.nodes[n]['y'], G.nodes[n]['x']] for n in shortest_route],
            "shortest_aqi": float(round(get_route_aqi(shortest_route), 2)),
            "shortest_km": float(round(get_route_distance_km(shortest_route), 2)),

            "balanced_path_coords": [[G.nodes[n]['y'], G.nodes[n]['x']] for n in balanced_route],
            "balanced_aqi": float(round(get_route_aqi(balanced_route), 2)),
            "balanced_km": float(round(get_route_distance_km(balanced_route), 2)),

            "extreme_path_coords": [[G.nodes[n]['y'], G.nodes[n]['x']] for n in extreme_route],
            "extreme_aqi": float(round(get_route_aqi(extreme_route), 2)),
            "extreme_km": float(round(get_route_distance_km(extreme_route), 2))
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}