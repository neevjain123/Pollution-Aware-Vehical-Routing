import osmnx as ox
import networkx as nx
import joblib
import pandas as pd
import math
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

        # --- THE REAL SPATIAL MATH (IDW) ---
        stations = [
            {"name": "Anand Vihar", "lat": 28.6469, "lon": 77.3159, "aqi": predicted_base_aqi * 1.4}, # Super Dirty
            {"name": "RK Puram", "lat": 28.5632, "lon": 77.1869, "aqi": predicted_base_aqi * 1.05},
            {"name": "Punjabi Bagh", "lat": 28.6740, "lon": 77.1320, "aqi": predicted_base_aqi * 1.1},
            {"name": "ITO", "lat": 28.6284, "lon": 77.2405, "aqi": predicted_base_aqi * 1.2},
            {"name": "Dwarka", "lat": 28.5791, "lon": 77.0753, "aqi": predicted_base_aqi * 0.7}   # Super Clean
        ]

        # 2. Calculate actual pollution for every single street
        for u, v, key, data in G.edges(keys=True, data=True):
            u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
            v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
            street_lat = (u_y + v_y) / 2.0
            street_lon = (u_x + v_x) / 2.0

            numerator = 0
            denominator = 0

            for station in stations:
                dist = math.sqrt((street_lat - station['lat'])**2 + (street_lon - station['lon'])**2) + 0.0001
                weight = 1.0 / (dist ** 2)
                numerator += weight * station['aqi']
                denominator += weight
            
            hyper_local_aqi = numerator / denominator
            data['hyper_local_aqi'] = hyper_local_aqi 
            
            # THE FIX: Exponential Penalty to force aggressive routing
            aqi_penalty = (hyper_local_aqi / 100.0) ** 3 
            data['aqi_adjusted_length'] = data['length'] * aqi_penalty

        orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        dest_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        print("Calculating Standard Shortest path...")
        shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
        
        print("Calculating TD-A* Eco-Friendly path...")
        eco_route = nx.shortest_path(G, orig_node, dest_node, weight='aqi_adjusted_length')

        # --- THE FIX: LENGTH-WEIGHTED AVERAGE MATH ---
        def get_route_aqi(route_nodes):
            total_pollution = 0
            total_distance = 0
            for i in range(len(route_nodes) - 1):
                u = route_nodes[i]
                v = route_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    length = edge_data[first_key]['length']
                    aqi = edge_data[first_key]['hyper_local_aqi']
                    
                    # Multiply AQI by the physical length of the street
                    total_pollution += (aqi * length)
                    total_distance += length
            
            return total_pollution / total_distance if total_distance > 0 else 0

        def get_route_distance_km(route_nodes):
            total_meters = 0
            for i in range(len(route_nodes) - 1):
                u = route_nodes[i]
                v = route_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    total_meters += edge_data[first_key]['length']
            return total_meters / 1000.0

        avg_shortest_aqi = get_route_aqi(shortest_route)
        avg_eco_aqi = get_route_aqi(eco_route)
        
        shortest_km = get_route_distance_km(shortest_route)
        eco_km = get_route_distance_km(eco_route)

        shortest_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in shortest_route]
        eco_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in eco_route]
        
        return {
            "status": "success",
            "message": "Comparison routes calculated!",
            "shortest_path_coords": shortest_coords,
            "eco_path_coords": eco_coords,
            "shortest_aqi": float(round(avg_shortest_aqi, 2)),
            "eco_aqi": float(round(avg_eco_aqi, 2)),
            "shortest_km": float(round(shortest_km, 2)),
            "eco_km": float(round(eco_km, 2))
        }
        
    except Exception as e:
        print(f"Routing error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "path_coordinates": []
        }