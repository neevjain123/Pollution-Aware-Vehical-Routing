import os
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

# --- CREATE CACHE DIRECTORY ---
CACHE_DIR = "graph_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# --- LEVEL 2: IN-MEMORY RAM CACHE ---
# This holds the street networks in active memory so we don't even read the hard drive!
RAM_CACHE = {}

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
        print(f"\n--- NEW ROUTE REQUEST ---")
        print(f"Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
        
        # --- THE ENTERPRISE CACHING ENGINE (ENTIRE CITY) ---
        cache_path = os.path.join(CACHE_DIR, "delhi_master_graph.pkl")

        if "delhi" in RAM_CACHE:
            print("🚀 INSTANT LOAD: Reading entire city from RAM...")
            G = RAM_CACHE["delhi"]
        elif os.path.exists(cache_path):
            print("⚡ FAST LOAD: Reading entire city from Hard Drive...")
            G = joblib.load(cache_path)
            RAM_CACHE["delhi"] = G
        else:
            print("🐢 MEGA SLOW LOAD (ONCE EVER): Downloading entire map of Delhi...")
            print("⏳ This will take 30-60 seconds, but you will NEVER have to do it again.")
            # Downloads a massive 20km radius around central Delhi (Covers Dwarka to Anand Vihar)
            G = ox.graph_from_point((28.6139, 77.2090), dist=20000, network_type='drive')
            print("💾 Saving Delhi to Hard Drive and RAM...")
            joblib.dump(G, cache_path)
            RAM_CACHE["delhi"] = G

        # --- ASK AI FOR BASE POLLUTION ---
        print("Asking AI for current pollution levels...")
        now = datetime.now()
        current_hour = now.hour
        current_day = now.weekday()

        if aqi_model:
            input_data = pd.DataFrame({'Hour': [current_hour], 'DayOfWeek': [current_day]})
            predicted_base_aqi = float(aqi_model.predict(input_data)[0])
        else:
            predicted_base_aqi = 200.0

        # --- THE REAL SPATIAL MATH (IDW) ---
        stations = [
            {"name": "Anand Vihar", "lat": 28.6469, "lon": 77.3159, "aqi": predicted_base_aqi * 1.4}, 
            {"name": "RK Puram", "lat": 28.5632, "lon": 77.1869, "aqi": predicted_base_aqi * 1.05},
            {"name": "Punjabi Bagh", "lat": 28.6740, "lon": 77.1320, "aqi": predicted_base_aqi * 1.1},
            {"name": "ITO", "lat": 28.6284, "lon": 77.2405, "aqi": predicted_base_aqi * 1.2},
            {"name": "Dwarka", "lat": 28.5791, "lon": 77.0753, "aqi": predicted_base_aqi * 0.7}   
        ]

        # Calculate actual pollution and MULTIPLE weights for every street
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
            
            # --- THE 3-WAY COST FUNCTION ---
            data['length_balanced'] = data['length'] * ((hyper_local_aqi / 100.0) ** 2)
            data['length_extreme'] = data['length'] * ((hyper_local_aqi / 100.0) ** 10)

        orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        dest_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        print("Calculating 3 distinct paths...")
        shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
        balanced_route = nx.shortest_path(G, orig_node, dest_node, weight='length_balanced')
        extreme_route = nx.shortest_path(G, orig_node, dest_node, weight='length_extreme')

        # --- MATH HELPER FUNCTIONS ---
        def get_route_aqi(route_nodes):
            total_pollution = 0
            total_distance = 0
            for i in range(len(route_nodes) - 1):
                u = route_nodes[i]
                v = route_nodes[i+1]
                edge_data = G.get_edge_data(u, v)
                if edge_data:
                    first_key = list(edge_data.keys())[0]
                    length = edge_data[first_key].get('length', 1.0)
                    aqi = edge_data[first_key].get('hyper_local_aqi', 100.0)
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
                    total_meters += edge_data[first_key].get('length', 1.0)
            return total_meters / 1000.0

        # --- EXTRACT ALL DATA ---
        shortest_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in shortest_route]
        balanced_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in balanced_route]
        extreme_coords = [[G.nodes[n]['y'], G.nodes[n]['x']] for n in extreme_route]
        
        print("Done! Sending payload to map.\n")
        return {
            "status": "success",
            "message": "3-Way Comparison routes calculated!",
            "stations": stations,
            
            "shortest_path_coords": shortest_coords,
            "shortest_aqi": float(round(get_route_aqi(shortest_route), 2)),
            "shortest_km": float(round(get_route_distance_km(shortest_route), 2)),

            "balanced_path_coords": balanced_coords,
            "balanced_aqi": float(round(get_route_aqi(balanced_route), 2)),
            "balanced_km": float(round(get_route_distance_km(balanced_route), 2)),

            "extreme_path_coords": extreme_coords,
            "extreme_aqi": float(round(get_route_aqi(extreme_route), 2)),
            "extreme_km": float(round(get_route_distance_km(extreme_route), 2))
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }