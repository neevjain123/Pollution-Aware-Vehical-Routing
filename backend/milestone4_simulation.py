import os
import osmnx as ox
import networkx as nx
import joblib
import pandas as pd
import math
from datetime import datetime
import xgboost  # Make sure this is imported so joblib can read the model!

def run_algorithm_simulation():
    print("\n" + "="*50)
    print("ECO-ROUTE OPTIMIZER: ALGORITHM SIMULATION ENGINE")
    print("Milestone 4: Spatial Interpolation & Routing Math")
    print("="*50 + "\n")

    # --- 1. THE MACHINE LEARNING BRAIN ---
    print("[1/4] Initializing XGBoost AQI Predictor...")
    try:
        # Bulletproof path resolution: finds exactly where this Python file is on your computer
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "..", "ml_model", "xgboost_aqi_model.pkl")
        
        aqi_model = joblib.load(model_path)
        now = datetime.now()
        input_data = pd.DataFrame({'Hour': [now.hour], 'DayOfWeek': [now.weekday()]})
        predicted_base_aqi = float(aqi_model.predict(input_data)[0])
        print(f"      -> AI Forecast for Current Hour: {predicted_base_aqi:.2f} AQI\n")
        
    except Exception as e:
        print(f"      -> ❌ [WARNING] Model failed to load because: {e}")
        predicted_base_aqi = 200.0
        print("      -> Using baseline 200.0 AQI for simulation.\n")

    # --- 2. THE GRAPH GENERATION ---
    print("[2/4] Generating Graph Network...")
    G = ox.graph_from_point((28.6139, 77.2090), dist=5000, network_type='drive')
    print(f"      -> Extracted {len(G.nodes)} nodes and {len(G.edges)} edges from OpenStreetMap.\n")

    # --- 3. THE INVERSE DISTANCE WEIGHTING (IDW) ALGORITHM ---
    print("[3/4] Running Inverse Distance Weighting (IDW) Spatial Math...")
    stations = [
        {"name": "Anand Vihar", "lat": 28.6469, "lon": 77.3159, "aqi": predicted_base_aqi * 1.4}, 
        {"name": "RK Puram", "lat": 28.5632, "lon": 77.1869, "aqi": predicted_base_aqi * 1.05},
        {"name": "Punjabi Bagh", "lat": 28.6740, "lon": 77.1320, "aqi": predicted_base_aqi * 1.1},
        {"name": "ITO", "lat": 28.6284, "lon": 77.2405, "aqi": predicted_base_aqi * 1.2},
        {"name": "Dwarka", "lat": 28.5791, "lon": 77.0753, "aqi": predicted_base_aqi * 0.7}   
    ]

    for u, v, key, data in G.edges(keys=True, data=True):
        u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
        v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
        street_lat, street_lon = (u_y + v_y) / 2.0, (u_x + v_x) / 2.0

        numerator, denominator = 0, 0
        for station in stations:
            dist = math.sqrt((street_lat - station['lat'])**2 + (street_lon - station['lon'])**2) + 0.0001
            weight = 1.0 / (dist ** 2)
            numerator += weight * station['aqi']
            denominator += weight
        
        hyper_local_aqi = numerator / denominator
        data['hyper_local_aqi'] = hyper_local_aqi 
        
        data['length_balanced'] = data['length'] * ((hyper_local_aqi / 100.0) ** 2)
        data['length_extreme'] = data['length'] * ((hyper_local_aqi / 100.0) ** 10)

    print("      -> Successfully applied hyper-local AQI weights to all street edges.\n")

    # --- 4. MULTI-OBJECTIVE ROUTING SIMULATION ---
    print("[4/4] Simulating Dijkstra/A* Route Traversal...")
    start_lat, start_lon = 28.6304, 77.2177 
    end_lat, end_lon = 28.5698, 77.2201   

    orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    dest_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

    shortest_route = nx.shortest_path(G, orig_node, dest_node, weight='length')
    balanced_route = nx.shortest_path(G, orig_node, dest_node, weight='length_balanced')
    extreme_route = nx.shortest_path(G, orig_node, dest_node, weight='length_extreme')

    def get_stats(route_nodes):
        total_pollution, total_meters = 0, 0
        for i in range(len(route_nodes) - 1):
            edge_data = G.get_edge_data(route_nodes[i], route_nodes[i+1])
            if edge_data:
                first_key = list(edge_data.keys())[0]
                length = edge_data[first_key].get('length', 1.0)
                aqi = edge_data[first_key].get('hyper_local_aqi', 100.0)
                total_pollution += (aqi * length)
                total_meters += length
        avg_aqi = total_pollution / total_meters if total_meters > 0 else 0
        return total_meters / 1000.0, avg_aqi

    short_km, short_aqi = get_stats(shortest_route)
    bal_km, bal_aqi = get_stats(balanced_route)
    ext_km, ext_aqi = get_stats(extreme_route)

    print("\n" + "="*50)
    print("🎯 SIMULATION RESULTS: MULTI-OBJECTIVE COMPARISON")
    print("="*50)
    print(f"🔴 Standard Route (Distance Only):")
    print(f"   -> Distance: {short_km:.2f} km")
    print(f"   -> Average Exposure AQI: {short_aqi:.2f}\n")

    print(f"🔵 Balanced Eco-Route (Alpha = 2):")
    print(f"   -> Distance: {bal_km:.2f} km")
    print(f"   -> Average Exposure AQI: {bal_aqi:.2f}\n")

    print(f"🟢 Extreme Eco-Route (Alpha = 10):")
    print(f"   -> Distance: {ext_km:.2f} km")
    print(f"   -> Average Exposure AQI: {ext_aqi:.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_algorithm_simulation()