import requests
import time

API_URL_DYNAMIC = "http://127.0.0.1:8000/api/v1/dynamic_route"
API_URL_STATIC = "http://127.0.0.1:8000/api/get_route"


start_coord = {"lat": 28.6139, "lon": 77.2090} # India Gate
end_coord = {"lat": 28.6469, "lon": 77.3159}   # Anand Vihar

print("=====================================================")
print("🚗 STARTING PRECISION TELEMETRY SIMULATOR 🚗")
print("=====================================================\n")


print("⏳ Fetching the exact AI-calculated route...")
warmup_params = {
    "start_lat": start_coord["lat"],
    "start_lon": start_coord["lon"],
    "end_lat": end_coord["lat"],
    "end_lon": end_coord["lon"]
}

try:
    response = requests.get(API_URL_STATIC, params=warmup_params)
    data = response.json()
    
    if data.get("status") != "success":
        print("❌ Error getting route:", data.get("message"))
        exit()
        
    
    exact_route_coords = [{"lat": coord[0], "lon": coord[1]} for coord in data["extreme_path_coords"]]
    
    
    mock_gps_path = exact_route_coords[::15] 
    
    
    mock_gps_path.append(end_coord)
    
    print(f"✅ Route locked! Extracted {len(mock_gps_path)} precision waypoints to drive.\n")
    
except Exception as e:
    print(f"❌ Server offline. Is Uvicorn running? Error: {e}")
    exit()


current_expected_cost = 999999999999.0 

for i, position in enumerate(mock_gps_path):
    print(f"[TICK {i+1}/{len(mock_gps_path)}] Car Location: Lat {position['lat']:.4f}, Lon {position['lon']:.4f}")
    
    payload = {
        "current_lat": position["lat"],
        "current_lon": position["lon"],
        "end_lat": end_coord["lat"],
        "end_lon": end_coord["lon"],
        "current_route_cost": current_expected_cost
    }
    
    try:
        res = requests.post(API_URL_DYNAMIC, json=payload)
        if res.status_code == 200:
            resp_data = res.json()
            
            if resp_data.get("status") == "REROUTE_TRIGGERED":
                print(f"   🚨 REROUTING! New Cost: {resp_data['cost']:.2f}")
                current_expected_cost = resp_data["cost"] 
            elif resp_data.get("status") == "ROUTE_STABLE":
                print(f"   ✅ ROUTE STABLE. Expected Cost: {resp_data['cost']:.2f}")
        else:
            print(f"   ❌ HTTP Error: {res.status_code}")
            
    except Exception as e:
        print("   ❌ FATAL CONNECTION ERROR")
        break
        
    print("-" * 50)
    time.sleep(3)

print("🏁 DESTINATION REACHED.")