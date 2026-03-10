import osmnx as ox
import networkx as nx
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the AQI Routing Backend!"}

@app.get("/api/get_route")
def get_real_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float):
    try:
        print(f"Calculating route from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
        
        # 1. Create a bounding box around the two clicks
        north = max(start_lat, end_lat) + 0.01
        south = min(start_lat, end_lat) - 0.01
        east = max(start_lon, end_lon) + 0.01
        west = min(start_lon, end_lon) - 0.01

        # 2. Download the drivable street network
        print("Downloading street network from OpenStreetMap...")
        # THE FIX: OSMnx 2.0+ strictly requires (west, south, east, north) order!
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')

        # 3. Snap your clicked coordinates to the nearest actual road intersections
        orig_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
        dest_node = ox.distance.nearest_nodes(G, X=end_lon, Y=end_lat)

        # 4. Calculate the shortest street path
        print("Calculating shortest path...")
        route = nx.shortest_path(G, orig_node, dest_node, weight='length')

        # 5. Convert the path of intersections back into Lat/Lon coordinates for Leaflet
        path_coords = []
        for node in route:
            lat = G.nodes[node]['y']
            lon = G.nodes[node]['x']
            path_coords.append([lat, lon])

        print("Route successfully sent to the map!")
        return {
            "status": "success",
            "message": "Real street route calculated!",
            "path_coordinates": path_coords
        }
        
    except Exception as e:
        print(f"Routing error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "path_coordinates": []
        }