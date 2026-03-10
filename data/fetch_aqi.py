import requests
import csv
import os

def get_delhi_aqi():
    city = "delhi"
    token = "demo" # Swap this for a real token from aqicn.org later
    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    
    # We will save the data right here in your data folder
    csv_filename = "delhi_aqi_log.csv"

    try:
        print(f"Fetching real-time AQI data for {city.upper()}...")
        response = requests.get(url)
        response.raise_for_status() 
        
        data = response.json()
        
        if data['status'] == 'ok':
            current_aqi = data['data']['aqi']
            primary_pollutant = data['data']['dominentpol']
            timestamp = data['data']['time']['s']
            
            # Check if the file already exists so we know if we need to add the header row
            file_exists = os.path.isfile(csv_filename)
            
            # Open the CSV in 'append' mode so we don't overwrite older data
            with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Write the column headers if this is a brand new file
                if not file_exists:
                    writer.writerow(['Timestamp', 'City', 'AQI', 'Primary_Pollutant'])
                
                # Write the actual data row
                writer.writerow([timestamp, city, current_aqi, primary_pollutant])
            
            print(f"Success! Saved AQI: {current_aqi} to {csv_filename}")
            
        else:
            print("Error: Could not retrieve data from the API.")
            
    except requests.exceptions.RequestException as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    get_delhi_aqi()