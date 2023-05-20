# Szymon Bartoszewicz
# WSB University 2023
# The purpose of this code is to fetch daily air pollution data for Warsaw, 365 days back to this day.
# The coode adds None to rows where data is not available.

import requests
import csv
import time
from datetime import datetime, timedelta
from collections import defaultdict

API_KEY = "removed"

cities = [
    {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122},
]

# Function to fetch air pollution data
def get_air_pollution_data(lat, lon, start, end, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start}&end={end}&appid={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()["list"]
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

# Function to aggregate raw data into daily averages
def aggregate_daily_data(data, start, end):
    daily_data = defaultdict(lambda: defaultdict(list))

    # Group data by date and pollutant
    for entry in data:
        date = datetime.fromtimestamp(entry["dt"]).strftime("%Y-%m-%d")
        for key, value in entry["components"].items():
            daily_data[date][key].append(value)

    # Prepare a complete list of dates in the range
    date_range = [start + timedelta(days=x) for x in range((end - start).days + 1)]

    # Calculate average values for each day
    aggregated_data = []
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        components = daily_data[date_str]
        if components:
            aggregated_components = {key: sum(values) / len(values) for key, values in components.items()}
        else:
            aggregated_components = {key: None for key in ["pm2_5", "pm10", "o3", "no2", "so2", "co"]}

        aggregated_data.append({"date": date_str, **aggregated_components})

    return aggregated_data

# Function to save the data to a CSV file
def save_to_csv(data, filename):
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["city", "date", "pm2_5", "pm10", "o3", "no2", "so2", "co"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in data:
            writer.writerow(row)

def main():
    data = []
    end = int(time.mktime((datetime.now() - timedelta(days=1)).timetuple()))
    start = int(time.mktime((datetime.now() - timedelta(days=365)).timetuple()))
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now() - timedelta(days=1)

    for city in cities:
        print(f"Fetching data for {city['name']}...")
        air_pollution_data = get_air_pollution_data(city["lat"], city["lon"], start, end, API_KEY)


        if air_pollution_data:
            daily_data = aggregate_daily_data(air_pollution_data, start_date, end_date)

            for entry in daily_data:
                row = {
                    "city": city["name"],
                    "date": entry["date"],
                    "pm2_5": entry["pm2_5"],
                    "pm10": entry["pm10"],
                    "o3": entry["o3"],
                    "no2": entry["no2"],
                    "so2": entry["so2"],
                    "co": entry["co"]
                }
                data.append(row)

    save_to_csv(data, "warsaw_daily_air_pollution_data.csv")
    print("Saved to warsaw_daily_air_pollution_data.csv")

if __name__ == "__main__":
    main()
