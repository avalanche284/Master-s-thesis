# Szymon Bartoszewicz
# WSB University 2023
# The purpose of this code is to fetch selected weather data for Warsaw, 364 days back to this day.

import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_weather_data(lat, lon, start, end, api_key):
    url = f"https://history.openweathermap.org/data/2.5/history/city?lat={lat}&lon={lon}&type=hour&start={start}&end={end}&appid={api_key}"
    response = requests.get(url)
    data = response.json()

    weather_data = {}
    for item in data['list']:
        timestamp = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
        temp = item['main']['temp']
        pressure = item['main']['pressure']
        humidity = item['main']['humidity']
        wind_speed = item['wind']['speed']
        clouds = item['clouds']['all']
        weather_data[timestamp] = [timestamp, temp, pressure, humidity, wind_speed, clouds]

    return weather_data

api_key = "REMOVED"
lat = "52.2297" # Warsaw
lon = "21.0122"

# Calculate start and end timestamps for the last 364 days
end = int(datetime.utcnow().timestamp())
start = int((datetime.utcnow() - timedelta(days=364)).timestamp())

weather_data = {}
current_start = start

# Fetch data in 7-day intervals cause OPenWeather API only allows for 7 days intervals and there must be
# more API calls
while current_start < end:
    current_end = min(current_start + 7 * 24 * 3600, end)
    weather_data.update(fetch_weather_data(lat, lon, current_start, current_end, api_key))
    current_start = current_end

df = pd.DataFrame(weather_data.values(), columns=['timestamp', 'temp', 'pressure', 'humidity', 'wind_speed', 'clouds'])

# Convert timestamp to datetime and set it as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Create a complete datetime range and merge with fetched data
date_range = pd.date_range(start=datetime.fromtimestamp(start).date(), end=datetime.fromtimestamp(end).date(), freq='D')
date_range_df = pd.DataFrame(date_range, columns=['timestamp']).set_index('timestamp')
merged_data = date_range_df.join(df, how='left')

#  None values for missing data
merged_data.to_csv('warsaw_daily_weather_data_364_days.csv')

print("Saved as 'warsaw_daily_weather_data_364_days.csv'.")
