import requests
import json
from datetime import datetime, timedelta
import pandas as pd

def fetch_station_data(station_id, start_time, end_time):
	# Example: Replace with your actual API endpoint and params
	url = f"http://localhost:8000/api/obs/{station_id}"
	params = {
		'start': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
		'end': end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
	}
	resp = requests.get(url, params=params)
	resp.raise_for_status()
	return pd.DataFrame(resp.json())

def main():
	# List of station IDs (replace with your actual station list)
	stations = ['STATION1', 'STATION2', 'STATION3']

	# Set time range for today (UTC)
	now = datetime.utcnow()
	start_time = datetime(now.year, now.month, now.day)
	end_time = start_time + timedelta(days=1)

	results = {}
	for station_id in stations:
		try:
			df = fetch_station_data(station_id, start_time, end_time)
			if df.empty:
				continue

			# Track min/max/mean for each parameter
			stats = {}
			for param in ['humidity', 'fuel_moisture', 'wind_speed', 'temperature']:
				if param in df.columns:
					stats[f'{param}_min'] = float(df[param].min())
					stats[f'{param}_max'] = float(df[param].max())
					stats[f'{param}_mean'] = float(df[param].mean())
				else:
					stats[f'{param}_min'] = None
					stats[f'{param}_max'] = None
					stats[f'{param}_mean'] = None

			# Optionally, add station metadata
			stats['station_id'] = station_id
			results[station_id] = stats
		except Exception as e:
			print(f"Failed to fetch/process data for {station_id}: {e}")

	# Save to JSON
	with open('daily_station_summary.json', 'w') as f:
		json.dump(results, f, indent=2)

if __name__ == '__main__':
	main()
