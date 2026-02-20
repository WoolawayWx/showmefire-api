import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from core.database import get_db_path
import sqlite3

def main():
	# 1. Load obs file
	obs_path = Path('archive/raw_data/raw_data_20260217.json')
	with open(obs_path) as f:
		obs_data = json.load(f)

	# 2. Insert/update stations
	db_path = get_db_path()
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	for stn in obs_data['STATION']:
		stn_id = stn['STID']
		lat = float(stn['LATITUDE'])
		lon = float(stn['LONGITUDE'])
		name = stn.get('NAME', '')
		state = stn.get('STATE', '')
		cursor.execute('''
			INSERT OR REPLACE INTO stations (id, name, lat, lon, state)
			VALUES (?, ?, ?, ?, ?)
		''', (stn_id, name, lat, lon, state))

	conn.commit()

	# 3. Load HRRR grid
	lat_grid = np.load('data/lat_grid.npy')
	lon_grid = np.load('data/lon_grid.npy')

	# 4. For each station, find nearest grid point and update station_grid_indices
	for stn in obs_data['STATION']:
		stn_id = stn['STID']
		lat = float(stn['LATITUDE'])
		lon = float(stn['LONGITUDE'])
		# Find nearest grid point
		dist = (lat_grid - lat)**2 + (lon_grid - lon)**2
		y, x = np.unravel_index(np.argmin(dist), dist.shape)
		# Insert or update station_grid_indices
		cursor.execute('''
			INSERT OR REPLACE INTO station_grid_indices (station_id, x, y, lat, lon)
			VALUES (?, ?, ?, ?, ?)
		''', (stn_id, int(x), int(y), lat, lon))

	conn.commit()
	conn.close()

if __name__ == '__main__':
	main()
