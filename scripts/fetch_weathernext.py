"""Run with the dedicated WeatherNext/Zarr-v3 Python environment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.fire_weather_hybrid.weathernext import main

if __name__ == "__main__":
    raise SystemExit(main())
