import xarray as xr
import numpy as np

def test_hrrr_spatial_variance():
    path = "cache/hrrr/hrrr_20260103_12z_f04-15.nc"
    ds = xr.open_dataset(path, drop_variables=['step'])
    
    # 1. Check specific points
    # KC (983, 553) vs STL (1110, 542)
    kc_temp = float(ds.t2m.isel(step=0, x=983, y=553).values) - 273.15
    stl_temp = float(ds.t2m.isel(step=0, x=1110, y=542).values) - 273.15
    
    print(f"📍 KC Temperature:  {kc_temp:.4f} C")
    print(f"📍 STL Temperature: {stl_temp:.4f} C")
    
    # 2. Check the "Variety" of the entire file
    t_min = float(ds.t2m.isel(step=0).min().values) - 273.15
    t_max = float(ds.t2m.isel(step=0).max().values) - 273.15
    t_std = float(ds.t2m.isel(step=0).std().values)
    
    print(f"\n🌎 Full Grid Stats:")
    print(f"   - Min Temp: {t_min:.2f} C")
    print(f"   - Max Temp: {t_max:.2f} C")
    print(f"   - Standard Deviation: {t_std:.4f}")
    
    if t_std == 0:
        print("\n❌ ALARM: The entire file has identical values. The data source is 'flat'.")
    else:
        print("\n✅ SUCCESS: The file has spatial variety. The bug is in the extraction loop logic.")

if __name__ == "__main__":
    test_hrrr_spatial_variance()