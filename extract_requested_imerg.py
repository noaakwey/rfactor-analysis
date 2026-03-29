import os
import glob
import pandas as pd
import rasterio

# Prompt coordinates (STATION)
LAT = 55.84325307292437
LON = 48.79867579590198

# Directory where IMERG R-factor TIFs are stored
# Based on 07_compare_k05_k082.py
IMERG_DIR = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg\annual"

def extract_imerg_rfactor():
    print(f"Extracting IMERG R-factor for station at ({LAT}, {LON})")
    
    tif_files = sorted(glob.glob(os.path.join(IMERG_DIR, "*.tif")))
    results = []
    
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        year = None
        # Extract year from filename like R_imerg_2024.tif
        for part in filename.replace('.tif', '').split('_'):
            if part.isdigit() and len(part) == 4:
                year = int(part)
                break
                
        if year is None:
            continue
            
        with rasterio.open(tif_path) as src:
            try:
                row, col = src.index(LON, LAT)
                val = src.read(1)[row, col]
                results.append({'year': year, 'imerg_rfactor': val})
            except Exception as e:
                print(f"Error for {year}: {e}")
            
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('year')
        print(df)
        df.to_csv(r"d:\Cache\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\imerg_comparison_requested_coords.csv", index=False)
        print("Saved IMERG comparison for requested coords")
    else:
        print("No IMERG data extracted.")

if __name__ == "__main__":
    extract_imerg_rfactor()
