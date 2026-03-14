import os
import glob
import pandas as pd
import rasterio

# Station coordinates
LAT = 55.84054151154241
LON = 48.812656762293386

IMERG_DIR = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\annual"

def extract_imerg_rfactor():
    print(f"Extracting IMERG R-factor for station at ({LAT}, {LON})")
    
    tif_files = sorted(glob.glob(os.path.join(IMERG_DIR, "*.tif")))
    results = []
    
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        year = None
        for part in filename.split('_'):
            if part.replace('.tif', '').isdigit() and len(part.replace('.tif', '')) == 4:
                year = int(part.replace('.tif', ''))
                break
                
        if year is None:
            continue
            
        with rasterio.open(tif_path) as src:
            # rasterio.index takes (lon, lat) but in CRS coordinate space
            # src.bounds: BoundingBox(left=45.9, bottom=54.7, right=50.7, top=57.0)
            row, col = src.index(LON, LAT)
            val = src.read(1)[row, col]
            
            # Print specifically to debug
            print(f"{year}: row={row}, col={col}, val={val}")
            
            # Keep negative / NoData to see what it actually is 
            results.append({
                'year': year,
                'imerg_rfactor': val
            })
            
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values('year')
        print(df)
        df.to_csv(r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\imerg_aws310_comparison.csv", index=False)
        print("Saved IMERG comparison")
    else:
        print("No IMERG data extracted.")

if __name__ == "__main__":
    extract_imerg_rfactor()
