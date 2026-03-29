import os
import glob
import rasterio
import numpy as np
from tqdm import tqdm

SRC_DIR = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k05\annual"
OUT_DIR = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_v2\annual"
SCALE_FACTOR = 3.98678 # (Mean V2 452.5 / Mean Raw 113.5)

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

tifs = glob.glob(os.path.join(SRC_DIR, "R_imerg_*.tif"))
print(f"Scaling {len(tifs)} TIFs by {SCALE_FACTOR:.4f}...")

for tif in tqdm(tifs):
    with rasterio.open(tif) as src:
        data = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        
        # Scale only positive values
        mask = (data > 0)
        data[mask] *= SCALE_FACTOR
        
        out_path = os.path.join(OUT_DIR, os.path.basename(tif))
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)

print("Done.")
