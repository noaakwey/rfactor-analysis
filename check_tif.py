import rasterio
import numpy as np
import os

path = r"Z:/R_imerg_2021.tif"
if not os.path.exists(path):
    # Try local path if Z: is not substed
    path = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg\annual\R_imerg_2021.tif"

try:
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata or 0
        valid = data[data > nodata]
        if len(valid) > 0:
            print(f"Mean (valid): {np.mean(valid):.2f}")
            print(f"Max: {np.max(valid):.2f}")
        else:
            print("No valid data found.")
except Exception as e:
    print(f"Error: {e}")
