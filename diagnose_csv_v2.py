import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(r"C:\Users\artur\Downloads\Biomet01_12_2024-31_12_2025.csv")

# Read data
df = pd.read_csv(csv_path, low_memory=False, skiprows=[1])

COL_TIME = "TIMESTAMP_1"
COL_P = "P_RAIN_1_1_1"
COL_T = "TA_1_1_1"
COL_RH = "RH_1_1_1"

# Basic cleaning
df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)

for col in [COL_P, COL_T, COL_RH]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Define NoData
NODATA = -9999.0

# Stats
total_points = len(df)
valid_p = df[df[COL_P] > NODATA]
valid_t = df[df[COL_T] > NODATA]

print(f"Total points: {total_points}")
print(f"Valid P points: {len(valid_p)} ({len(valid_p)/total_points*100:.2f}%)")
print(f"Valid T points: {len(valid_t)} ({len(valid_t)/total_points*100:.2f}%)")

# Sum of clipped precipitation (what the script does)
p_clipped_mm = df[COL_P].clip(lower=0.0).sum() * 1000.0
print(f"P sum (clipped): {p_clipped_mm:.2f} mm")

# Sum of valid precipitation (excluding -9999)
p_valid_mm = df[df[COL_P] > NODATA][COL_P].sum() * 1000.0
print(f"P sum (valid only): {p_valid_mm:.2f} mm")

# Temperature check (excluding -9999)
t_valid = df[df[COL_T] > NODATA][COL_T]
print(f"Temp valid range: {t_valid.min():.2f} to {t_valid.max():.2f}")
if t_valid.mean() > 200:
    print("CONFIRMED: Kelvin")
    print(f"Celsius range: {t_valid.min()-273.15:.2f} to {t_valid.max()-273.15:.2f} C")

# Missing data by month
df['is_missing_p'] = df[COL_P] <= NODATA
missing_by_month = df.groupby([df[COL_TIME].dt.year, df[COL_TIME].dt.month])['is_missing_p'].sum()
total_by_month = df.groupby([df[COL_TIME].dt.year, df[COL_TIME].dt.month]).size()
print("\nMissing P counts by month:")
print(pd.concat([missing_by_month, total_by_month], axis=1, keys=['missing', 'total']))
