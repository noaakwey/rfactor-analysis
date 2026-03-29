import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path(r"C:\Users\artur\Downloads\Biomet01_12_2024-31_12_2025.csv")

# Read data
df = pd.read_csv(csv_path, low_memory=False, skiprows=[1]) # Skip the unit row

# Check columns and units
print("Columns:", df.columns.tolist())
print("First row values:")
print(df.iloc[0])

# Time analysis
COL_TIME = "TIMESTAMP_1" 
df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME)

start_date = df[COL_TIME].min()
end_date = df[COL_TIME].max()
duration = end_date - start_date
expected_rows = (duration.total_seconds() / 1800) + 1
actual_rows = len(df)

print(f"\nPeriod: {start_date} to {end_date}")
print(f"Expected rows (30 min): {expected_rows}")
print(f"Actual rows: {actual_rows}")
print(f"Coverage: {actual_rows / expected_rows * 100:.2f}%")

# Precipitation analysis
COL_P = "P_RAIN_1_1_1"
df[COL_P] = pd.to_numeric(df[COL_P], errors="coerce").fillna(0.0)
p_sum_m = df[COL_P].sum()
p_sum_mm = p_sum_m * 1000.0
print(f"\nTotal Precipitation: {p_sum_m:.4f} m ({p_sum_mm:.2f} mm)")

# Temperature analysis
COL_T = "TA_1_1_1"
df[COL_T] = pd.to_numeric(df[COL_T], errors="coerce")
print(f"\nTemperature range (Raw): {df[COL_T].min():.2f} to {df[COL_T].max():.2f}")
if df[COL_T].mean() > 200:
    print("Detected KELVIN. Max in Celsius:", df[COL_T].max() - 273.15)
    print("Min in Celsius:", df[COL_T].min() - 273.15)
else:
    print("Detected CELSIUS.")

# Check for gaps in time
df['diff'] = df[COL_TIME].diff()
gaps = df[df['diff'] > pd.Timedelta(minutes=30)]
if not gaps.empty:
    print(f"\nFound {len(gaps)} gaps larger than 30 minutes.")
    print("Largest gap:", gaps['diff'].max())
    print("Top 5 gaps:")
    print(gaps.sort_values('diff', ascending=False).head(5)[[COL_TIME, 'diff']])

# Monthly precip
df['month'] = df[COL_TIME].dt.month
df['year'] = df[COL_TIME].dt.year
monthly = df.groupby(['year', 'month'])[COL_P].sum() * 1000
print("\nMonthly Precipitation (mm):")
print(monthly)
