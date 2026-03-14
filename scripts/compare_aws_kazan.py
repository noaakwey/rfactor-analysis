import pandas as pd
import numpy as np
import os

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
kazan_synoptic_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib\output\calib_imerg_newtest\station_27595_calib.csv"

# 1. Calculate AWS total precipitation
print("Loading AWS data...")
df_aws = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
# Rename required columns
df_aws.rename(columns={
    'datetime_utc': 'timestamp',
    'Pluvio2_1.value1': 'P_cumulative',
    'HMP155.T': 'temp_c'
}, inplace=True)
df_aws['timestamp'] = pd.to_datetime(df_aws['timestamp'], utc=True)
df_aws['P_1min'] = df_aws['P_cumulative'].diff().fillna(0)

# Filter 2023
df_2023 = df_aws[df_aws['timestamp'].dt.year == 2023].copy()

# Apply the same filters as in calc_ei30
df_2023.loc[df_2023['P_1min'] > 5.0, 'P_1min'] = 0.0
df_2023.loc[df_2023['P_1min'] < 0, 'P_1min'] = 0.0

# Total precipitation in 2023
total_p = df_2023['P_1min'].sum()
print(f"Total raw AWS precipitation in 2023 (ignoring <0 and >5mm/min): {total_p:.1f} mm")

# Warm season precipitation (Apr-Oct)
warm_mask = (df_2023['timestamp'].dt.month >= 4) & (df_2023['timestamp'].dt.month <= 10)
total_p_warm = df_2023.loc[warm_mask, 'P_1min'].sum()
print(f"AWS precipitation in warm season (Apr-Oct, 2023): {total_p_warm:.1f} mm")

# Liquid precipitation (T > 0 °C)
liquid_mask = warm_mask & (df_2023['temp_c'] > 0)
total_p_liquid = df_2023.loc[liquid_mask, 'P_1min'].sum()
print(f"AWS liquid precipitation in warm season (Apr-Oct, T>0, 2023): {total_p_liquid:.1f} mm")

# For erosive events
event_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_erosivity_events.csv"
df_events = pd.read_csv(event_file)
print(f"Precipitation strictly belonging to erosive events (RUSLE2 definition): {df_events['P_total_mm'].sum():.1f} mm")

# 2. Complete Kazan comparisons
print("-" * 50)
print("Loading Kazan 3-hourly Synoptic data (WMO: 27595)...")
df_kazan = pd.read_csv(kazan_synoptic_file, parse_dates=['datetime'])
df_kazan['datetime'] = pd.to_datetime(df_kazan['datetime'], utc=True)

df_k2023 = df_kazan[df_kazan['datetime'].dt.year == 2023].copy()
kwarm_mask = (df_k2023['datetime'].dt.month >= 4) & (df_k2023['datetime'].dt.month <= 10)

kazan_total_warm = df_k2023.loc[kwarm_mask, 'P_corrected_mm'].sum()
kazan_total_year = df_k2023['P_corrected_mm'].sum()

print(f"Kazan Synoptic total precipitation in 2023: {kazan_total_year:.1f} mm")
print(f"Kazan Synoptic warm season (Apr-Oct, 2023): {kazan_total_warm:.1f} mm")
print("-" * 50)
print(f"Difference (AWS - Kazan) for warm season: {total_p_warm - kazan_total_warm:.1f} mm")

print("\n--- Monthly Breakdown (Warm Season 2023) ---")
aws_monthly = df_2023.loc[warm_mask].groupby(df_2023['timestamp'].dt.month)['P_1min'].sum()
kazan_monthly = df_k2023.loc[kwarm_mask].groupby(df_k2023['datetime'].dt.month)['P_corrected_mm'].sum()

for m in range(4, 11):
    aws_m = aws_monthly.get(m, 0.0)
    kazan_m = kazan_monthly.get(m, 0.0)
    print(f"Month {m:02d}: AWS = {aws_m:>6.1f} mm | Kazan = {kazan_m:>6.1f} mm | Diff = {aws_m - kazan_m:>6.1f} mm")
