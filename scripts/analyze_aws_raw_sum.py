import pandas as pd
import numpy as np

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"

df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_cumulative'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

df_2023 = df[df['timestamp'].dt.year == 2023].copy()
df_2023 = df_2023.sort_values('timestamp')

# Filter exactly warm season
warm_mask = (df_2023['timestamp'].dt.month >= 4) & (df_2023['timestamp'].dt.month <= 10)
df_warm = df_2023[warm_mask].copy()

# Cumulative bucket positive differences
df_warm['P_1min'] = df_warm['P_cumulative'].diff()
raw_positive = df_warm.loc[df_warm['P_1min'] > 0, 'P_1min'].sum()
print(f"Total positive bucket increases (Warm Season): {raw_positive:.1f} mm")

spikes = df_warm[df_warm['P_1min'] > 5.0]
print(f"Number of >5mm/min spikes in warm season: {len(spikes)}")
print(f"Precipitation from these spikes: {spikes['P_1min'].sum():.1f} mm")

# Remove spikes
clean_positive = df_warm.loc[(df_warm['P_1min'] > 0) & (df_warm['P_1min'] <= 5.0), 'P_1min'].sum()
print(f"Clean continuous positive bucket increases (<= 5mm/min): {clean_positive:.1f} mm")

# What about Pluvio2_1.value2 ? (This might be the daily/hourly cumulative or smoothed value?)
df_warm['P_1min_v2'] = df_warm['Pluvio2_1.value2'].diff()
v2_positive = df_warm.loc[df_warm['P_1min_v2'] > 0, 'P_1min_v2'].sum()
print(f"\nValue2 positive increases: {v2_positive:.1f} mm")

# Intensity
intensity_sum = (df_warm['Pluvio_intensity.value'] / 60.0).sum()
print(f"Derived from Pluvio_intensity (mm/h) taking 1 min: {intensity_sum:.1f} mm")

# Look closely at where Kazan had large rain (October = 92 mm, AWS = 7.2 mm? August = Kazan 9mm, AWS = 23 mm?)
aug = df_warm[df_warm['timestamp'].dt.month == 8]
aug_pos = aug.loc[(aug['P_1min'] > 0) & (aug['P_1min'] <= 5.0), 'P_1min'].sum()
print(f"\nAugust clean positive: {aug_pos:.1f} mm")

oct_df = df_warm[df_warm['timestamp'].dt.month == 10]
oct_pos = oct_df.loc[(oct_df['P_1min'] > 0) & (oct_df['P_1min'] <= 5.0), 'P_1min'].sum()
print(f"October clean positive: {oct_pos:.1f} mm")
