import pandas as pd
import numpy as np

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
print("Loading data...")
df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_raw'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_warm = df[(df['timestamp'].dt.year == 2023) & (df['timestamp'].dt.month >= 4) & (df['timestamp'].dt.month <= 10)].copy()
df_warm = df_warm.sort_values('timestamp').reset_index(drop=True)

# Math model to recover true precipitation from wind-noisy load cell raw weight:

# 1. Calculate a 10-minute rolling median (to reject sharp wind spikes).
df_warm = df_warm.set_index('timestamp')
df_warm['P_smooth'] = df_warm['P_raw'].rolling('10min').median()
df_warm = df_warm.reset_index()

# 2. Differentiate the smooth signal
df_warm['P_diff'] = df_warm['P_smooth'].diff().fillna(0)

# 3. Suppress tiny noise variations
# 0.01 mm per minute is 0.6 mm/h (light drizzle).
threshold = 0.01
df_warm.loc[df_warm['P_diff'] < threshold, 'P_diff'] = 0.0

# 4. Suppress massive bucket emptying drops (e.g. > -5 mm) 
# Oh wait, we already thresholded < 0.05 to 0.0, so drops are already 0.0.

# 5. Sometimes, after an empty event, the smooth signal slowly recovers.
# Let's cleanly handle resets in the original smooth signal.
resets = df_warm['P_smooth'].diff() < -2.0
df_warm.loc[resets, 'P_diff'] = 0.0

# Let's sum the filtered precipitation!
total_DSP_rain = df_warm['P_diff'].sum()

print(f"\n--- DSP Filtered Precipitation (Warm Season 2023) ---")
print(f"Total extracted rain using 10-min rolling median on raw load cell: {total_DSP_rain:.2f} mm")

# Monthly breakdown of the DSP rain
print("\nMonthly breakdown:")
df_warm['month'] = df_warm['timestamp'].dt.month
monthly = df_warm.groupby('month')['P_diff'].sum()
for m in range(4, 11):
    print(f"Month {m:02d}: {monthly.get(m, 0.0):.2f} mm")

print("\nKazan Synoptic Station (for reference):")
print("Month 04: 12.9 mm\nMonth 05: 56.0 mm\nMonth 06: 9.3 mm\nMonth 07: 55.6 mm\nMonth 08: 9.0 mm\nMonth 09: 7.2 mm\nMonth 10: 92.8 mm")
print("Total Synoptic (Apr-Oct): 242.8 mm")
