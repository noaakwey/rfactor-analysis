import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
print("Loading data...")
df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_raw'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_warm = df[(df['timestamp'].dt.year == 2023) & (df['timestamp'].dt.month >= 4) & (df['timestamp'].dt.month <= 10)].copy()
df_warm = df_warm.sort_values('timestamp').reset_index(drop=True)

# 1. Plot the raw bucket weight over the season to visually see the water filling up and emptying
first_val = df_warm['P_raw'].iloc[0]
last_val = df_warm['P_raw'].iloc[-1]

print(f"Bucket weight start (Apr): {first_val:.2f} mm")
print(f"Bucket weight end (Oct): {last_val:.2f} mm")

# Identify all massive drops (bucket emptying events).
# A physical bucket empty event is usually an immediate drop of > 5 mm.
df_warm['diff'] = df_warm['P_raw'].diff()
resets = df_warm[df_warm['diff'] < -5.0]

print(f"\nFound {len(resets)} manual bucket emptying events:")
total_emptied = 0
for idx, row in resets.iterrows():
    # To be extremely precise, we take the bucket weight just BEFORE the reset
    # and subtract the weight just AFTER the reset.
    val_before = df_warm['P_raw'].iloc[idx-1]
    val_after = row['P_raw']
    drop = val_before - val_after
    print(f"  {row['timestamp']} | Dropped from {val_before:.2f} to {val_after:.2f} (Amount emptied: {drop:.2f} mm)")
    total_emptied += drop

# True total precipitation = (Final - Initial) + Total Emptied
true_P_total = (last_val - first_val) + total_emptied

print(f"\n--- TRUE SEASONAL PRECIPITATION (MATH MODEL) ---")
print(f"P_true = ({last_val:.2f} - {first_val:.2f}) + {total_emptied:.2f} = {true_P_total:.2f} mm")

# Let's compare this mathematical total to Kazan synoptic!
print(f"\nKazan Synoptic Station 2023 Warm Season: 242.8 mm")
print(f"Difference: {true_P_total - 242.8:.2f} mm")
