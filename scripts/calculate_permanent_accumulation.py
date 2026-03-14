import pandas as pd

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
print("Loading data...")
df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_raw'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_warm = df[(df['timestamp'].dt.year == 2023) & (df['timestamp'].dt.month >= 4) & (df['timestamp'].dt.month <= 10)].copy()
df_warm = df_warm.sort_values('timestamp').reset_index(drop=True)

# To find TRUE manual emptying events (vs temporary wind lift):
# We look for a drop, where the values AFTER the drop stay low for at least an hour.
df_warm['rolling_max_1h_after'] = df_warm['P_raw'].shift(-60).rolling(60).max()

resets = []
for i in range(1, len(df_warm) - 60):
    val_prev = df_warm['P_raw'].iloc[i-1]
    val_curr = df_warm['P_raw'].iloc[i]
    if val_prev - val_curr > 5.0:
        # Check if it bounced back within an hour
        max_after = df_warm['P_raw'].iloc[i:i+60].max()
        if max_after < val_prev - 2.0:
            # It's a real permanent empty!
            # But wait, we only want to trigger this once per empty event.
            if not resets or i - resets[-1]['idx'] > 60:
                resets.append({
                    'idx': i,
                    'timestamp': df_warm['timestamp'].iloc[i],
                    'drop_amount': val_prev - val_curr
                })

# Actually, the simplest way to get the total accumulated rain is:
# Rain = Sum of (Permanent Max just before empty - Permanent Min just after empty) + Final value
# To do this robustly: track the running absolute maximum of the bucket.
# But if it empties, we reset the running max baseline.

total_rain = 0.0
current_baseline = 0.0
max_so_far = 0.0

print(f"Start value: {df_warm['P_raw'].iloc[0]}")
for i in range(len(df_warm)):
    val = df_warm['P_raw'].iloc[i]
    if pd.isna(val): continue
    
    if val > max_so_far:
        max_so_far = val
        
    # Detect a permanent reset: 
    # If the current value is < max_so_far - 10 mm, 
    # AND the maximum over the next 60 minutes is also < max_so_far - 10 mm.
    # To save time, we already computed df_warm['rolling_max_1h_after'] which is the max from i to i+60
    if i < len(df_warm) - 60:
        max_next_hour = df_warm['rolling_max_1h_after'].iloc[i]
        if val < max_so_far - 10.0 and max_next_hour < max_so_far - 10.0:
            # OK, the bucket was permanently emptied.
            print(f"Emptied at {df_warm['timestamp'].iloc[i]}: Max was {max_so_far:.2f}, dropped to {val:.2f}, next 1h max {max_next_hour:.2f}")
            total_rain += max_so_far
            max_so_far = val

total_rain += max_so_far
print(f"\nTotal physical rain (Warm Season): {total_rain:.2f} mm")
