import pandas as pd

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
print("Loading data...")
df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_cumulative'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_aug = df[df['timestamp'].dt.month == 8].copy()
df_aug = df_aug.sort_values('timestamp')

df_aug['P_1min'] = df_aug['P_cumulative'].diff()
df_aug['P_1min_v2'] = df_aug['Pluvio2_1.value2'].diff()
df_aug['clean_P'] = df_aug['P_1min']
df_aug.loc[(df_aug['clean_P'] < 0) | (df_aug['clean_P'] > 5.0), 'clean_P'] = 0.0

# Find the hour with the maximum clean rain
hourly = df_aug.set_index('timestamp').resample('1h')['clean_P'].sum()
max_hour = hourly.idxmax()
print(f"Max rain hour: {max_hour} with {hourly.max():.2f} mm")

if hourly.max() > 0:
    start = max_hour - pd.Timedelta(minutes=10)
    end = max_hour + pd.Timedelta(minutes=70)
    storm = df_aug[(df_aug['timestamp'] >= start) & (df_aug['timestamp'] <= end)]
    
    print("\nStorm profile (values > 0):")
    storm_active = storm[(storm['clean_P'] > 0) | (storm['P_1min_v2'] > 0)]
    print(storm_active[['timestamp', 'P_cumulative', 'P_1min', 'clean_P', 'Pluvio2_1.value2', 'P_1min_v2', 'Pluvio_intensity.value']].head(50))
    
    print(f"\nStorm total clean P1 (<= 5mm): {storm['clean_P'].sum():.2f} mm")
    v2_pos = storm.loc[storm['P_1min_v2'] > 0, 'P_1min_v2'].sum()
    print(f"Storm total P2 (value2): {v2_pos:.2f} mm")
