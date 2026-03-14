import pandas as pd

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"

df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_cumulative'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_2023 = df[df['timestamp'].dt.year == 2023].copy()
df_2023 = df_2023.sort_values('timestamp')

# Filter exactly August 21st, 2023 (the big storm we found in calc_ei30)
storm = df_2023[(df_2023['timestamp'] >= pd.to_datetime('2023-08-21 12:00:00+00:00')) & 
                (df_2023['timestamp'] <= pd.to_datetime('2023-08-21 16:00:00+00:00'))].copy()

storm['P_1min'] = storm['P_cumulative'].diff()
storm['P_1min_v2'] = storm['Pluvio2_1.value2'].diff()

print("August 21st Storm Data:")
storm_rain = storm[storm['P_1min'] > 0]
if len(storm_rain) > 0:
    print(storm_rain[['timestamp', 'P_cumulative', 'P_1min', 'Pluvio2_1.value2', 'P_1min_v2', 'Pluvio_intensity.value']].head(30))

print("\n--- Summary of storm ---")
print(f"Total positive value1: {storm.loc[storm['P_1min'] > 0, 'P_1min'].sum():.1f} mm")
print(f"Total positive value1 (<= 5mm): {storm.loc[(storm['P_1min'] > 0) & (storm['P_1min'] <= 5.0), 'P_1min'].sum():.1f} mm")
print(f"Total positive value2: {storm.loc[storm['P_1min_v2'] > 0, 'P_1min_v2'].sum():.1f} mm")

# What about the massive noise? Let's find one
noise = df_2023[df_2023['P_cumulative'].diff() > 20.0].head(1)
if len(noise) > 0:
    t_noise = noise['timestamp'].iloc[0]
    print(f"\n--- Example of massive noise at {t_noise} ---")
    window = df_2023[(df_2023['timestamp'] >= t_noise - pd.Timedelta(minutes=5)) & 
                     (df_2023['timestamp'] <= t_noise + pd.Timedelta(minutes=5))]
    print(window[['timestamp', 'P_cumulative', 'Pluvio2_1.value2']])
