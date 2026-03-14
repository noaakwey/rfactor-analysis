import pandas as pd
import numpy as np

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"

print("Loading AWS data...")
df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_cumulative'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

df_2023 = df[df['timestamp'].dt.year == 2023].copy()
df_2023 = df_2023.sort_values('timestamp')

# Calculate time diff
df_2023['time_diff'] = df_2023['timestamp'].diff()
gaps = df_2023[df_2023['time_diff'] > pd.Timedelta(minutes=5)]

print(f"\nTotal records in 2023: {len(df_2023)}")
print(f"Number of gaps > 5 mins: {len(gaps)}")
if len(gaps) > 0:
    print("Top 10 largest gaps:")
    print(gaps[['timestamp', 'time_diff']].sort_values('time_diff', ascending=False).head(10))

# Check monthly counts
print("\nMonthly record counts (ideal for 31 days is 44,640):")
print(df_2023['timestamp'].dt.month.value_counts().sort_index())

# Check for resets in the cumulative variable
df_2023['P_diff'] = df_2023['P_cumulative'].diff()
resets = df_2023[df_2023['P_diff'] < -0.1]
print(f"\nNumber of significant bucket resets: {len(resets)}")
if len(resets) > 0:
    print(resets[['timestamp', 'P_cumulative', 'P_diff']].head())

# Look at June specifically
print("\nJune total missing records (out of 43,200):")
june = df_2023[df_2023['timestamp'].dt.month == 6]
print(43200 - len(june))
