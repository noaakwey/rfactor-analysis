import pandas as pd

aws_file = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
df = pd.read_csv(aws_file, parse_dates=['datetime_utc'])
df.rename(columns={'datetime_utc': 'timestamp', 'Pluvio2_1.value1': 'P_cumulative'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
df_2023 = df[df['timestamp'].dt.year == 2023].copy()
df_2023 = df_2023.sort_values('timestamp')

df_2023['P_1min'] = df_2023['P_cumulative'].diff()
df_2023['clean_P'] = df_2023['P_1min']
df_2023.loc[(df_2023['clean_P'] < 0) | (df_2023['clean_P'] > 5.0), 'clean_P'] = 0.0

def check_date(date_str):
    print(f"\n--- Checking {date_str} ---")
    day = df_2023[df_2023['timestamp'].dt.date == pd.to_datetime(date_str).date()]
    print(f"Total clean value1: {day['clean_P'].sum():.2f} mm")
    
    day['P_1min_v2'] = day['Pluvio2_1.value2'].diff()
    v2_pos = day.loc[day['P_1min_v2'] > 0, 'P_1min_v2'].sum()
    print(f"Total value2: {v2_pos:.2f} mm")
    
check_date('2023-07-29')
check_date('2023-05-05')
check_date('2023-05-06')
check_date('2023-10-02')

# Also, check the IMERG pixel value for Kazan (55.84N, 48.81E) from the comparison CSV
imerg_csv = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\imerg_aws310_comparison.csv"
try:
    imerg_df = pd.read_csv(imerg_csv)
    print("\n--- IMERG R-factor extracted at Kazan ---")
    print(imerg_df)
except:
    pass
