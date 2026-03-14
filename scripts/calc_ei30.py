import os
import pandas as pd
import numpy as np

# Inputs
AWS_CSV = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"
OUTPUT_EVENTS = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_erosivity_events.csv"

def calculate_erosivity():
    print("Loading 1-minute AWS data (v2)...")
    df = pd.read_csv(AWS_CSV, parse_dates=['datetime_utc'])
    
    # We apply a robust custom Digital Signal Processing algorithm to Pluvio2_1.value1 (raw load cell weight)
    # to cancel severe wind pumping noise and evaporation drift.
    print("Applying 10-minute Rolling Median DSP to raw bucket weight...")
    df = df.set_index('datetime_utc')
    df['P_smooth'] = df['Pluvio2_1.value1'].rolling('10min').median()
    df['P_1min'] = df['P_smooth'].diff().fillna(0)
    
    # Suppress tiny noise variations (< 0.01 mm/min)
    df.loc[df['P_1min'] < 0.01, 'P_1min'] = 0.0
    
    # Suppress massive empty bucket drops
    resets = df['P_smooth'].diff() < -2.0
    df.loc[resets, 'P_1min'] = 0.0
    df = df.reset_index()

    
    # Phase mask: only liquid season (Apr-Oct) and positive temps
    # Snow melt inside the rain gauge causes intense false signals in winter
    print("Applying physical phase mask (April-October and T > 0C)...")
    valid_season = df['datetime_utc'].dt.month.isin([4, 5, 6, 7, 8, 9, 10])
    
    if 'HMP155.T' in df.columns:
        valid_temp = df['HMP155.T'] > 0
    else:
        valid_temp = pd.Series(True, index=df.index)
        
    df.loc[~(valid_season & valid_temp), 'P_1min'] = 0.0
    
    df = df.set_index('datetime_utc')
    res = df['P_1min'].resample('10min').sum().fillna(0)
    
    # Filter 10-min world record (approx 42 mm). Let's filter > 30 mm/10min.
    res[res > 30.0] = 0.0
    
    print("Identifying erosive events (>6h dry gaps)...")
    is_dry = res == 0
    dry_periods = is_dry.groupby((~is_dry).cumsum()).cumsum()
    
    event_ids = (dry_periods >= 36).shift(1).fillna(0).cumsum()
    events = res.groupby(event_ids)
    event_stats = []
    
    for eid, e_data in events:
        e_data = e_data[e_data > 0]
        if len(e_data) == 0:
            continue
            
        start_time = e_data.index[0]
        end_time = e_data.index[-1]
        
        P_total = e_data.sum()
        
        rolling_30 = e_data.rolling(window=3, min_periods=1).sum()
        max_P_30min = rolling_30.max()
        I30 = max_P_30min * 2 # convert mm/30m to mm/h
        
        # RUSLE2 minimum thresholds for valid erosive event
        if P_total < 12.7 and I30 < 12.7:
            continue
            
        intensity = e_data * 6 # mm / 10min to mm/h
        e_r = 0.29 * (1 - 0.72 * np.exp(-0.05 * intensity[intensity > 0]))
        E_total = (e_r * e_data[intensity > 0]).sum()
        EI30 = E_total * I30
        
        event_stats.append({
            'event_id': eid,
            'start': start_time,
            'end': end_time,
            'duration_h': (end_time - start_time).total_seconds() / 3600 + (10/60),
            'P_total_mm': round(P_total, 2),
            'I30_mmh': round(I30, 2),
            'E_total_MJ_ha': round(E_total, 2),
            'EI30_MJ_mm_ha_h': round(EI30, 2)
        })
        
    out_df = pd.DataFrame(event_stats)
    print(f"\nTotal true erosive events found (Summer): {len(out_df)}")
    if len(out_df) > 0:
        print("\nSummary of top 5 erosivity events:")
        print(out_df.sort_values('EI30_MJ_mm_ha_h', ascending=False).head())
        
        out_df.to_csv(OUTPUT_EVENTS, index=False)
        print(f"\nSaved erosive events to {OUTPUT_EVENTS}")
        
        out_df['year'] = out_df['start'].dt.year
        annual_R = out_df.groupby('year')['EI30_MJ_mm_ha_h'].sum()
        print("\nAnnual R-factor (Station AWS310):")
        print(annual_R)

if __name__ == "__main__":
    calculate_erosivity()
