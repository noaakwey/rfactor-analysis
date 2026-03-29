import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Define base dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.qm_v2 import fit_season_models, apply_season_models, get_season

# =========================
# RUSLE2 EVENT PARAMS (Batch files are 3-hourly!)
# =========================
MIN_EVENT_RAIN_MM = 12.7       
SEPARATION_HOURS = 6           
SEP_MAX_RAIN_MM = 1.27         
DT_HOURS = 3.0                 
SEP_STEPS = int(SEPARATION_HOURS / DT_HOURS)

def unit_energy_brown_foster(i_mm_h):
    i_mm_h = np.maximum(i_mm_h, 0.0)
    return 0.29 * (1.0 - 0.72 * np.exp(-0.05 * i_mm_h))

def find_events(rain_mm):
    rain_mm = np.asarray(rain_mm, dtype=float)
    wet_idx = np.where(rain_mm > 0)[0]
    if len(wet_idx) == 0:
        return []

    events = []
    start = wet_idx[0]
    last_wet = wet_idx[0]

    for idx in wet_idx[1:]:
        gap = idx - last_wet
        if gap > SEP_STEPS:
            gap_rain = rain_mm[last_wet + 1: idx].sum()
            if gap_rain < SEP_MAX_RAIN_MM:
                events.append((start, last_wet))
                start = idx
        last_wet = idx

    events.append((start, last_wet))
    return events

def calc_rusle2_r(rain_series):
    # rain_series is 30-min mm values
    df = pd.DataFrame({'rain_mm': rain_series}).fillna(0.0)
    df["i_mm_h"] = df["rain_mm"] / DT_HOURS
    df["e_MJ_ha_mm"] = unit_energy_brown_foster(df["i_mm_h"])
    df["E_interval_MJ_ha"] = df["e_MJ_ha_mm"] * df["rain_mm"]
    
    events = find_events(df["rain_mm"].values)
    
    total_r = 0.0
    for (i0, i1) in events:
        sub = df.iloc[i0:i1+1]
        P_event = sub["rain_mm"].sum()
        if P_event < MIN_EVENT_RAIN_MM:
            # Check secondary criteria: max 30-min intensity >= 25.4 mm/h (12.7 mm per 30m)
            I30 = sub["i_mm_h"].max()
            if I30 < 25.4:
                continue
        else:
            I30 = sub["i_mm_h"].max()
            
        E_event = sub["E_interval_MJ_ha"].sum()
        EI30 = E_event * I30
        total_r += EI30
        
    return total_r / 24.0 # return annual average (24 years: 2001-2024. wait, length might vary. We will divide by years dynamically)

def process_stations():
    BASE_DIR = r"D:\Cache\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib"
    CALIB_DIR = os.path.join(BASE_DIR, "output", "calib_imerg")
    files = glob.glob(os.path.join(CALIB_DIR, "*_calib.csv"))
    
    results = []
    
    TAIL_POWER = 1.05 # Let's try 1.05 and VIF = 1.10 based on previous good results
    VIF = 1.10
    
    # We will process validation period 2016-2024 (9 years)
    for f in tqdm(files[:]): 
        try:
            df = pd.read_csv(f)
            if 'yyyy' in str(df.iloc[0,0]):
                df = pd.read_csv(f, skiprows=[1])
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            if 'season' not in df.columns:
                df['season'] = df['datetime'].dt.month.apply(get_season)
            
            train_mask = (df['datetime'] >= '2001-01-01') & (df['datetime'] <= '2015-12-31')
            val_mask = (df['datetime'] >= '2016-01-01') & (df['datetime'] <= '2024-12-31')
            
            if train_mask.sum() < 100 or val_mask.sum() < 100:
                continue
                
            models = fit_season_models(df[train_mask])
            df['P_v2'] = apply_season_models(df, models, tail_power=TAIL_POWER) * VIF
            
            val_df = df[val_mask].copy()
            val_df['P_station_mm'] = val_df['P_station_mm'].fillna(0.0)
            val_df['P_sat_mm'] = val_df['P_sat_mm'].fillna(0.0)
            val_df['P_corrected_mm'] = val_df['P_corrected_mm'].fillna(0.0)
            val_df['P_v2'] = val_df['P_v2'].fillna(0.0)
            
            # Since station data is 3-hourly and IMERG is 0.5-hourly
            # We must be careful! IMERG is 0.5 hourly.
            # Station is 3-hourly. So calculate_rusle2_r on station will underestimate I30.
            # We already know this. We can just evaluate V1 vs V2 vs RAW using the proper RUSLE2 function!
            
            years = (val_df['datetime'].max() - val_df['datetime'].min()).days / 365.25
            
            r_raw = calc_rusle2_r(val_df['P_sat_mm']) * (24.0 / years) # normalized
            r_v1 = calc_rusle2_r(val_df['P_corrected_mm']) * (24.0 / years)
            r_v2 = calc_rusle2_r(val_df['P_v2']) * (24.0 / years)
            
            # Station proxy
            # calculate proxy for station, raw, v2
            def calculate_erosivity_proxy(series_mm, dt_h=3.0):
                return np.sum(series_mm * (series_mm / dt_h)) / years
                
            p_st = calculate_erosivity_proxy(val_df['P_station_mm'], 3.0)
            p_raw = calculate_erosivity_proxy(val_df['P_sat_mm'], 0.5)
            p_v2 = calculate_erosivity_proxy(val_df['P_v2'], 0.5)
            
            results.append({
                'station': os.path.basename(f),
                'R_raw': r_raw, 
                'R_v1': r_v1,
                'R_v2': r_v2,
                'Proxy_st': p_st,
                'Proxy_raw': p_raw,
                'Proxy_v2': p_v2
            })
            
        except Exception as e:
            pass

    res_df = pd.DataFrame(results)
    print("\n=== TRUE RUSLE2 R-FACTOR SUMMARY ===")
    print(f"Mean Raw R: {res_df['R_raw'].mean():.1f}")
    print(f"Mean V1 R:  {res_df['R_v1'].mean():.1f}")
    print(f"Mean V2 R:  {res_df['R_v2'].mean():.1f}")
    
    print("\nProxy Comparison (Gap analysis):")
    print(f"Median Proxy Station: {res_df['Proxy_st'].median():.1f}")
    print(f"Median Proxy Raw:     {res_df['Proxy_raw'].median():.1f}")
    print(f"Median Proxy V2:      {res_df['Proxy_v2'].median():.1f}")

if __name__ == '__main__':
    process_stations()
