import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from lib.qm_v2 import fit_season_models, apply_season_models, get_season

# =========================
# SETTINGS
# =========================
BASE_DIR = r"D:\Cache\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib"
CALIB_DIR = os.path.join(BASE_DIR, "output", "calib_imerg")
OUT_FILE = r"d:\Cache\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\batch_v2_verification.csv"

# V2 Params
TAIL_POWER = 1.00 # Set to 1.0 for VIF-only focus
VIF        = 1.18 # The multiplier that should close the -29% gap

TRAIN_START = '2001-01-01'
TRAIN_END   = '2015-12-31'
VAL_START   = '2016-01-01'
VAL_END     = '2021-12-31'

def calculate_erosivity_proxy(series_mm, dt_h=3.0):
    return np.sum(series_mm * (series_mm / dt_h))

def batch_verify():
    files = glob.glob(os.path.join(CALIB_DIR, "*_calib.csv"))
    print(f"Starting batch verification on {len(files)} stations...")
    
    results = []
    
    for f in tqdm(files):
        try:
            df = pd.read_csv(f)
            if 'yyyy' in str(df.iloc[0,0]):
                df = pd.read_csv(f, skiprows=[1])
            
            df['datetime'] = pd.to_datetime(df['datetime'])
            if 'season' not in df.columns:
                df['season'] = df['datetime'].dt.month.apply(get_season)
            
            # Split
            train_mask = (df['datetime'] >= TRAIN_START) & (df['datetime'] <= TRAIN_END)
            val_mask = (df['datetime'] >= VAL_START) & (df['datetime'] <= VAL_END)
            
            if train_mask.sum() < 100 or val_mask.sum() < 100:
                continue
                
            models = fit_season_models(df[train_mask])
            df['P_v2'] = apply_season_models(df, models, tail_power=TAIL_POWER) * VIF
            
            val_df = df[val_mask].copy()
            
            r_st   = calculate_erosivity_proxy(val_df['P_station_mm'])
            r_raw  = calculate_erosivity_proxy(val_df['P_sat_mm'])
            r_v1   = calculate_erosivity_proxy(val_df['P_corrected_mm'])
            r_v2   = calculate_erosivity_proxy(val_df['P_v2'])
            
            results.append({
                'station': os.path.basename(f),
                'R_st': r_st, 'R_raw': r_raw, 'R_v1': r_v1, 'R_v2': r_v2,
                'PBIAS_R_raw': 100 * (r_raw - r_st) / r_st if r_st > 0 else 0,
                'PBIAS_R_v1':  100 * (r_v1 - r_st) / r_st if r_st > 0 else 0,
                'PBIAS_R_v2':  100 * (r_v2 - r_st) / r_st if r_st > 0 else 0,
            })
        except: continue
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_FILE, index=False)
    print("\n=== GLOBAL RESULTS (Median PBIAS of R-factor) ===")
    print(f"  Raw IMERG R Bias:    {res_df['PBIAS_R_raw'].median():.1f}%")
    print(f"  Standard QM (V1) R Bias: {res_df['PBIAS_R_v1'].median():.1f}%")
    print(f"  Intensity-Aware (V2) R Bias: {res_df['PBIAS_R_v2'].median():.1f}%")

if __name__ == "__main__":
    batch_verify()
