import os
import pandas as pd
import numpy as np
from pathlib import Path
from lib.qm_v2 import fit_season_models, apply_season_models

# =========================
# SETTINGS
# =========================
BIOMET_CSV = r"C:\Users\artur\Downloads\Biomet01_12_2024-31_12_2025.csv"
IMERG_CSV  = r"D:\Cache\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib\output\calib_imerg\Казань_27595_calib.csv"

# Intensity-Aware parameters (V2)
TAIL_POWER = 1.15  # Recommended for R-factor synchronization
VIF        = 1.05  # Spatial expansion factor

def unified_pipeline():
    print("=== UNIFIED R-FACTOR PIPELINE (V2) ===")
    
    # 1. Load Biomet (Target for verification)
    bio = pd.read_csv(BIOMET_CSV, skiprows=[1])
    bio['dt'] = pd.to_datetime(bio['TIMESTAMP_1'])
    bio['P_mm'] = bio['P_RAIN_1_1_1'].replace(-9999, 0) * 1000
    bio['T_c'] = bio['TA_1_1_1'].replace(-9999, np.nan) - 273.15
    bio = bio.set_index('dt')
    bio_30m = bio['P_mm'].resample('30min').sum()
    
    # 2. Load Calibration Data (training)
    cal = pd.read_csv(IMERG_CSV) # Synoptic usually doesn't have units row, but check
    if 'yyyy' in str(cal.iloc[0,0]): # safety check
        cal = pd.read_csv(IMERG_CSV, skiprows=[1])
    cal['datetime'] = pd.to_datetime(cal['datetime'])
    
    # 3. Fit Models (Seasonal)
    print("Fitting seasonal QM models...")
    models = fit_season_models(cal)
    
    # 4. Apply Correction to IMERG 2024 (from the cal table if available)
    # Note: IMERG_CSV should contain 2024 data as per its 1966-2025 range
    imerg_2024 = cal[cal['datetime'].dt.year == 2024].copy()
    if imerg_2024.empty:
        print("Error: 2024 data not found in calibration table.")
        return
        
    print(f"Applying Tail-Inflated QM (TailPower={TAIL_POWER})...")
    imerg_2024['P_corr_v2'] = apply_season_models(imerg_2024, models, tail_power=TAIL_POWER)
    
    # 5. Apply VIF (Variance Inflation)
    imerg_2024['P_corr_v2'] *= VIF
    
    # 6. Compare Statistics
    raw_r = imerg_2024['P_sat_mm'].sum() # Simple volume for comparison
    corr_r = imerg_2024['P_corr_v2'].sum()
    st_r = imerg_2024['P_station_mm'].sum()
    
    print("\n[Annual Volume Comparison 2024]")
    print(f"  Raw IMERG:     {raw_r:.1f} mm")
    print(f"  Calibrated V2: {corr_r:.1f} mm")
    print(f"  Station (Syn): {st_r:.1f} mm")
    
    # 7. PDF of Peak Intensities (Placeholder logic)
    max_raw = imerg_2024['P_sat_mm'].max()
    max_corr = imerg_2024['P_corr_v2'].max()
    max_st = imerg_2024['P_station_mm'].max()
    
    print("\n[Peak Intensity Comparison (3h)]")
    print(f"  Max Raw:      {max_raw:.1f} mm/3h")
    print(f"  Max Corr V2:  {max_corr:.1f} mm/3h")
    print(f"  Max Station:  {max_st:.1f} mm/3h")
    
    print("\nPipeline execution complete. R-factor calculation would follow on 30-min data.")

if __name__ == "__main__":
    unified_pipeline()
