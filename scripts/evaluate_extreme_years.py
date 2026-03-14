import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CALIB_DIR = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib\output\calib_imerg"
OUTPUT_DIR = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output"

def analyze_extreme_years():
    print("Анализ метрик калибровки в экстремальные годы эрозии (2001, 2007)...")
    
    csv_files = glob.glob(os.path.join(CALIB_DIR, "*_calib.csv"))
    
    annual_metrics = []
    
    for fp in csv_files:
        filename = os.path.basename(fp)
        station_name = filename.split('_calib')[0]
        
        try:
            df = pd.read_csv(fp)
            if 'date' in df.columns:
                df['year'] = df['date'].astype(str).str[:4].astype(int)
            elif 'time' in df.columns:
                df['year'] = df['time'].astype(str).str[:4].astype(int)
            elif 'datetime' in df.columns:
                df['year'] = df['datetime'].astype(str).str[:4].astype(int)
            else:
                continue
                
            # Aggregate to annual sums
            # Usually columns are: observed (meteo), imerg_raw, imerg_corr
            cols_to_check = [('precip', 'imerg', 'imerg_qm'), 
                             ('obs', 'raw', 'corr'),
                             ('p_obs', 'p_raw', 'p_corr'),
                             ('obs', 'sat', 'corr')]
                             
            obs_col = sat_col = corr_col = None
            for o, s, c in cols_to_check:
                if o in df.columns and s in df.columns:
                    obs_col, sat_col = o, s
                    corr_col = c if c in df.columns else None
                    break
                    
            if not obs_col:
                if 'P_station_mm' in df.columns and 'P_sat_mm' in df.columns:
                    obs_col, sat_col = 'P_station_mm', 'P_sat_mm'
                else:
                    possible_obs = [c for c in df.columns if 'obs' in c or 'meteo' in c or 'precip' in c]
                    possible_sat = [c for c in df.columns if 'imerg' in c or 'sat' in c or 'raw' in c]
                    if possible_obs and possible_sat:
                        obs_col = possible_obs[0]
                        sat_col = possible_sat[0]
                    else:
                        continue
                    
            annual = df.groupby('year')[[obs_col, sat_col]].sum()
            
            for index, row in annual.iterrows():
                obs = row[obs_col]
                sat = row[sat_col]
                
                if obs > 0:
                    pbias = 100 * (sat - obs) / obs
                    annual_metrics.append({
                        'station': station_name,
                        'year': index,
                        'obs': obs,
                        'sat': sat,
                        'pbias': pbias
                    })
        except Exception as e:
            pass
            
    res_df = pd.DataFrame(annual_metrics)
    if len(res_df) == 0:
        print("Не удалось извлечь данные осадков по годам из файлов calib.csv")
        return
        
    print(res_df.head())
    
    # Filter out years before IMERG/TRMM exists or years where satellite is 0 across the board
    res_df = res_df[(res_df['year'] >= 2001) & (res_df['year'] <= 2024)]
    # Filter where satellite data is missing (PBIAS == -100%)
    res_df = res_df[res_df['pbias'] > -99.0]
    
    # Calculate median PBIAS per year
    median_pbias = res_df.groupby('year')['pbias'].median()
    mean_pbias = res_df.groupby('year')['pbias'].mean()
    
    print("\nМедианный PBIAS (IMERG) по годам:")
    print(median_pbias)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(median_pbias.index, median_pbias.values, 'b-o', label='Медианный PBIAS (%)')
    
    # Highlight 2001 and 2007
    plt.axvline(x=2001, color='r', linestyle='--', alpha=0.7, label='Экстремальный год (2001)')
    plt.axvline(x=2007, color='m', linestyle='--', alpha=0.7, label='Экстремальный год (2007)')
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    plt.title('Процентное смещение (PBIAS) осадков IMERG по годам\nОценка стабильности модели в годы с аномальной эрозией')
    plt.xlabel('Год')
    plt.ylabel('PBIAS (%)')
    plt.legend()
    plt.grid(True, alpha=0.4)
    
    text_content = (
        f"PBIAS 2001: {median_pbias.get(2001, np.nan):.1f}%\n"
        f"PBIAS 2007: {median_pbias.get(2007, np.nan):.1f}%\n"
        "Средний: " + str(round(median_pbias.mean(), 1)) + "%"
    )
    plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
             
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "extreme_years_pbias.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nГрафик сохранен в: {plot_path}")

if __name__ == "__main__":
    analyze_extreme_years()
