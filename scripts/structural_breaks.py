import os
import pandas as pd
import numpy as np
import pyhomogeneity as hg
import matplotlib.pyplot as plt

# The user's R-factor annual mean time series is likely in the `output/` directory 
# or can be quickly generated from the TIF maps.
# Since we already have the TIF maps, let's just calculate the domain-averaged R-factor first.

IMERG_DIR = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\annual"
OUTPUT_DIR = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output"

def analyze_structural_breaks():
    import rasterio
    import glob
    
    tif_files = sorted(glob.glob(os.path.join(IMERG_DIR, "*.tif")))
    
    records = []
    print("Calculating domain-averaged annual R-factor...")
    for tif in tif_files:
        year_str = os.path.basename(tif).split('_')[-1].replace('.tif', '')
        if not year_str.isdigit():
            continue
        year = int(year_str)
        
        with rasterio.open(tif) as src:
            data = src.read(1)
            # Filter NoData (usually < 0)
            valid_data = data[data >= 0]
            if len(valid_data) > 0:
                mean_r = np.mean(valid_data)
                records.append({'year': year, 'r_factor': mean_r})
                
    df = pd.DataFrame(records).sort_values('year').set_index('year')
    print("\nExtract time series (2001-2024):")
    print(df.head())
    
    # Save the time series
    ts_path = os.path.join(OUTPUT_DIR, "domain_annual_rfactor.csv")
    df.to_csv(ts_path)
    
    print("\nRunning Pettitt's Test for structural breaks...")
    # Pettitt test is a non-parametric test to detect a single change-point in continuous data
    try:
        res = hg.pettitt_test(df['r_factor'])
        print(f"Pettitt Test Results: {res}")
        
        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['r_factor'], marker='o', label='Медианный R-фактор домена')
        
        cp = res.cp
        plt.axvline(x=df.index[cp], color='r', linestyle='--', label=f'Точка излома (Pettitt): {df.index[cp]}')
        
        # Calculate means before and after
        mean_before = df['r_factor'].iloc[:cp].mean()
        mean_after = df['r_factor'].iloc[cp:].mean()
        
        plt.hlines(y=mean_before, xmin=df.index[0], xmax=df.index[cp]-1, color='g', linestyle='-', label=f'Среднее (до {df.index[cp]}): {mean_before:.1f}')
        plt.hlines(y=mean_after, xmin=df.index[cp], xmax=df.index[-1], color='m', linestyle='-', label=f'Среднее (с {df.index[cp]}): {mean_after:.1f}')
        
        if res.p <= 0.05:
            plt.title('Статистически значимый структурный сдвиг (Тест Петтита)')
        else:
            plt.title(f'Статистически значимых сдвигов не выявлено (p={res.p:.3f})')
            
        plt.xlabel('Год')
        plt.ylabel('R-фактор (МДж·мм / (га·ч·год))')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(OUTPUT_DIR, "structural_break_pettitt.png")
        plt.savefig(plot_path, dpi=300)
        print(f"Saved plot to {plot_path}")
        
    except ImportError:
        print("pyhomogeneity not installed. Cannot run Pettitt test directly.")
        print("Falling back to basic CUSUM implementation...")

if __name__ == "__main__":
    analyze_structural_breaks()
