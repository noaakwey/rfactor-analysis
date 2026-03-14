import os
import glob
import pandas as pd
import struct
from datetime import datetime, timezone

AWS_DATA_DIR = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\MAWS_Observatoriya"
OUTPUT_FILE = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio.csv"

def extract_pluvio_robust():
    dat_files = sorted(glob.glob(os.path.join(AWS_DATA_DIR, "*.DAT")))
    all_records = []
    
    # Iterate through all files and extract timestamp and Pluvio variables
    for fp in dat_files:
        print(f"Parsing {os.path.basename(fp)}...")
        with open(fp, "rb") as f:
            data = f.read()
            
        j = 0
        while j < len(data) - 90:
            val_be = struct.unpack(">I", data[j:j+4])[0]
            if 1670000000 <= val_be <= 1735689600: # Valid window for this dataset
                dt = datetime.fromtimestamp(val_be, timezone.utc)
                
                # Offset 16 is Pluvio2_v1/v2 or similar (near 94.13)
                # Offset 79/80 is Temperature (near -4.68)
                # We extract specific float offsets:
                # Based on previous dump: +16 = Pluvio_value, +79 = Temperature
                try:
                    f_pluvio = struct.unpack(">f", data[j+16:j+20])[0]
                    f_temp = struct.unpack(">f", data[j+79:j+83])[0]
                    
                    if -100 <= f_pluvio <= 5000: # Pluvio cumulative counter or mm intensity
                        all_records.append({
                            'datetime_utc': dt,
                            'pluvio_val': round(f_pluvio, 3),
                            'temp_c': round(f_temp, 2) if -60 <= f_temp <= 60 else None
                        })
                except:
                    pass
                
                # Skip to next record
                j += 40
            else:
                j += 1

    df = pd.DataFrame(all_records)
    print(f"Total timestamps parsed: {len(df)}")
    
    if len(df) > 0:
        # Sort and deduplicate
        df = df.sort_values('datetime_utc').drop_duplicates('datetime_utc').reset_index(drop=True)
        print(f"Total unique timestamps: {len(df)}")
        
        # Calculate precipitation intensity from accumulated Pluvio value 
        # Pluvio series is usually a cumulative bucket weight or exact precipitation rate.
        # "value1" is usually the cumulative sum. If so, diff() gives the 1-min rainfall.
        # Let's check max val
        
        print("Pluvio basic stats:")
        print(df['pluvio_val'].describe())
        
        # Detect if it's cumulative (strictly increasing except drops/resets)
        df['P_1min'] = df['pluvio_val'].diff()
        
        # Set negative diffs to 0 (reset/maintenance)
        df.loc[df['P_1min'] < 0, 'P_1min'] = 0
        
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved extraction to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_pluvio_robust()
