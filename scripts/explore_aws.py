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
    
    # We found that 91 bytes is WRONG. Why did it skip so many records?
    # Because Vaisala loggers often record DIFFERENT TABLES in the same file!
    # A single .DAT file might contain 1-min data, 10-min data, and daily data interleaved!
    # That's why the timestamp delta is not always 60s at exactly 91 bytes away!
    #
    # How to identify the 1-minute precipitation table?
    # The header has multiple definitions or maybe we just scan for the Table ID.
    # We saw "02 fe" at the start of a record.
    # 
    # Let's just scan for ANY valid Big Endian Unix timestamp in 2022-2025.
    # And then extract the FLOAT at offset +16 from the timestamp.
    # This is a bit hacky but if Pluvio Intensity is always exactly +16 bytes from the timestamp 
    # in the 1-minute table, we can just grab it.
    
    # Wait, the 10-minute table might also have a timestamp, but the data length is different.
    # To be safe, let's grab the floats at +16, +20, +24, +28, +32.
    # One of them will be the Pluvio Intensity (which is often mostly 0.0 or a valid float like 94.13).
    
    total_valid_ts = 0
    temp_records = []

    for fp in dat_files[:10]: # test on first 10 files
        print(f"Parsing {os.path.basename(fp)}...")
        with open(fp, "rb") as f:
            data = f.read()
            
        # find all valid timestamps
        j = 0
        while j < len(data) - 40:
            val_be = struct.unpack(">I", data[j:j+4])[0]
            if 1670000000 <= val_be <= 1735689600: # Dec 2022 to Dec 2024
                # It's a timestamp!
                dt = datetime.fromtimestamp(val_be, timezone.utc)
                total_valid_ts += 1
                
                # Extract 5 floats after the timestamp, jumping over what might be IDs
                # Let's just grab words at j+4, j+8, j+12, j+16...
                floats = []
                for offset in range(4, 36, 4):
                    try:
                        fval = struct.unpack(">f", data[j+offset:j+offset+4])[0]
                        # Discard nan and extreme junk
                        if -100 <= fval <= 1000:
                            floats.append(round(fval, 3))
                        else:
                            floats.append(None)
                    except:
                        floats.append(None)
                        
                temp_records.append({
                    'timestamp': dt,
                    'f1': floats[0], 'f2': floats[1], 'f3': floats[2], 
                    'f4': floats[3], 'f5': floats[4], 'f6': floats[5], 'f7': floats[6]
                })
                # Skip at least 10 bytes to avoid overlapping inner timestamps
                j += 10
            else:
                j += 1
                
    df = pd.DataFrame(temp_records)
    print(f"Total timestamps found: {total_valid_ts}")
    if len(df) > 0:
        print("\nData Preview (Non-zero f4):")
        print(df[df['f4'] > 0].head(20))

if __name__ == "__main__":
    extract_pluvio_robust()
