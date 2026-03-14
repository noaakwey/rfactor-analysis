import os
import glob
import struct
import datetime
import pandas as pd

AWS_DATA_DIR = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\MAWS_Observatoriya"
OUTPUT_FILE = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis\output\aws310_pluvio_v2.csv"

def parse_vaisala_dat(filepath):
    print(f"Parsing {os.path.basename(filepath)}...")
    with open(filepath, "rb") as f:
        raw = f.read()

    # Read header: [4B: n_fields] then fields [4B len][name][4B len][agg][1B dtype]
    n_fields = struct.unpack_from(">I", raw, 0)[0]
    columns, pos = [], 4
    for _ in range(n_fields):
        slen = struct.unpack_from(">I", raw, pos)[0]; pos += 4
        sensor = raw[pos:pos+slen].decode(); pos += slen
        alen = struct.unpack_from(">I", raw, pos)[0]; pos += 4
        agg = raw[pos:pos+alen].decode(); pos += alen
        pos += 1  # dtype
        columns.append(f"{sensor}.{agg}")

    # Scan body by timestamp
    fname = filepath.split("\\")[-1].split("/")[-1]
    # Name format usually something like L1230529.DAT
    # The actual date YYMMDD starts at index 2 (ignoring typical prefixes like 'L1' or 'L2')
    date_str = fname[2:8]
    if len(date_str) == 6 and date_str.isdigit():
        year = 2000 + int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        if 1 <= month <= 12 and 1 <= day <= 31:
            ts_start = int(datetime.datetime(year, month, day, tzinfo=datetime.timezone.utc).timestamp())
            ts_start -= 43200 
            ts_end = ts_start + 86400 * 2
        else:
            ts_start = 0
            ts_end = 2000000000
    else:
        # Fallback if filename doesn't match
        ts_start = 0
        ts_end = 2000000000

    rows, seen = [], set()
    for offset in range(pos, len(raw) - 4):
        # We look for a valid timestamp
        try:
            ts = struct.unpack_from(">I", raw, offset)[0]
        except:
            continue
            
        if ts not in seen and ts_start <= ts <= ts_end:
            try:
                n = raw[offset + 12]
                if 0 < n <= 40 and offset + 13 + n*7 <= len(raw):
                    # Potential record match
                    dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
                    row = {"datetime_utc": dt}
                    valid_row = True
                    for i in range(n):
                        fp = offset + 13 + i * 7
                        status = raw[fp + 1]
                        idx = raw[fp + 2]
                        val = struct.unpack_from(">f", raw, fp + 3)[0]
                        if idx < len(columns):
                            row[columns[idx]] = round(val, 4) if status == 1 else None
                        else:
                            valid_row = False
                            break
                    
                    if valid_row:
                        rows.append(row)
                        seen.add(ts)
            except Exception as e:
                pass

    if len(rows) == 0:
        return pd.DataFrame()
        
    df = pd.DataFrame(rows, columns=["datetime_utc"] + columns)
    return df.sort_values("datetime_utc").reset_index(drop=True)

def process_all():
    dat_files = sorted(glob.glob(os.path.join(AWS_DATA_DIR, "*.DAT")))
    all_dfs = []
    
    for fp in dat_files:
        df = parse_vaisala_dat(fp)
        if not df.empty:
            all_dfs.append(df)
            
    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        full_df = full_df.sort_values('datetime_utc').drop_duplicates('datetime_utc').reset_index(drop=True)
        
        print(f"Total rows parsed: {len(full_df)}")
        print("Columns found:")
        print(full_df.columns.tolist())
        
        # Identify the precipitation column. Vaisala Pluvio is usually something like 'Pluvio_precip' or 'Rain_accum'
        possible_precip = [c for c in full_df.columns if 'precip' in c.lower() or 'pluvio' in c.lower() or 'rain' in c.lower()]
        possible_temp = [c for c in full_df.columns if 'temp' in c.lower() or 'ta' in c.lower()]
        
        print(f"Possible precip columns: {possible_precip}")
        print(f"Possible temp columns: {possible_temp}")
        
        if len(possible_precip) > 0:
            pc = possible_precip[0]
            # Print basic stats to see if cumulative
            print(full_df[pc].describe())
            
            # Save the raw dataframe to inspect the header columns
            full_df.to_csv(OUTPUT_FILE, index=False)
            print(f"Saved to {OUTPUT_FILE}")
            
if __name__ == "__main__":
    process_all()
