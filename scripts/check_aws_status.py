import struct
import os

AWS_DATA_DIR = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\MAWS_Observatoriya"
files = [os.path.join(AWS_DATA_DIR, "L1230821.DAT"), os.path.join(AWS_DATA_DIR, "L1230729.DAT")]

for fp in files:
    print(f"\nScanning file: {os.path.basename(fp)}")
    with open(fp, "rb") as f:
        raw = f.read()
        
    n_fields = struct.unpack_from(">I", raw, 0)[0]
    pos = 4
    for _ in range(n_fields):
        slen = struct.unpack_from(">I", raw, pos)[0]; pos += 4
        pos += slen
        alen = struct.unpack_from(">I", raw, pos)[0]; pos += 4
        pos += alen
        pos += 1
        
    status_counts = {}
    for offset in range(pos, len(raw) - 4):
        try:
            ts = struct.unpack_from(">I", raw, offset)[0]
            n = raw[offset + 12]
            if 0 < n <= 40 and offset + 13 + n*7 <= len(raw):
                for i in range(n):
                    fp_offset = offset + 13 + i * 7
                    status = raw[fp_offset + 1]
                    idx = raw[fp_offset + 2]
                    # We just log statuses
                    if idx not in status_counts:
                        status_counts[idx] = {}
                    if status not in status_counts[idx]:
                        status_counts[idx][status] = 0
                    status_counts[idx][status] += 1
        except Exception as e:
            pass
            
    for idx, counts in status_counts.items():
        if len(counts) > 1 or 1 not in counts:
            print(f"Col {idx} has statuses: {counts}")
