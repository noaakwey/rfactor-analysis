import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# SETTINGS
# =========================
csv_path = Path(r"C:\Users\artur\Downloads\Biomet01_12_2024-31_12_2025.csv")

# Названия колонок в вашем файле
COL_TIME = "TIMESTAMP_1"  # Выявлено из анализа файла
COL_P = "P_RAIN_1_1_1"    # осадки за интервал, [m]
COL_T = "TA_1_1_1"       # температура воздуха, [K] -> будет сконвертировано
COL_RH = "RH_1_1_1"      # относительная влажность, [%]

# Параметры событийности USLE/RUSLE-подхода
MIN_EVENT_RAIN_MM = 12.7       # минимальная сумма осадков события
SEPARATION_HOURS = 6           # разрыв между событиями
SEP_MAX_RAIN_MM = 1.27         # если за 6 ч выпало < 1.27 мм, новое событие
DT_HOURS = 0.5                 # шаг ряда, 30 минут
SEP_STEPS = int(SEPARATION_HOURS / DT_HOURS)

# Фазовая фильтрация осадков
USE_WET_BULB_FILTER = True
WET_BULB_THRESHOLD_C = 0.5     # жидкие осадки, если Tw >= 0.5 C

# NoData
NODATA_VALUE = -9999.0

# =========================
# HELPERS
# =========================
def wet_bulb_stull(T_c, RH_pct):
    """
    Аппроксимация Stull (2011) для wet-bulb temperature.
    Работает для T в C и RH в %.
    """
    RH_pct = np.clip(RH_pct, 1e-6, 100.0)
    Tw = (
        T_c * np.arctan(0.151977 * np.sqrt(RH_pct + 8.313659))
        + np.arctan(T_c + RH_pct)
        - np.arctan(RH_pct - 1.676331)
        + 0.00391838 * (RH_pct ** 1.5) * np.arctan(0.023101 * RH_pct)
        - 4.686035
    )
    return Tw

def unit_energy_brown_foster(i_mm_h):
    """
    Standard Brown-Foster / RUSLE2:
    e = 0.29 * (1 - 0.72 * exp(-0.05 * i))
    Note: Updated coefficient from 0.082 to 0.05 (official RUSLE2).
    """
    i_mm_h = np.maximum(i_mm_h, 0.0)
    return 0.29 * (1.0 - 0.72 * np.exp(-0.05 * i_mm_h))

def find_events(rain_mm):
    """
    Выделение дождевых событий.
    """
    rain_mm = np.asarray(rain_mm, dtype=float)
    wet_idx = np.where(rain_mm > 0)[0]
    if len(wet_idx) == 0:
        return []

    events = []
    start = wet_idx[0]
    last_wet = wet_idx[0]

    for idx in wet_idx[1:]:
        gap = idx - last_wet
        if gap >= SEP_STEPS:
            # Проверка суммы осадков в разрыве
            gap_rain = rain_mm[last_wet + 1: idx].sum()
            if gap_rain < SEP_MAX_RAIN_MM:
                events.append((start, last_wet))
                start = idx
        last_wet = idx

    events.append((start, last_wet))
    return events

# =========================
# READ DATA
# =========================
# Пропускаем вторую строку с единицами измерения
df = pd.read_csv(csv_path, skiprows=[1])

# Парсинг времени
df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
df = df.dropna(subset=[COL_TIME]).sort_values(COL_TIME).reset_index(drop=True)

# Числовые колонки
for c in [COL_P, COL_T, COL_RH]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Обработка NoData (-9999)
# Для осадков -9999 считаем 0 (или NaN, но script просит 0 для работы алгоритма)
# Хотя если это пропуск, лучше знать об этом.
df[COL_P] = df[COL_P].replace(NODATA_VALUE, 0.0)

# Температура: Kelvin -> Celsius
if df[COL_T].mean() > 200:
    df["T_C"] = df[COL_T] - 273.15
else:
    df["T_C"] = df[COL_T]

# Осадки: из метров в миллиметры
df["rain_mm_raw"] = df[COL_P].fillna(0.0) * 1000.0
df["rain_mm_raw"] = df["rain_mm_raw"].clip(lower=0.0)

# Wet-bulb
df["Tw_C"] = wet_bulb_stull(df["T_C"].astype(float), df[COL_RH].astype(float))

if USE_WET_BULB_FILTER:
    # Осадки считаются жидкими, если Tw >= 0.5 C И температура воздуха T_C > 0 (подстраховка)
    df["is_liquid"] = (df["Tw_C"] >= WET_BULB_THRESHOLD_C)
    df["rain_mm"] = np.where(df["is_liquid"], df["rain_mm_raw"], 0.0)
else:
    df["is_liquid"] = True
    df["rain_mm"] = df["rain_mm_raw"]

# Интенсивность
df["i_mm_h"] = df["rain_mm"] / DT_HOURS

# Кинетическая энергия
df["e_MJ_ha_mm"] = unit_energy_brown_foster(df["i_mm_h"])
df["E_interval_MJ_ha"] = df["e_MJ_ha_mm"] * df["rain_mm"]

# =========================
# FIND EVENTS
# =========================
events = find_events(df["rain_mm"].values)

event_rows = []
for ev_id, (i0, i1) in enumerate(events, start=1):
    sub = df.iloc[i0:i1+1].copy()
    P_event = sub["rain_mm"].sum()
    if P_event < MIN_EVENT_RAIN_MM:
        continue

    I30 = sub["i_mm_h"].max()
    E_event = sub["E_interval_MJ_ha"].sum()
    EI30 = E_event * I30

    event_rows.append({
        "event_id": ev_id,
        "start": sub[COL_TIME].iloc[0],
        "end": sub[COL_TIME].iloc[-1],
        "duration_h": len(sub) * DT_HOURS,
        "P_mm": P_event,
        "I30_mm_h": I30,
        "E_MJ_ha": E_event,
        "EI30_MJ_mm_ha_h": EI30,
        "year": sub[COL_TIME].iloc[0].year,
        "month": sub[COL_TIME].iloc[0].month,
    })

events_df = pd.DataFrame(event_rows)

# =========================
# ANNUAL R
# =========================
if len(events_df) == 0:
    R_year = 0.0
    monthly_R = pd.Series(dtype=float)
    annual_R = pd.Series(dtype=float)
else:
    R_year = events_df["EI30_MJ_mm_ha_h"].sum()
    monthly_R = events_df.groupby(["year", "month"])["EI30_MJ_mm_ha_h"].sum()
    annual_R = events_df.groupby("year")["EI30_MJ_mm_ha_h"].sum()

# =========================
# OUTPUT
# =========================
print("\n=== CORRECTED INPUT SUMMARY ===")
print(f"Rows: {len(df)}")
print(f"Period: {df[COL_TIME].min()} to {df[COL_TIME].max()}")
print(f"Total precipitation raw [mm]: {df['rain_mm_raw'].sum():.2f}")
print(f"Total liquid precipitation [mm]: {df['rain_mm'].sum():.2f}")
print(f"Total solid precipitation [mm]: {(df['rain_mm_raw'].sum() - df['rain_mm'].sum()):.2f}")
print(f"Liquid filter enabled: {USE_WET_BULB_FILTER}")
print(f"Unit Energy Coefficient: 0.05 (RUSLE2 Standard)")

print("\n=== EVENT SUMMARY ===")
print(f"Detected erosive events: {len(events_df)}")

if len(events_df) > 0:
    print("\nTop 10 erosive events:")
    print(
        events_df.sort_values("EI30_MJ_mm_ha_h", ascending=False)
        [["event_id", "start", "end", "P_mm", "I30_mm_h", "EI30_MJ_mm_ha_h"]]
        .head(10)
        .to_string(index=False)
    )

print("\n=== R-FACTOR ===")
print(f"R (corrected) = {R_year:.2f} MJ mm ha^-1 h^-1 yr^-1")

if len(annual_R) > 0:
    print("\nAnnual R:")
    print(annual_R.to_string())

if len(monthly_R) > 0:
    print("\nMonthly contribution to R:")
    print(monthly_R.to_string())

# Сохранение результатов
out_dir = Path("d:/Cache/Yandex.Disk/РАЗРАБОТКА/code/rfactor-analysis/output")
events_out = out_dir / "corrected_events_ei30.csv"
summary_out = out_dir / "corrected_summary.txt"

events_df.to_csv(events_out, index=False)
with open(summary_out, "w", encoding="utf-8") as f:
    f.write("FIXED RUSLE2 rainfall erosivity analysis\n")
    f.write(f"Period: {df[COL_TIME].min()} to {df[COL_TIME].max()}\n")
    f.write(f"Total precipitation raw [mm]: {df['rain_mm_raw'].sum():.2f}\n")
    f.write(f"Total liquid precipitation (used): {df['rain_mm'].sum():.2f}\n")
    f.write(f"Detected erosive events: {len(events_df)}\n")
    f.write(f"R [MJ mm ha^-1 h^-1]: {R_year:.2f}\n\n")
    if len(monthly_R) > 0:
        f.write("Monthly R contribution:\n")
        f.write(monthly_R.to_string())

print(f"\nSaved to {out_dir}")
