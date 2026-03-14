"""
Анализ данных осадков датчика-опрокидывателя TR-525M (Biomet)
Координаты станции: 55.2694°N, 49.2802°E

1) Фильтрация зимних артефактов по температуре воздуха
2) Сравнение с IMERG для того же пикселя
3) Расчёт EI₃₀ (R-фактора) за конвективный сезон 2025
"""

import pandas as pd
import numpy as np
import os, glob

# ============================================================
# 1. Загрузка и очистка данных Biomet
# ============================================================
biomet_file = r"C:\Users\artur\Downloads\Biomet 07_08-15_11_2025.csv"

print("=" * 70)
print("1. ЗАГРУЗКА ДАННЫХ TR-525M")
print("=" * 70)

df = pd.read_csv(biomet_file, skiprows=[1])  # skip units row
df['TIMESTAMP_1'] = pd.to_datetime(df['TIMESTAMP_1'], format='%Y-%m-%d %H%M')
df = df.set_index('TIMESTAMP_1')

# Осадки в мм (исходно в метрах)
df['rain_mm'] = df['P_RAIN_1_1_1'].replace(-9999, 0) * 1000

# Температура воздуха - в Кельвинах, переводим в Цельсии
df['ta_c'] = df['TA_1_1_1'].replace(-9999, np.nan) - 273.15

print(f"Период: {df.index.min()} — {df.index.max()}")
print(f"Всего записей: {len(df)}")
print(f"Суммарные осадки (без фильтрации): {df['rain_mm'].sum():.1f} мм")

# ============================================================
# 2. Фильтрация по температуре (только жидкие осадки, T > 2°C)
# ============================================================
print("\n" + "=" * 70)
print("2. ФИЛЬТРАЦИЯ ПО ТЕМПЕРАТУРЕ ВОЗДУХА")
print("=" * 70)

# Используем порог 2°C — при более низких T осадки могут быть
# смешанными (дождь со снегом) или конденсатом
T_THRESHOLD = 2.0
df['rain_liquid'] = df['rain_mm'].copy()
cold_mask = df['ta_c'] < T_THRESHOLD
df.loc[cold_mask, 'rain_liquid'] = 0.0

rejected_mm = df.loc[cold_mask, 'rain_mm'].sum()
print(f"Порог температуры: T > {T_THRESHOLD}°C")
print(f"Отфильтровано (холодные осадки/артефакты): {rejected_mm:.1f} мм")
print(f"Итого жидкие осадки: {df['rain_liquid'].sum():.1f} мм")

# Помесячная разбивка
print("\nПомесячная разбивка (только жидкие):")
monthly = df.resample('ME')['rain_liquid'].sum()
for idx, val in monthly.items():
    if val > 0:
        print(f"  {idx.strftime('%Y-%m')}: {val:.1f} мм")

# ============================================================
# 3. Конвективный сезон 2025: май — сентябрь
# ============================================================
print("\n" + "=" * 70)
print("3. КОНВЕКТИВНЫЙ СЕЗОН 2025 (май — сентябрь)")
print("=" * 70)

conv_mask = (df.index >= '2025-05-01') & (df.index < '2025-10-01')
conv = df.loc[conv_mask].copy()
conv_total = conv['rain_liquid'].sum()
print(f"Сумма жидких осадков (май-сен 2025): {conv_total:.1f} мм")
print(f"Макс. за 30 мин: {conv['rain_liquid'].max():.2f} мм")

conv_monthly = conv.resample('ME')['rain_liquid'].sum()
for idx, val in conv_monthly.items():
    print(f"  {idx.strftime('%Y-%m')}: {val:.1f} мм")

# ============================================================
# 4. Сравнение с IMERG
# ============================================================
print("\n" + "=" * 70)
print("4. СРАВНЕНИЕ С IMERG (пиксель 55.27°N, 49.28°E)")
print("=" * 70)

# Загрузка данных IMERG — ищем ближайший пиксель
# Координаты станции
lat_station = 55.2694
lon_station = 49.2802

# Ищем файл R-фактора IMERG
imerg_dir = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis"
rfactor_file = os.path.join(imerg_dir, "output", "domain_annual_rfactor.csv")

if os.path.exists(rfactor_file):
    rf_df = pd.read_csv(rfactor_file)
    print(f"Доменный R-фактор IMERG (по годам):")
    for _, row in rf_df.iterrows():
        print(f"  {int(row.iloc[0])}: R = {row.iloc[1]:.1f}")
else:
    print(f"Файл {rfactor_file} не найден")

# Попытка загрузить попиксельные данные IMERG для конкретного пикселя
# Ищем NetCDF файлы с суточными осадками IMERG
import subprocess
try:
    import xarray as xr
    # Ищем файлы IMERG
    imerg_nc_pattern = os.path.join(imerg_dir, "data", "imerg", "*.nc*")
    imerg_files = sorted(glob.glob(imerg_nc_pattern))
    
    if not imerg_files:
        # Пробуем другие паттерны
        for pattern in ["data/**/*.nc*", "data/**/*.hdf*", "input/**/*.nc*"]:
            imerg_files = sorted(glob.glob(os.path.join(imerg_dir, pattern), recursive=True))
            if imerg_files:
                break
    
    if imerg_files:
        print(f"\nНайдено {len(imerg_files)} IMERG файлов")
        # Пытаемся извлечь данные для пикселя станции
        sample = xr.open_dataset(imerg_files[0])
        print(f"Переменные: {list(sample.data_vars)}")
        print(f"Координаты: {list(sample.coords)}")
        sample.close()
    else:
        print("\nIMERG NetCDF файлы не найдены в стандартных директориях.")
        print("Сравнение будет проведено по доменным средним значениям.")
        
except ImportError:
    print("xarray не установлен, пропускаем прямое сравнение с IMERG.")

# ============================================================
# 5. Расчёт EI₃₀ (R-фактор) по методологии RUSLE2
# ============================================================
print("\n" + "=" * 70)
print("5. RASCHET R-FAKTORA (EI30) ZA KONVEKTIVNYJ SEZON 2025")
print("=" * 70)

def calculate_ei30(rain_series, dt_minutes=30):
    """
    Расчёт EI₃₀ по данным осадков с фиксированным временным шагом.
    
    Методология RUSLE2:
    - Эрозивное событие: сумма ≥ 12.7 мм ИЛИ пиковая I₃₀ ≥ 25.4 мм/ч
    - Разделение событий: ≥ 6 часов без осадков (≥ 12 записей по 30-мин)
    - Кинетическая энергия: e = 0.29 * (1 - 0.72 * exp(-0.05 * i))
      где i — интенсивность в мм/ч
    - EI₃₀ = E_total * I₃₀_max для каждого события
    
    Parameters
    ----------
    rain_series : pd.Series with DatetimeIndex, rainfall in mm per timestep
    dt_minutes : int, timestep in minutes
    
    Returns
    -------
    events : list of dicts with event details
    r_factor : float, sum of EI₃₀
    """
    rain = rain_series.copy()
    rain = rain.fillna(0)
    
    # Интенсивность в мм/ч
    intensity = rain * (60.0 / dt_minutes)
    
    # Идентификация событий: разделение по 6-часовым паузам
    # Для 30-мин данных: пауза = 12 нулевых записей подряд
    gap_threshold = int(6 * 60 / dt_minutes)  # 12 для 30-мин
    
    is_rain = rain > 0
    events_list = []
    
    if not is_rain.any():
        print("  Осадков не зарегистрировано в данном периоде.")
        return [], 0.0
    
    # Группировка в события
    # Помечаем начало нового события, если перед ним было >= gap_threshold нулей
    event_id = 0
    event_ids = pd.Series(0, index=rain.index)
    consecutive_dry = 0
    in_event = False
    
    for i, (idx, r) in enumerate(rain.items()):
        if r > 0:
            if not in_event:
                event_id += 1
                in_event = True
            consecutive_dry = 0
            event_ids.iloc[i] = event_id
        else:
            consecutive_dry += 1
            if consecutive_dry >= gap_threshold:
                in_event = False
            if in_event:
                event_ids.iloc[i] = event_id
    
    # Обработка каждого события
    for eid in range(1, event_id + 1):
        mask = event_ids == eid
        evt_rain = rain[mask]
        evt_intensity = intensity[mask]
        
        total_rain = evt_rain.sum()
        max_i30 = evt_intensity.max()  # Уже за 30-мин окно
        duration_h = len(evt_rain) * dt_minutes / 60.0
        
        # Проверка эрозивности: total ≥ 12.7 мм ИЛИ I₃₀ ≥ 25.4 мм/ч
        is_erosive = (total_rain >= 12.7) or (max_i30 >= 25.4)
        
        # Кинетическая энергия по Brown & Foster (1987)
        # e = 0.29 * (1 - 0.72 * exp(-0.05 * i)) МДж/(га·мм)
        # E_increment = e_i * rain_i (мм)
        e_unit = 0.29 * (1 - 0.72 * np.exp(-0.05 * evt_intensity.values))
        e_total = np.sum(e_unit * evt_rain.values)  # МДж/га
        
        # EI₃₀ = E_total * I₃₀_max
        ei30 = e_total * max_i30
        
        events_list.append({
            'start': evt_rain.index[0],
            'end': evt_rain.index[-1],
            'duration_h': duration_h,
            'total_mm': total_rain,
            'max_i30': max_i30,
            'e_total': e_total,
            'ei30': ei30,
            'is_erosive': is_erosive
        })
    
    return events_list, sum(e['ei30'] for e in events_list if e['is_erosive'])


# Расчёт для конвективного сезона 2025
events, r_factor = calculate_ei30(conv['rain_liquid'])

print(f"\nВсего идентифицировано событий: {len(events)}")
erosive_events = [e for e in events if e['is_erosive']]
print(f"Из них эрозивных (≥12.7 мм или I₃₀ ≥ 25.4 мм/ч): {len(erosive_events)}")
print(f"\n{'='*70}")
print(f"R-ФАКТОР (сумма EI₃₀ эрозивных событий): {r_factor:.1f} МДж·мм·га⁻¹·ч⁻¹")
print(f"{'='*70}")

if events:
    print("\nТоп-10 событий по EI₃₀:")
    events_sorted = sorted(events, key=lambda x: x['ei30'], reverse=True)
    print(f"{'Дата':20s} {'Длит.(ч)':>8s} {'Сумма(мм)':>10s} {'I₃₀(мм/ч)':>10s} {'EI₃₀':>10s} {'Эрозивн.':>10s}")
    print("-" * 70)
    for e in events_sorted[:10]:
        print(f"{str(e['start']):20s} {e['duration_h']:8.1f} {e['total_mm']:10.1f} {e['max_i30']:10.1f} {e['ei30']:10.1f} {'ДА' if e['is_erosive'] else 'нет':>10s}")

# Полный список всех событий >= 5 мм
print("\nВсе события >= 5 мм:")
print(f"{'Дата':20s} {'Длит.(ч)':>8s} {'Сумма(мм)':>10s} {'I₃₀(мм/ч)':>10s} {'EI₃₀':>10s} {'Эрозивн.':>10s}")
print("-" * 70)
for e in events_sorted:
    if e['total_mm'] >= 5:
        print(f"{str(e['start']):20s} {e['duration_h']:8.1f} {e['total_mm']:10.1f} {e['max_i30']:10.1f} {e['ei30']:10.1f} {'ДА' if e['is_erosive'] else 'нет':>10s}")

# ============================================================
# 6. Сравнение с синоптической станцией Казань
# ============================================================
print("\n" + "=" * 70)
print("6. СРАВНЕНИЕ С СИНОПТИЧЕСКОЙ СТАНЦИЕЙ КАЗАНЬ (ЕСЛИ ДАННЫЕ ДОСТУПНЫ)")
print("=" * 70)

kazan_file = os.path.join(imerg_dir, "data", "stations", "kazan_precip.csv")
if os.path.exists(kazan_file):
    kz = pd.read_csv(kazan_file, parse_dates=['date'])
    kz_2025 = kz[(kz['date'] >= '2025-05-01') & (kz['date'] < '2025-10-01')]
    if len(kz_2025) > 0:
        print(f"  Казань, конвективный сезон 2025: {kz_2025['precip'].sum():.1f} мм")
    else:
        print("  Данные за 2025 год отсутствуют.")
else:
    print("  Файл синоптических данных Казани не найден.")

print("\n" + "=" * 70)
print("ИТОГОВАЯ СВОДКА")
print("=" * 70)
print(f"Станция: TR-525M (55.27°N, 49.28°E)")
print(f"Тип датчика: тиковый плювиограф (только жидкие осадки)")
print(f"Временное разрешение: 30 минут")
print(f"Конвективный сезон 2025 (май–сентябрь):")
print(f"  Сумма жидких осадков: {conv_total:.1f} мм")
print(f"  Эрозивных событий: {len(erosive_events)}")
print(f"  R-фактор: {r_factor:.1f} МДж·мм·га⁻¹·ч⁻¹")
