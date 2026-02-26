# -*- coding: utf-8 -*-
"""
01_compute_stats.py — Сводная статистика R-фактора по годам.

Результат: output/tables/annual_stats.csv
"""
import os
import sys
import glob

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, OUTPUT_DIR, YEARS, FILENAME_TEMPLATE, NODATA_VALUE


def load_raster(year: int):
    """Загрузить годовой растр и вернуть (data, profile)."""
    path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(year=year))
    with rasterio.open(path) as ds:
        data = ds.read(1)
        profile = ds.profile.copy()
    return data, profile


def valid_mask(data: np.ndarray) -> np.ndarray:
    """Маска валидных пикселей (исключая NoData=0 и NaN)."""
    return (data != NODATA_VALUE) & np.isfinite(data)


def compute_annual_stats():
    """Рассчитать статистику по всем годам."""
    rows = []
    for year in YEARS:
        data, _ = load_raster(year)
        mask = valid_mask(data)
        v = data[mask]

        if v.size == 0:
            print(f"  {year}: нет валидных пикселей!")
            continue

        row = {
            "year": year,
            "n_valid": v.size,
            "n_nodata": int(np.sum(~mask)),
            "min": np.min(v),
            "p5": np.percentile(v, 5),
            "p25": np.percentile(v, 25),
            "median": np.median(v),
            "mean": np.mean(v),
            "p75": np.percentile(v, 75),
            "p95": np.percentile(v, 95),
            "max": np.max(v),
            "std": np.std(v),
            "cv": np.std(v) / np.mean(v) * 100,
        }
        rows.append(row)
        print(f"  {year}: mean={row['mean']:.1f}, median={row['median']:.1f}, "
              f"P5-P95=[{row['p5']:.0f}–{row['p95']:.0f}], n={row['n_valid']}")

    return rows


def save_csv(rows, path):
    """Сохранить таблицу в CSV."""
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows([{k: (f"{v:.2f}" if isinstance(v, float) else v)
                           for k, v in row.items()} for row in rows])
    print(f"\n  Сохранено: {path}")


def main():
    print("=" * 60)
    print("Расчёт годовой статистики R-фактора IMERG (2001–2024)")
    print("=" * 60)

    rows = compute_annual_stats()

    # Сохранить CSV
    csv_path = os.path.join(OUTPUT_DIR, "tables", "annual_stats.csv")
    save_csv(rows, csv_path)

    # Итоговая сводка
    means = [r["mean"] for r in rows]
    print(f"\n{'='*60}")
    print(f"  Средне-доменный R: {np.mean(means):.1f} ({min(means):.1f} – {max(means):.1f})")
    print(f"  CV межгодовой: {np.std(means)/np.mean(means)*100:.1f}%")
    print(f"  Валидных пикселей: {rows[0]['n_valid']}")


if __name__ == "__main__":
    main()
