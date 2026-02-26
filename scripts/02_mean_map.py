# -*- coding: utf-8 -*-
"""
02_mean_map.py — Карта среднемноголетнего R-фактора.

Результат:
  output/maps/R_mean_2001_2024.tif   — GeoTIFF
  output/maps/R_mean_2001_2024.png   — визуализация
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (DATA_DIR, OUTPUT_DIR, YEARS, FILENAME_TEMPLATE,
                    NODATA_VALUE, R_UNITS, CMAP_R, DPI, FIGSIZE_MAP)


def compute_mean_raster():
    """Рассчитать среднемноголетний R (2001–2024)."""
    stack = []
    profile = None
    for year in YEARS:
        path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(year=year))
        with rasterio.open(path) as ds:
            data = ds.read(1).astype(np.float64)
            if profile is None:
                profile = ds.profile.copy()
            # Заменить NoData на NaN для корректного усреднения
            data[data == NODATA_VALUE] = np.nan
            stack.append(data)

    stack = np.array(stack)  # (N, H, W)
    with np.errstate(invalid='ignore'):
        mean_r = np.nanmean(stack, axis=0)

    return mean_r, profile


def save_geotiff(data, profile, path):
    """Сохранить GeoTIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan,
             compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as ds:
        ds.write(data.astype(np.float32), 1)
        ds.update_tags(
            description=f"Mean annual R-factor ({YEARS[0]}-{YEARS[-1]})",
            units="MJ mm ha-1 h-1 yr-1",
        )
    print(f"  GeoTIFF: {path}")


def plot_mean_map(mean_r, profile, png_path):
    """Визуализация среднемноголетнего R."""
    bounds = profile["transform"] * (0, 0), profile["transform"] * (profile["width"], profile["height"])
    extent = [bounds[0][0], bounds[1][0], bounds[1][1], bounds[0][1]]

    # Маскировать NoData для отображения
    display = np.where(np.isnan(mean_r), np.nan, mean_r)

    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MAP)

    vmin = np.nanpercentile(display, 2)
    vmax = np.nanpercentile(display, 98)

    im = ax.imshow(display, extent=extent, cmap=CMAP_R,
                   vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"R-фактор, {R_UNITS}", fontsize=11)

    ax.set_xlabel("Долгота, °E", fontsize=11)
    ax.set_ylabel("Широта, °N", fontsize=11)
    ax.set_title(f"Среднемноголетний R-фактор RUSLE ({YEARS[0]}–{YEARS[-1]})\n"
                 f"IMERG V07 (калиброванный), 0.1°",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linewidth=0.5)

    # Статистика в углу
    valid = display[np.isfinite(display)]
    stats_text = (f"Mean: {np.mean(valid):.0f}\n"
                  f"Median: {np.median(valid):.0f}\n"
                  f"Min–Max: {np.min(valid):.0f}–{np.max(valid):.0f}")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Карта: {png_path}")


def main():
    print("=" * 60)
    print(f"Карта среднемноголетнего R-фактора ({YEARS[0]}–{YEARS[-1]})")
    print("=" * 60)

    mean_r, profile = compute_mean_raster()

    tif_path = os.path.join(OUTPUT_DIR, "maps", f"R_mean_{YEARS[0]}_{YEARS[-1]}.tif")
    save_geotiff(mean_r, profile, tif_path)

    png_path = os.path.join(OUTPUT_DIR, "maps", f"R_mean_{YEARS[0]}_{YEARS[-1]}.png")
    plot_mean_map(mean_r, profile, png_path)


if __name__ == "__main__":
    main()
