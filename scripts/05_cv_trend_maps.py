# -*- coding: utf-8 -*-
"""
05_cv_trend_maps.py — Карты пространственного CV и попиксельного тренда R-фактора.

Результат:
  output/maps/R_cv_2001_2024.png
  output/maps/R_trend_2001_2024.png
  output/maps/R_cv_2001_2024.tif
  output/maps/R_trend_2001_2024.tif
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (DATA_DIR, OUTPUT_DIR, YEARS, FILENAME_TEMPLATE,
                    NODATA_VALUE, R_UNITS, CMAP_CV, CMAP_TREND, DPI, FIGSIZE_MAP)


def load_stack():
    """Загрузить все годовые растры в 3D-стек."""
    stack = []
    profile = None
    for year in YEARS:
        path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(year=year))
        with rasterio.open(path) as ds:
            data = ds.read(1).astype(np.float64)
            if profile is None:
                profile = ds.profile.copy()
            data[data == NODATA_VALUE] = np.nan
            stack.append(data)
    return np.array(stack), profile  # (N, H, W)


def compute_cv(stack):
    """Попиксельный коэффициент вариации (%)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        cv = np.where(mean > 0, std / mean * 100, np.nan)
    return cv


def compute_pixel_trend(stack, years):
    """Попиксельный линейный тренд (slope, p-value)."""
    N, H, W = stack.shape
    slope_map = np.full((H, W), np.nan)
    pval_map = np.full((H, W), np.nan)

    years_arr = np.array(years, dtype=np.float64)

    for i in range(H):
        for j in range(W):
            ts = stack[:, i, j]
            valid = np.isfinite(ts)
            if np.sum(valid) < 5:
                continue
            s, _intercept, _r, p, _se = sp_stats.linregress(years_arr[valid], ts[valid])
            slope_map[i, j] = s
            pval_map[i, j] = p

    return slope_map, pval_map


def save_geotiff(data, profile, path, description=""):
    """Сохранить GeoTIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    p = profile.copy()
    p.update(dtype="float32", count=1, nodata=np.nan,
             compress="deflate", predictor=3)
    with rasterio.open(path, "w", **p) as ds:
        ds.write(data.astype(np.float32), 1)
        if description:
            ds.update_tags(description=description)
    print(f"  GeoTIFF: {path}")


def get_extent(profile):
    t = profile["transform"]
    w, h = profile["width"], profile["height"]
    return [t.c, t.c + t.a * w, t.f + t.e * h, t.f]


def plot_cv(cv, profile, png_path):
    """Визуализация CV."""
    extent = get_extent(profile)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MAP)

    vmin = np.nanpercentile(cv, 2)
    vmax = np.nanpercentile(cv, 98)

    im = ax.imshow(cv, extent=extent, cmap=CMAP_CV,
                   vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("CV, %", fontsize=11)

    ax.set_xlabel("Долгота, °E", fontsize=11)
    ax.set_ylabel("Широта, °N", fontsize=11)
    ax.set_title(f"Коэффициент вариации R-фактора ({YEARS[0]}–{YEARS[-1]})",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linewidth=0.5)

    valid = cv[np.isfinite(cv)]
    ax.text(0.02, 0.02,
            f"Mean CV: {np.mean(valid):.0f}%\nMedian CV: {np.median(valid):.0f}%",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Карта: {png_path}")


def plot_trend(slope_map, pval_map, profile, png_path):
    """Визуализация тренда с маскировкой незначимых пикселей."""
    extent = get_extent(profile)
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE_MAP)

    # Маскировать незначимые пиксели (p >= 0.05)
    display = np.where(pval_map < 0.05, slope_map, np.nan)

    # Дивергентная шкала
    valid = display[np.isfinite(display)]
    if valid.size > 0:
        abs_max = max(abs(np.nanpercentile(valid, 2)), abs(np.nanpercentile(valid, 98)))
    else:
        abs_max = 5.0

    im = ax.imshow(display, extent=extent, cmap=CMAP_TREND,
                   vmin=-abs_max, vmax=abs_max, interpolation="nearest", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Тренд R, ед./год", fontsize=11)

    # Показать незначимые пиксели серым фоном
    bg = slope_map.copy()
    bg[np.isfinite(display)] = np.nan
    bg[np.isnan(slope_map)] = np.nan
    ax.imshow(np.where(np.isfinite(bg), 0.5, np.nan), extent=extent,
              cmap="Greys", vmin=0, vmax=1, alpha=0.15, interpolation="nearest",
              aspect="auto")

    ax.set_xlabel("Долгота, °E", fontsize=11)
    ax.set_ylabel("Широта, °N", fontsize=11)
    ax.set_title(f"Линейный тренд R-фактора ({YEARS[0]}–{YEARS[-1]})\n"
                 f"Показаны только значимые пиксели (p < 0.05)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, linewidth=0.5)

    n_sig = np.sum(np.isfinite(display))
    n_total = np.sum(np.isfinite(slope_map))
    ax.text(0.02, 0.02,
            f"Значимых пикселей: {n_sig}/{n_total} ({n_sig/max(n_total,1)*100:.0f}%)",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Карта: {png_path}")


def main():
    print("=" * 60)
    print(f"Карты CV и тренда R-фактора ({YEARS[0]}–{YEARS[-1]})")
    print("=" * 60)

    stack, profile = load_stack()

    # CV
    print("\nCV...")
    cv = compute_cv(stack)
    save_geotiff(cv, profile,
                 os.path.join(OUTPUT_DIR, "maps", f"R_cv_{YEARS[0]}_{YEARS[-1]}.tif"),
                 description="Pixel-wise CV of annual R-factor (%)")
    plot_cv(cv, profile,
            os.path.join(OUTPUT_DIR, "maps", f"R_cv_{YEARS[0]}_{YEARS[-1]}.png"))

    # Тренд
    print("\nПопиксельный тренд...")
    slope_map, pval_map = compute_pixel_trend(stack, YEARS)
    save_geotiff(slope_map, profile,
                 os.path.join(OUTPUT_DIR, "maps", f"R_trend_{YEARS[0]}_{YEARS[-1]}.tif"),
                 description="Pixel-wise linear trend slope (units/yr)")
    plot_trend(slope_map, pval_map, profile,
               os.path.join(OUTPUT_DIR, "maps", f"R_trend_{YEARS[0]}_{YEARS[-1]}.png"))


if __name__ == "__main__":
    main()
