# -*- coding: utf-8 -*-
"""
03_annual_maps.py — Small-multiples карты R-фактора по годам.

Результат: output/maps/R_annual_small_multiples.png
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (DATA_DIR, OUTPUT_DIR, YEARS, FILENAME_TEMPLATE,
                    NODATA_VALUE, R_UNITS, CMAP_R, DPI)


def load_all_years():
    """Загрузить все годовые растры, возвращает (stack, bounds, profile)."""
    stack = []
    profile = None
    for year in YEARS:
        path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(year=year))
        with rasterio.open(path) as ds:
            data = ds.read(1).astype(np.float32)
            if profile is None:
                profile = ds.profile.copy()
            data[data == NODATA_VALUE] = np.nan
            stack.append(data)
    return np.array(stack), profile


def main():
    print("=" * 60)
    print("Карты R-фактора по годам (small multiples)")
    print("=" * 60)

    stack, profile = load_all_years()

    # Экстент
    t = profile["transform"]
    w, h = profile["width"], profile["height"]
    extent = [t.c, t.c + t.a * w, t.f + t.e * h, t.f]

    # Единая шкала
    vmin = np.nanpercentile(stack, 1)
    vmax = np.nanpercentile(stack, 99)

    nrows, ncols = 4, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 13),
                             constrained_layout=True)

    for idx, (year, data) in enumerate(zip(YEARS, stack)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        im = ax.imshow(data, extent=extent, cmap=CMAP_R,
                       vmin=vmin, vmax=vmax, interpolation="nearest",
                       aspect="auto")
        valid = data[np.isfinite(data)]
        ax.set_title(f"{year}  (μ={np.mean(valid):.0f})", fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.2, linewidth=0.3)

    # Скрыть лишние субплоты
    for idx in range(len(YEARS), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    # Общий colourbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(f"R-фактор, {R_UNITS}", fontsize=11)

    fig.suptitle("Годовой R-фактор RUSLE (IMERG V07, калиброванный)",
                 fontsize=14, fontweight="bold")

    out_path = os.path.join(OUTPUT_DIR, "maps", "R_annual_small_multiples.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Сохранено: {out_path}")


if __name__ == "__main__":
    main()
