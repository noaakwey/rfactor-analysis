# -*- coding: utf-8 -*-
"""
07_compare_k05_k082.py — Сравнение R-фактора, рассчитанного с k=0.05 и k=0.082.

Что делает скрипт:
  1. Читает оба набора годовых TIF (k=0.05 и k=0.082) за 2001–2024.
  2. Вычисляет попиксельный ratio = R_082 / R_05, статистику по домену.
  3. Строит 6 публикационных графиков:
       fig01_mean_spatial_comparison.png  — карты средних R двух вариантов
       fig02_ratio_spatial.png            — карта ratio и его гистограмма
       fig03_timeseries_comparison.png    — временные ряды обоих вариантов
       fig04_scatter.png                  — scatter plot пиксель-к-пикселю
       fig05_energy_curve.png             — e(i)-кривые при k=0.05 и k=0.082
       fig06_lit_comparison.png           — сравнение с мировыми данными
  4. Сохраняет CSV со сводной статистикой.

Вывод: docs/figures/ и output/tables/compare_k05_k082.csv
"""
import os
import sys
import csv

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import NODATA_VALUE, R_UNITS, DPI, CMAP_R

REPO_DIR   = os.path.dirname(os.path.dirname(__file__))
DOCS_FIG   = os.path.join(REPO_DIR, "docs", "figures")
TABLE_DIR  = os.path.join(REPO_DIR, "output", "tables")
os.makedirs(DOCS_FIG, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

YEARS = list(range(2001, 2025))

# Пути к наборам данных
DIR_K05  = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg\annual"
DIR_K082 = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\annual"

FNAME = "R_imerg_{year}.tif"

FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK  = 9

# ---------------------------------------------------------------------------
# Литературные значения R-фактора для умеренного климата (SI: MJ mm ha-1 h-1 yr-1)
# ---------------------------------------------------------------------------
LIT_DATA = [
    # region, R_mean, R_min, R_max, source
    ("Татарстан (Ларионов 1993)",      110,  70, 160, "Ларионов (1993)"),
    ("Европейская Россия (Зона B)",     150, 100, 220, "Панов et al. (2020)"),
    ("Центральная Европа",              500, 300, 800, "Panagos et al. (2015)"),
    ("Восточная Европа (ср.)",          300, 150, 500, "Ballabio et al. (2017)"),
    ("Сев.-Зап. Китай (аридный)",       100,  40, 200, "Xu et al. (2022)"),
    ("Корея (умеренный)",              2500,1800,3500, "Kim et al. (2021)"),
    ("Западная Европа (ср.)",           700, 300,1500, "Panagos et al. (2015)"),
    ("Средиземноморье",                1200, 500,3000, "Panagos et al. (2015)"),
]

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def load_annual(data_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Загрузить все годы в 3-D стек, вернуть (stack, valid_mask_2d, profile)."""
    stack = []
    profile = None
    for y in YEARS:
        p = os.path.join(data_dir, FNAME.format(year=y))
        with rasterio.open(p) as ds:
            band = ds.read(1).astype(np.float64)
            if profile is None:
                profile = ds.profile.copy()
        band[band == NODATA_VALUE] = np.nan
        stack.append(band)
    stack = np.array(stack)                   # (N, H, W)
    valid2d = np.isfinite(stack[0])           # domain mask
    return stack, valid2d, profile


def get_extent(profile: dict) -> list:
    t = profile["transform"]
    w, h = profile["width"], profile["height"]
    return [t.c, t.c + t.a * w, t.f + t.e * h, t.f]


def domain_mean(stack: np.ndarray) -> np.ndarray:
    """Среднее по домену для каждого года."""
    N = stack.shape[0]
    return np.array([np.nanmean(stack[i]) for i in range(N)])


def domain_percentile(stack: np.ndarray, q: float) -> np.ndarray:
    N = stack.shape[0]
    return np.array([np.nanpercentile(stack[i], q) for i in range(N)])


def save_fig(fig, filename: str) -> None:
    path = os.path.join(DOCS_FIG, filename)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Сохранено: {path}")


# ---------------------------------------------------------------------------
# Рисунок 1 — Пространственные карты среднемноголетнего R
# ---------------------------------------------------------------------------

def plot_mean_spatial(mean05, mean082, profile):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5),
                              gridspec_kw={"width_ratios": [1, 1, 1]},
                              constrained_layout=True)
    extent = get_extent(profile)

    vmin = np.nanpercentile(np.stack([mean05, mean082]), 2)
    vmax = np.nanpercentile(np.stack([mean05, mean082]), 98)

    for ax, data, lbl in zip(axes[:2], [mean05, mean082],
                              ["$k = 0.05$ (классика)", "$k = 0.082$ (RUSLE2)"]):
        im = ax.imshow(data, extent=extent, cmap=CMAP_R,
                       vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(f"Среднемноголетний R\n{lbl}", fontsize=FONT_TITLE, fontweight="bold")
        ax.set_xlabel("Долгота, °E", fontsize=FONT_LABEL)
        ax.set_ylabel("Широта, °N", fontsize=FONT_LABEL)
        ax.grid(alpha=0.25, lw=0.5)
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02,
                     label=f"R, {R_UNITS}")
        v = data[np.isfinite(data)]
        ax.text(0.02, 0.02,
                f"Mean={np.mean(v):.0f}\nMedian={np.median(v):.0f}",
                transform=ax.transAxes, fontsize=FONT_TICK, va="bottom",
                bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.3"))

    # Разность
    diff = mean082 - mean05
    abs_max = np.nanpercentile(np.abs(diff), 98)
    im2 = axes[2].imshow(diff, extent=extent, cmap="RdBu_r",
                          vmin=-abs_max, vmax=abs_max, interpolation="nearest", aspect="auto")
    axes[2].set_title("Разность R (k=0.082 − k=0.05)", fontsize=FONT_TITLE, fontweight="bold")
    axes[2].set_xlabel("Долгота, °E", fontsize=FONT_LABEL)
    axes[2].set_ylabel("Широта, °N", fontsize=FONT_LABEL)
    axes[2].grid(alpha=0.25, lw=0.5)
    fig.colorbar(im2, ax=axes[2], shrink=0.82, pad=0.02,
                 label=f"ΔR, {R_UNITS}")

    save_fig(fig, "fig01_mean_spatial_comparison.png")


# ---------------------------------------------------------------------------
# Рисунок 2 — Карта отношения R_082/R_05
# ---------------------------------------------------------------------------

def plot_ratio_spatial(mean05, mean082, profile):
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(mean05 > 0, mean082 / mean05, np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    constrained_layout=True)
    extent = get_extent(profile)

    im = ax1.imshow(ratio, extent=extent, cmap="RdYlGn",
                    vmin=1.0, vmax=1.4, interpolation="nearest", aspect="auto")
    ax1.set_title("Отношение R (k=0.082) / R (k=0.05)", fontsize=FONT_TITLE, fontweight="bold")
    ax1.set_xlabel("Долгота, °E", fontsize=FONT_LABEL)
    ax1.set_ylabel("Широта, °N", fontsize=FONT_LABEL)
    ax1.grid(alpha=0.25, lw=0.5)
    cbar = fig.colorbar(im, ax=ax1, shrink=0.82, pad=0.02, label="Ratio R₀.₀₈₂ / R₀.₀₅")
    r = ratio[np.isfinite(ratio)]
    ax1.text(0.02, 0.02,
             f"Mean ratio={np.mean(r):.3f}\nMedian={np.median(r):.3f}\nCV={np.std(r)/np.mean(r)*100:.1f}%",
             transform=ax1.transAxes, fontsize=FONT_TICK, va="bottom",
             bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.3"))

    # Гистограмма
    ax2.hist(r, bins=60, color="#2c7bb6", edgecolor="none", alpha=0.8)
    ax2.axvline(np.mean(r), color="red", lw=1.5, ls="--",
                label=f"Mean = {np.mean(r):.3f}")
    ax2.axvline(np.median(r), color="orange", lw=1.5, ls=":",
                label=f"Median = {np.median(r):.3f}")
    ax2.set_xlabel("R₀.₀₈₂ / R₀.₀₅", fontsize=FONT_LABEL)
    ax2.set_ylabel("Количество пикселей", fontsize=FONT_LABEL)
    ax2.set_title("Распределение отношения R₀.₀₈₂/R₀.₀₅", fontsize=FONT_TITLE, fontweight="bold")
    ax2.legend(fontsize=FONT_TICK)
    ax2.grid(alpha=0.3, axis="y")

    save_fig(fig, "fig02_ratio_spatial.png")
    return ratio


# ---------------------------------------------------------------------------
# Рисунок 3 — Временные ряды
# ---------------------------------------------------------------------------

def plot_timeseries(stack05, stack082):
    m05   = domain_mean(stack05)
    m082  = domain_mean(stack082)
    p25_05  = domain_percentile(stack05, 25)
    p75_05  = domain_percentile(stack05, 75)
    p25_082 = domain_percentile(stack082, 25)
    p75_082 = domain_percentile(stack082, 75)

    years = np.array(YEARS, dtype=float)
    sl05, ic05, r05, p05, _ = sp_stats.linregress(years, m05)
    sl82, ic82, r82, p82, _ = sp_stats.linregress(years, m082)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9),
                                    gridspec_kw={"height_ratios": [3, 1.2]},
                                    constrained_layout=True)

    # Верхний
    ax1.fill_between(YEARS, p25_05, p75_05, alpha=0.12, color="#2c7bb6", label="P25–P75 (k=0.05)")
    ax1.fill_between(YEARS, p25_082, p75_082, alpha=0.12, color="#d7191c", label="P25–P75 (k=0.082)")
    ax1.plot(YEARS, m05, "o-", color="#2c7bb6", lw=1.8, ms=5, label=f"k=0.05 (mean={np.mean(m05):.0f})")
    ax1.plot(YEARS, m082, "s-", color="#d7191c", lw=1.8, ms=5, label=f"k=0.082 (mean={np.mean(m082):.0f})")
    ax1.plot(years, sl05 * years + ic05, "--", color="#2c7bb6", lw=1.0, alpha=0.7,
             label=f"Тренд k=0.05: {sl05:+.1f} ед./год (p={p05:.3f})")
    ax1.plot(years, sl82 * years + ic82, "--", color="#d7191c", lw=1.0, alpha=0.7,
             label=f"Тренд k=0.082: {sl82:+.1f} ед./год (p={p82:.3f})")
    ax1.set_ylabel(f"R-фактор, {R_UNITS}", fontsize=FONT_LABEL)
    ax1.set_title("Межгодовая динамика R-фактора RUSLE: сравнение формул кинетической энергии",
                  fontsize=FONT_TITLE, fontweight="bold")
    ax1.legend(fontsize=FONT_TICK, ncol=2, framealpha=0.9)
    ax1.grid(alpha=0.3, axis="y")
    ax1.set_xticks(YEARS)
    ax1.set_xticklabels([str(y) if y % 2 == 1 else "" for y in YEARS], fontsize=8)

    # Нижний — аномалии разности
    delta = m082 - m05
    colors = ["#d7191c" if d > 0 else "#2c7bb6" for d in delta]
    ax2.bar(YEARS, delta, color=colors, edgecolor="gray", linewidth=0.3, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(np.mean(delta), color="darkorange", lw=1.2, ls="--",
                label=f"ΔR среднее = {np.mean(delta):.1f}")
    ax2.set_ylabel("ΔR (k=0.082 − k=0.05)", fontsize=FONT_LABEL)
    ax2.set_xlabel("Год", fontsize=FONT_LABEL)
    ax2.legend(fontsize=FONT_TICK)
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_xticks(YEARS)
    ax2.set_xticklabels([str(y) if y % 2 == 1 else "" for y in YEARS], fontsize=8)

    save_fig(fig, "fig03_timeseries_comparison.png")
    return m05, m082


# ---------------------------------------------------------------------------
# Рисунок 4 — Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(stack05, stack082):
    # Усреднённые карты
    with np.errstate(invalid="ignore"):
        mean05  = np.nanmean(stack05, axis=0)
        mean082 = np.nanmean(stack082, axis=0)

    v05  = mean05[np.isfinite(mean05)  & np.isfinite(mean082)]
    v082 = mean082[np.isfinite(mean05) & np.isfinite(mean082)]

    slope, intercept, rval, _, _ = sp_stats.linregress(v05, v082)
    x_line = np.linspace(v05.min(), v05.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(v05, v082, s=6, alpha=0.3, color="#404040", rasterized=True, label="Пиксели")
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.5,
            label=f"OLS: y={slope:.3f}x+{intercept:.1f}, R²={rval**2:.4f}")
    ax.plot(x_line, x_line, "k:", lw=1.0, label="1:1")
    ax.set_xlabel(f"R (k=0.05), {R_UNITS}", fontsize=FONT_LABEL)
    ax.set_ylabel(f"R (k=0.082), {R_UNITS}", fontsize=FONT_LABEL)
    ax.set_title("Попиксельное сравнение среднемноголетнего R\n(k=0.082 vs k=0.05)",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_TICK)
    ax.grid(alpha=0.3)

    save_fig(fig, "fig04_scatter.png")
    return slope, rval


# ---------------------------------------------------------------------------
# Рисунок 5 — Кривые единичной кинетической энергии e(i)
# ---------------------------------------------------------------------------

def plot_energy_curve():
    i_arr = np.linspace(0, 100, 500)
    e05   = 0.29 * (1.0 - 0.72 * np.exp(-0.05  * i_arr))
    e082  = 0.29 * (1.0 - 0.72 * np.exp(-0.082 * i_arr))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Кривые e(i)
    ax1.plot(i_arr, e05,  color="#2c7bb6", lw=2.0, label="$k=0.05$ (классика)")
    ax1.plot(i_arr, e082, color="#d7191c", lw=2.0, label="$k=0.082$ (RUSLE2)")
    ax1.axhline(0.29, color="gray", ls=":", lw=1.0, label="$e_{\\max}=0.29$")
    ax1.axvline(25.4, color="green", ls="--", lw=1.0, alpha=0.7, label="$i=25.4$ мм/ч")
    ax1.set_xlabel("Интенсивность дождя $i$, мм/ч", fontsize=FONT_LABEL)
    ax1.set_ylabel("$e(i)$, МДж·га$^{-1}$·мм$^{-1}$", fontsize=FONT_LABEL)
    ax1.set_title("Кинетическая энергия единицы осадков\n$e(i) = 0.29 (1 - 0.72 e^{-ki})$",
                  fontsize=FONT_TITLE, fontweight="bold")
    ax1.legend(fontsize=FONT_TICK)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 100)

    # Относительная разность
    with np.errstate(invalid="ignore", divide="ignore"):
        rel_diff = np.where(e05 > 0, (e082 - e05) / e05 * 100, np.nan)
    ax2.plot(i_arr, rel_diff, color="darkorange", lw=2.0)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.axvline(25.4, color="green", ls="--", lw=1.0, alpha=0.7, label="$i=25.4$ мм/ч")
    ax2.fill_between(i_arr, 0, rel_diff, alpha=0.15, color="darkorange")
    ax2.set_xlabel("Интенсивность дождя $i$, мм/ч", fontsize=FONT_LABEL)
    ax2.set_ylabel("Относительное превышение e(0.082) над e(0.05), %", fontsize=FONT_LABEL)
    ax2.set_title("Относительная разность $e(0.082)/e(0.05) - 1$",
                  fontsize=FONT_TITLE, fontweight="bold")
    ax2.legend(fontsize=FONT_TICK)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 100)

    save_fig(fig, "fig05_energy_curve.png")


# ---------------------------------------------------------------------------
# Рисунок 6 — Сравнение с мировой литературой
# ---------------------------------------------------------------------------

def plot_lit_comparison(mean_r05, mean_r082):
    regions = [d[0] for d in LIT_DATA]
    r_mean  = [d[1] for d in LIT_DATA]
    r_min   = [d[2] for d in LIT_DATA]
    r_max   = [d[3] for d in LIT_DATA]
    sources = [d[4] for d in LIT_DATA]

    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)

    y_pos = np.arange(len(regions))
    xerr_lo = [rm - rmin for rm, rmin in zip(r_mean, r_min)]
    xerr_hi = [rmax - rm for rm, rmax in zip(r_mean, r_max)]

    ax.barh(y_pos, r_mean, xerr=[xerr_lo, xerr_hi],
            height=0.55, color="#7fcdbb", edgecolor="grey", linewidth=0.5,
            error_kw=dict(ecolor="gray", lw=1.2, capsize=4), alpha=0.85,
            label="Литературные данные (среднее ± диапазон)")

    ax.axvline(mean_r05, color="#2c7bb6", lw=2.0, ls="-.",
               label=f"Данное исследование (k=0.05): {mean_r05:.0f}")
    ax.axvline(mean_r082, color="#d7191c", lw=2.0, ls="--",
               label=f"Данное исследование (k=0.082): {mean_r082:.0f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(regions, fontsize=FONT_TICK)
    ax.set_xlabel(f"R-фактор, {R_UNITS}", fontsize=FONT_LABEL)
    ax.set_title("Сравнение рассчитанного R-фактора с опубликованными данными\n"
                 "для умеренных и других климатических зон",
                 fontsize=FONT_TITLE, fontweight="bold")
    ax.legend(fontsize=FONT_TICK, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, max(r_max) * 1.15)

    save_fig(fig, "fig06_lit_comparison.png")


# ---------------------------------------------------------------------------
# CSV-сводка
# ---------------------------------------------------------------------------

def save_summary_csv(m05, m082, ratio, slope):
    rows = []
    years = np.array(YEARS, dtype=float)
    sl05, _, r05, p05, _ = sp_stats.linregress(years, m05)
    sl82, _, r82, p82, _ = sp_stats.linregress(years, m082)

    for i, y in enumerate(YEARS):
        rows.append({
            "year": y,
            "R_k05":   f"{m05[i]:.2f}",
            "R_k082":  f"{m082[i]:.2f}",
            "delta":   f"{m082[i] - m05[i]:.2f}",
            "ratio":   f"{m082[i] / m05[i] if m05[i] > 0 else 'nan':.4f}",
        })

    stat_row = {
        "year": "MEAN",
        "R_k05":  f"{np.mean(m05):.2f}",
        "R_k082": f"{np.mean(m082):.2f}",
        "delta":  f"{np.mean(m082) - np.mean(m05):.2f}",
        "ratio":  f"{np.mean(m082) / np.mean(m05):.4f}" if np.mean(m05) > 0 else "nan",
    }
    rows.append(stat_row)

    path = os.path.join(TABLE_DIR, "compare_k05_k082.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["year","R_k05","R_k082","delta","ratio"])
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV: {path}")

    # Сводные статистики
    print("\n  === СВОДНЫЕ СТАТИСТИКИ ===")
    print(f"  R (k=0.05):  mean={np.mean(m05):.1f}, std={np.std(m05):.1f}, "
          f"CV={np.std(m05)/np.mean(m05)*100:.1f}%, тренд={sl05:+.2f} ед/год (p={p05:.3f})")
    print(f"  R (k=0.082): mean={np.mean(m082):.1f}, std={np.std(m082):.1f}, "
          f"CV={np.std(m082)/np.mean(m082)*100:.1f}%, тренд={sl82:+.2f} ед/год (p={p82:.3f})")
    print(f"  Ratio mean:  {np.mean(m082)/np.mean(m05):.4f}  "
          f"(пространственный: mean={np.nanmean(ratio):.4f}, std={np.nanstd(ratio):.4f})")

    return sl05, p05, sl82, p82


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("Сравнительный анализ R-фактора: k=0.05 vs k=0.082")
    print("=" * 65)

    print("\n[1/7] Загрузка данных k=0.05 ...")
    stack05, valid2d, profile = load_annual(DIR_K05)

    print("[2/7] Загрузка данных k=0.082 ...")
    stack082, _, _ = load_annual(DIR_K082)

    with np.errstate(invalid="ignore"):
        mean05  = np.nanmean(stack05, axis=0)
        mean082 = np.nanmean(stack082, axis=0)

    print("[3/7] Пространственные карты среднемноголетнего R ...")
    plot_mean_spatial(mean05, mean082, profile)

    print("[4/7] Карта отношения R082/R05 ...")
    ratio = plot_ratio_spatial(mean05, mean082, profile)

    print("[5/7] Временные ряды ...")
    m05, m082 = plot_timeseries(stack05, stack082)

    print("[6/7] Scatter plot ...")
    slope_ols, rval = plot_scatter(stack05, stack082)

    print("[7/7] Кривые e(i) и сравнение с литературой ...")
    plot_energy_curve()
    plot_lit_comparison(np.mean(m05), np.mean(m082))

    sl05, p05, sl82, p82 = save_summary_csv(m05, m082, ratio, slope_ols)

    print(f"\n  OLS scatter: slope={slope_ols:.4f}, R2={rval**2:.4f}")
    print("\nГотово. Рисунки в docs/figures/, таблица в output/tables/")


if __name__ == "__main__":
    main()
