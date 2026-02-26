# -*- coding: utf-8 -*-
"""
04_timeseries.py — Временной ряд R-фактора с трендом.

Результат: output/plots/R_timeseries_trend.png
"""
import os
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (DATA_DIR, OUTPUT_DIR, YEARS, FILENAME_TEMPLATE,
                    NODATA_VALUE, R_UNITS, DPI, FIGSIZE_TS)


def compute_domain_stats():
    """Для каждого года: mean, P25, P75 по валидным пикселям."""
    means, p25s, p75s, medians, p5s, p95s = [], [], [], [], [], []
    for year in YEARS:
        path = os.path.join(DATA_DIR, FILENAME_TEMPLATE.format(year=year))
        with rasterio.open(path) as ds:
            data = ds.read(1)
        mask = (data != NODATA_VALUE) & np.isfinite(data)
        v = data[mask]
        means.append(np.mean(v))
        medians.append(np.median(v))
        p25s.append(np.percentile(v, 25))
        p75s.append(np.percentile(v, 75))
        p5s.append(np.percentile(v, 5))
        p95s.append(np.percentile(v, 95))
    return (np.array(means), np.array(medians),
            np.array(p25s), np.array(p75s),
            np.array(p5s), np.array(p95s))


def main():
    print("=" * 60)
    print("Временной ряд R-фактора с трендом")
    print("=" * 60)

    means, medians, p25, p75, p5, p95 = compute_domain_stats()
    years = np.array(YEARS, dtype=float)

    # Линейный тренд (OLS)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(years, means)
    trend_line = slope * years + intercept

    # Mann-Kendall тренд
    try:
        import pymannkendall as mk
        mk_result = mk.original_test(means)
        mk_text = f"Mann-Kendall p={mk_result.p:.3f}, τ={mk_result.Tau:.3f}"
        print(f"  {mk_text}")
    except ImportError:
        mk_text = None
        print("  (pymannkendall не установлен, только OLS)")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                    gridspec_kw={"height_ratios": [3, 1.2]},
                                    constrained_layout=True)

    # Верхний: barplot + тренд + envelope
    colors = plt.cm.YlOrRd(np.interp(means, [means.min(), means.max()], [0.3, 0.9]))
    ax1.bar(YEARS, means, color=colors, edgecolor="gray", linewidth=0.5,
            alpha=0.85, zorder=2, label="Средне-доменный R")
    ax1.fill_between(YEARS, p25, p75, alpha=0.15, color="steelblue",
                     label="P25–P75", zorder=1)
    ax1.fill_between(YEARS, p5, p95, alpha=0.07, color="steelblue",
                     label="P5–P95", zorder=0)

    ax1.plot(YEARS, trend_line, "k--", linewidth=1.5, alpha=0.7,
             label=f"Тренд: {slope:+.1f} ед./год (p={p_value:.3f})")

    ax1.axhline(np.mean(means), color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax1.set_ylabel(f"R-фактор, {R_UNITS}", fontsize=12)
    ax1.set_title("Межгодовая динамика R-фактора RUSLE (IMERG V07, калиброванный)",
                  fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.3)

    # Аннотация экстремальных лет
    idx_max = np.argmax(means)
    idx_min = np.argmin(means)
    ax1.annotate(f"{YEARS[idx_max]}\n{means[idx_max]:.0f}",
                 xy=(YEARS[idx_max], means[idx_max]),
                 xytext=(0, 15), textcoords="offset points",
                 ha="center", fontsize=8, fontweight="bold", color="darkred",
                 arrowprops=dict(arrowstyle="-", color="darkred", lw=0.5))
    ax1.annotate(f"{YEARS[idx_min]}\n{means[idx_min]:.0f}",
                 xy=(YEARS[idx_min], means[idx_min]),
                 xytext=(0, 15), textcoords="offset points",
                 ha="center", fontsize=8, fontweight="bold", color="navy",
                 arrowprops=dict(arrowstyle="-", color="navy", lw=0.5))

    # Нижний: аномалии
    long_mean = np.mean(means)
    anomalies = means - long_mean
    c_pos = "#d73027"
    c_neg = "#4575b4"
    bar_colors = [c_pos if a > 0 else c_neg for a in anomalies]
    ax2.bar(YEARS, anomalies, color=bar_colors, edgecolor="gray",
            linewidth=0.3, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("Аномалия R", fontsize=11)
    ax2.set_xlabel("Год", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # Текстовая сводка
    summary = (f"Среднее: {long_mean:.0f} {R_UNITS}\n"
               f"CV: {np.std(means)/long_mean*100:.1f}%\n"
               f"Тренд: {slope:+.1f} ед./год (R²={r_value**2:.3f})")
    if mk_text:
        summary += f"\n{mk_text}"
    ax2.text(0.01, 0.95, summary, transform=ax2.transAxes,
             fontsize=8, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

    out_path = os.path.join(OUTPUT_DIR, "plots", "R_timeseries_trend.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Сохранено: {out_path}")

    # Консольная сводка
    print(f"\n  Средний R: {long_mean:.1f}")
    print(f"  Линейный тренд: {slope:+.2f} ед./год, p={p_value:.4f}")
    print(f"  R^2={r_value**2:.4f}")


if __name__ == "__main__":
    main()
