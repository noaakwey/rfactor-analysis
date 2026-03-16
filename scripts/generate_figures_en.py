# -*- coding: utf-8 -*-
"""
generate_figures_en.py
Regenerates ALL 20 figures for the English manuscript at 300 DPI.
Output: docs/figures_en/

Covers:
  fig07–fig14  (from 08_analysis_paper.py)
  fig15–fig16  (from 09_uncertainty.py)
  fig17         (from spatial_autocorr.py)
  fig18         (from evaluate_extreme_years.py)
  fig19         (from structural_breaks.py)
  fig20         (from teleconnections.py)
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist
import rasterio

# ------------------------------------------------------------------
# PATHS (adjust as needed)
# ------------------------------------------------------------------
REPO     = os.path.dirname(os.path.dirname(__file__))
FIG_DIR  = os.path.join(REPO, "docs", "figures_en")
os.makedirs(FIG_DIR, exist_ok=True)

DIR_K082  = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\annual"
MEAN_TIF  = r"D:\Artur\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\R_imerg_2001_2024_MEAN.tif"
PBIAS_CSV = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib\docs\station_annual_pbias_summary_imerg.csv"
COMPARE_CSV = os.path.join(REPO, "output", "tables", "compare_k05_k082.csv")
CALIB_DIR   = r"D:\Artur\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib\output\calib_imerg"
OUTPUT_DIR  = os.path.join(REPO, "output")
RFACTOR_CSV = os.path.join(OUTPUT_DIR, "domain_annual_rfactor.csv")

DPI   = 300
YEARS = list(range(2001, 2025))
NODATA = 0.0

BLUE  = "#2166ac"
RED   = "#d6604d"
GREEN = "#4dac26"
GRAY  = "#636363"

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
def load_annual_k082():
    stack, profile = [], None
    for y in YEARS:
        p = os.path.join(DIR_K082, f"R_imerg_{y}.tif")
        with rasterio.open(p) as ds:
            band = ds.read(1).astype(np.float64)
            if profile is None:
                profile = ds.profile.copy()
        band[band == NODATA] = np.nan
        stack.append(band)
    return np.array(stack), profile

def get_extent(profile):
    t = profile["transform"]
    return [t.c, t.c + t.a * profile["width"],
            t.f + t.e * profile["height"], t.f]

def domain_stat(stack, func):
    return np.array([func(stack[i][np.isfinite(stack[i])]) for i in range(len(stack))])

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved:", path)


# ===================================================================
# FIG 07 — Annual small multiples
# ===================================================================
def fig07_small_multiples(stack, profile):
    ext   = get_extent(profile)
    means = domain_stat(stack, np.nanmean)
    vmin  = np.nanpercentile(stack, 2)
    vmax  = np.nanpercentile(stack, 98)

    fig, axes = plt.subplots(4, 6, figsize=(18, 14), constrained_layout=True)
    cmap = plt.cm.YlOrRd
    for i, y in enumerate(YEARS):
        ax = axes[i // 6, i % 6]
        im = ax.imshow(stack[i], extent=ext, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(f"{y}\n{means[i]:.0f}", fontsize=8, fontweight="bold",
                     color="darkred" if means[i] > np.nanmean(means) else "navy")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_linewidth(0.4)

    fig.colorbar(im, ax=axes, shrink=0.4, pad=0.01,
                 label="R (k = 0.082), MJ·mm·ha⁻¹·h⁻¹·yr⁻¹",
                 orientation="horizontal", aspect=40)
    fig.suptitle("Annual R-Factor Maps (k = 0.082), 2001–2024\n"
                 "IMERG V07 (calibrated), 0.1°",
                 fontsize=14, fontweight="bold", y=1.03)
    save(fig, "fig07_annual_multiples.png")


# ===================================================================
# FIG 08 — Mean map + PDF + CDF
# ===================================================================
def fig08_mean_pdf(stack, profile):
    mean_r = np.nanmean(stack, axis=0)
    ext = get_extent(profile)
    vals = mean_r[np.isfinite(mean_r)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1, 1]})

    ax = axes[0]
    im = ax.imshow(mean_r, extent=ext, cmap="YlOrRd",
                   vmin=np.nanpercentile(mean_r, 2),
                   vmax=np.nanpercentile(mean_r, 98),
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                 label="MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax.set_xlabel("Longitude, °E"); ax.set_ylabel("Latitude, °N")
    ax.set_title("Long-Term Mean R-Factor (2001–2024)\n$k = 0.082$ (RUSLE2)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, lw=0.5)
    ax.text(0.02, 0.02,
            f"Mean = {np.mean(vals):.0f}\nMedian = {np.median(vals):.0f}\n"
            f"P5–P95 = {np.percentile(vals, 5):.0f}–{np.percentile(vals, 95):.0f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(fc="white", alpha=0.85, boxstyle="round,pad=0.3"))

    ax2 = axes[1]
    ax2.hist(vals, bins=40, color=RED, alpha=0.75, edgecolor="none", density=True)
    ax2.axvline(np.mean(vals), color="black", lw=1.5, ls="--", label=f"Mean={np.mean(vals):.0f}")
    ax2.axvline(np.median(vals), color=BLUE, lw=1.5, ls=":", label=f"Median={np.median(vals):.0f}")
    ax2.set_xlabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"); ax2.set_ylabel("Density")
    ax2.set_title("Distribution of\nLong-Term Mean R", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    ax3 = axes[2]
    sorted_v = np.sort(vals)
    cdf = np.arange(1, len(sorted_v)+1) / len(sorted_v)
    ax3.plot(sorted_v, cdf * 100, color=RED, lw=2)
    for p, ls in [(10, ":"), (25, "--"), (50, "-"), (75, "--"), (90, ":")]:
        v = np.percentile(sorted_v, p)
        ax3.axhline(p, color=GRAY, lw=0.8, ls=ls, alpha=0.6)
        ax3.axvline(v, color=GRAY, lw=0.8, ls=ls, alpha=0.6)
        ax3.text(v + 2, p + 1, f"P{p}={v:.0f}", fontsize=7, color=GRAY)
    ax3.set_xlabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"); ax3.set_ylabel("CDF, %")
    ax3.set_title("Cumulative Distribution\nFunction", fontsize=11, fontweight="bold")
    ax3.grid(alpha=0.3); ax3.set_ylim(0, 100)

    save(fig, "fig08_mean_pdf_cdf.png")
    return vals, mean_r


# ===================================================================
# FIG 09 — Temporal dynamics
# ===================================================================
def fig09_temporal(stack):
    means   = domain_stat(stack, np.nanmean)
    medians = domain_stat(stack, np.nanmedian)
    p25 = domain_stat(stack, lambda x: np.percentile(x, 25))
    p75 = domain_stat(stack, lambda x: np.percentile(x, 75))
    p5  = domain_stat(stack, lambda x: np.percentile(x, 5))
    p95 = domain_stat(stack, lambda x: np.percentile(x, 95))

    years = np.array(YEARS, dtype=float)
    mu = np.mean(means)
    sl, ic, r_, p_, _ = sp_stats.linregress(years, means)

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[3, 1.2, 1.2], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])

    colors = plt.cm.YlOrRd(np.interp(means, [means.min(), means.max()], [0.25, 0.92]))
    ax1.bar(YEARS, means, color=colors, edgecolor="gray", lw=0.4, alpha=0.88, zorder=3)
    ax1.fill_between(YEARS, p5, p95, alpha=0.08, color=BLUE, label="P5–P95")
    ax1.fill_between(YEARS, p25, p75, alpha=0.2, color=BLUE, label="P25–P75")
    ax1.plot(YEARS, medians, "o-", color=BLUE, ms=3, lw=1.2, alpha=0.7, label="Median")
    ax1.plot(years, sl * years + ic, "k--", lw=1.5,
             label=f"OLS: {sl:+.1f} units/yr (p={p_:.3f}, R²={r_**2:.3f})")
    ax1.axhline(mu, color=GRAY, lw=1.0, ls=":", alpha=0.7, label=f"Mean: {mu:.0f}")

    imax = np.argmax(means); imin = np.argmin(means)
    for idx, clr, lbl in [(imax, "#a50026", "max"), (imin, "#313695", "min")]:
        ax1.annotate(f"{YEARS[idx]}: {means[idx]:.0f}",
                     xy=(YEARS[idx], means[idx]),
                     xytext=(0, 18 if idx == imax else -28),
                     textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                     color=clr, arrowprops=dict(arrowstyle="-", color=clr, lw=0.7))

    ax1.set_ylabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹", fontsize=11)
    ax1.set_title("Interannual R-Factor Dynamics (RUSLE, k = 0.082)\n"
                  "IMERG V07 (calibrated), 2001–2024",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, ncol=3, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_xlim(2000.5, 2024.5); ax1.set_xticks([])

    anom = means - mu
    bc = [RED if a > 0 else BLUE for a in anom]
    ax2.bar(YEARS, anom, color=bc, edgecolor="gray", lw=0.3, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("R Anomaly", fontsize=10)
    ax2.grid(axis="y", alpha=0.25); ax2.set_xlim(2000.5, 2024.5); ax2.set_xticks([])

    cum = np.cumsum(anom)
    ax3.fill_between(YEARS, 0, cum, where=cum >= 0, color=RED, alpha=0.4)
    ax3.fill_between(YEARS, 0, cum, where=cum < 0, color=BLUE, alpha=0.4)
    ax3.plot(YEARS, cum, color="black", lw=1.5)
    ax3.axhline(0, color="black", lw=0.7)
    ax3.set_ylabel("Cumul. Anomaly", fontsize=10)
    ax3.set_xlabel("Year", fontsize=11)
    ax3.grid(axis="y", alpha=0.25); ax3.set_xlim(2000.5, 2024.5)
    ax3.set_xticks(YEARS)
    ax3.set_xticklabels([str(y) if y % 2 == 1 else "" for y in YEARS], fontsize=8)

    save(fig, "fig09_temporal_dynamics.png")
    return means, mu, sl, p_


# ===================================================================
# FIG 10 — CV + trend
# ===================================================================
def fig10_cv_trend(stack, profile):
    ext = get_extent(profile)
    mu  = np.nanmean(stack, axis=0)
    sig = np.nanstd(stack, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(mu > 0, sig / mu * 100, np.nan)

    years_arr = np.array(YEARS, dtype=float)
    H, W = stack.shape[1], stack.shape[2]
    slope_map = np.full((H, W), np.nan)
    pval_map  = np.full((H, W), np.nan)
    for i in range(H):
        for j in range(W):
            ts = stack[:, i, j]
            ok = np.isfinite(ts)
            if ok.sum() < 6: continue
            s, _, _, p, _ = sp_stats.linregress(years_arr[ok], ts[ok])
            slope_map[i, j] = s
            pval_map[i, j]  = p

    sig_mask = np.where(pval_map < 0.05, slope_map, np.nan)
    abs_max  = np.nanpercentile(np.abs(sig_mask[np.isfinite(sig_mask)]), 98) if np.any(np.isfinite(sig_mask)) else 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    ax = axes[0]
    im = ax.imshow(cv, extent=ext, cmap="viridis",
                   vmin=np.nanpercentile(cv, 2), vmax=np.nanpercentile(cv, 98),
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="CV, %")
    ax.set_title(f"Spatial CV of R-Factor (2001–2024)\nMean CV = {np.nanmean(cv):.0f}%",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude, °E"); ax.set_ylabel("Latitude, °N")
    ax.grid(alpha=0.2, lw=0.5)

    ax2 = axes[1]
    im2 = ax2.imshow(sig_mask, extent=ext, cmap="RdBu_r",
                     vmin=-abs_max, vmax=abs_max, interpolation="nearest", aspect="auto")
    bg = np.where(np.isfinite(slope_map) & ~np.isfinite(sig_mask), 1.0, np.nan)
    ax2.imshow(bg, extent=ext, cmap="Greys", vmin=0, vmax=2,
               alpha=0.18, interpolation="nearest", aspect="auto")
    fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02,
                 label="R Trend, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹/yr")
    n_sig = int(np.sum(np.isfinite(sig_mask)))
    n_tot = int(np.sum(np.isfinite(slope_map)))
    ax2.set_title(f"Linear R-Factor Trend (p < 0.05)\n{n_sig}/{n_tot} significant pixels ({n_sig/max(n_tot,1)*100:.0f}%)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Longitude, °E"); ax2.set_ylabel("Latitude, °N")
    ax2.grid(alpha=0.2, lw=0.5)

    save(fig, "fig10_cv_trend.png")


# ===================================================================
# FIG 11 — Decadal comparison
# ===================================================================
def fig11_decadal(stack, profile):
    ext = get_extent(profile)
    d1 = np.nanmean(stack[0:8],  axis=0)
    d2 = np.nanmean(stack[8:16], axis=0)
    d3 = np.nanmean(stack[16:],  axis=0)
    vmin = np.nanpercentile(np.stack([d1, d2, d3]), 2)
    vmax = np.nanpercentile(np.stack([d1, d2, d3]), 98)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True,
                              gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    lbls   = ["2001–2008", "2009–2016", "2017–2024"]
    data_d = [d1, d2, d3]
    for ax, d, lbl in zip(axes[:3], data_d, lbls):
        im = ax.imshow(d, extent=ext, cmap="YlOrRd",
                       vmin=vmin, vmax=vmax, interpolation="nearest", aspect="auto")
        ax.set_title(f"{lbl}\nMean={np.nanmean(d):.0f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude, °E"); ax.set_ylabel("Latitude, °N")
        ax.grid(alpha=0.2, lw=0.5)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")

    diff = d3 - d1
    amax = np.nanpercentile(np.abs(diff), 98)
    im4 = axes[3].imshow(diff, extent=ext, cmap="RdBu_r",
                          vmin=-amax, vmax=amax, interpolation="nearest", aspect="auto")
    axes[3].set_title(f"2017–2024 minus 2001–2008\nMean delta={np.nanmean(diff):.0f}",
                      fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Longitude, °E"); axes[3].set_ylabel("Latitude, °N")
    axes[3].grid(alpha=0.2, lw=0.5)
    fig.colorbar(im4, ax=axes[3], shrink=0.85, pad=0.02, label="ΔR, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")

    save(fig, "fig11_decadal.png")
    return np.nanmean(d1), np.nanmean(d2), np.nanmean(d3)


# ===================================================================
# FIG 12 — Percentile maps
# ===================================================================
def fig12_percentile_maps(stack, profile):
    ext = get_extent(profile)
    p50 = np.nanpercentile(stack, 50, axis=0)
    p95 = np.nanpercentile(stack, 95, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(p50 > 0, p95 / p50, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, data, cmap, label, title in [
        (axes[0], p50, "YlOrRd", "MJ·mm·ha⁻¹·h⁻¹·yr⁻¹", "Median R (P50)"),
        (axes[1], p95, "YlOrRd", "MJ·mm·ha⁻¹·h⁻¹·yr⁻¹", "95th Percentile R (P95)"),
        (axes[2], ratio, "plasma", "P95/P50", "P95/P50 Ratio\n(positive skewness measure)"),
    ]:
        vn = np.nanpercentile(data, 2); vx = np.nanpercentile(data, 98)
        im = ax.imshow(data, extent=ext, cmap=cmap,
                       vmin=vn, vmax=vx, interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Longitude, °E"); ax.set_ylabel("Latitude, °N")
        ax.grid(alpha=0.2, lw=0.5)
        v = data[np.isfinite(data)]
        ax.text(0.02, 0.02, f"Mean={np.mean(v):.0f}", transform=ax.transAxes,
                fontsize=9, va="bottom",
                bbox=dict(fc="white", alpha=0.85, boxstyle="round,pad=0.3"))

    save(fig, "fig12_percentile_maps.png")


# ===================================================================
# FIG 13 — Literature comparison
# ===================================================================
def fig13_lit_comparison(mean_r082):
    LIT = [
        ("Tatarstan\n(Larionov 1993)",           115,  70,  160, "♦"),
        ("European Russia, zone B\n(Panov et al. 2020)", 160, 100, 220, "♦"),
        ("This study (k=0.082)",                 mean_r082, None, None, "★"),
        ("Eastern Europe (mean)\n(Ballabio et al. 2017)", 300, 150, 500, "○"),
        ("Central Europe\n(Panagos et al. 2015)",        500, 300, 800, "○"),
        ("Western Europe\n(Panagos et al. 2015)",        700, 300,1500, "○"),
        ("Mediterranean\n(Panagos et al. 2015)",        1200, 500,3000, "○"),
        ("NW China\n(Xu et al. 2022)",                   100,  40, 200, "△"),
        ("Korea (Kim et al. 2021)",             2500,1800,3500, "△"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    colors_map = {"♦": "#d73027", "★": "#1a9641", "○": "#4575b4", "△": "#fdae61"}

    for i, (lbl, mean_, lo, hi, mk) in enumerate(LIT):
        clr = colors_map[mk]
        if lo is not None:
            xerr = [[mean_ - lo], [hi - mean_]]
            ax.barh(i, mean_, xerr=xerr, height=0.55,
                    color=clr, alpha=0.75, edgecolor="gray", lw=0.5,
                    error_kw=dict(ecolor="gray", lw=1.5, capsize=5))
        else:
            ax.barh(i, mean_, height=0.55, color=clr, alpha=0.9,
                    edgecolor="darkgreen", lw=1.5)
            ax.text(mean_ + 20, i, f"→ {mean_:.0f}", va="center",
                    fontsize=10, fontweight="bold", color="#1a9641")

    ax.set_yticks(range(len(LIT)))
    ax.set_yticklabels([l[0] for l in LIT], fontsize=10)
    ax.set_xlabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹", fontsize=12)
    ax.set_title("R-Factor Comparison with Global Published Data\n"
                 "This study highlighted in green",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="x"); ax.set_xlim(0, 4200)

    legend_elem = [
        Patch(color="#d73027", alpha=0.8, label="Russian data"),
        Patch(color="#1a9641", alpha=0.9, label="This study"),
        Patch(color="#4575b4", alpha=0.8, label="European data"),
        Patch(color="#fdae61", alpha=0.8, label="Asian data"),
    ]
    ax.legend(handles=legend_elem, fontsize=9, loc="lower right")
    save(fig, "fig13_lit_comparison.png")


# ===================================================================
# FIG 14 — Summary table
# ===================================================================
def fig14_summary_table(means):
    years = np.array(YEARS)
    sl, ic, rv, pv, _ = sp_stats.linregress(years.astype(float), means)

    table_data = [
        ["Parameter", "Value", "Units"],
        ["Data source", "IMERG V07 Final", "—"],
        ["Period", "2001–2024", "24 years"],
        ["Spatial resolution", "0.1° (~11 km)", "—"],
        ["Kinetic energy formula", "e=0.29(1−0.72e^{−0.082i})", "(Foster 2003)"],
        ["Domain mean R", f"{np.mean(means):.1f}", "MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"],
        ["Domain median R", f"{np.median(means):.1f}", "—"],
        ["Minimum (year)", f"{np.min(means):.1f} ({YEARS[np.argmin(means)]})", "—"],
        ["Maximum (year)", f"{np.max(means):.1f} ({YEARS[np.argmax(means)]})", "—"],
        ["Interannual CV", f"{np.std(means)/np.mean(means)*100:.1f}%", "—"],
        ["OLS trend", f"{sl:+.2f} units/yr", f"p={pv:.3f}"],
        ["Significant trend area (p<0.05)", "< 5% of domain", "—"],
        ["Spatial range (P5–P95)", "~70–430", "MJ·mm·ha⁻¹·h⁻¹·yr⁻¹"],
        ["R082/R05 ratio", "1.166 ± 0.003", "—"],
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.axis("off")
    table = ax.table(
        cellText=[r[1:] for r in table_data[1:]],
        colLabels=table_data[0][1:],
        rowLabels=[r[0] for r in table_data[1:]],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.7)

    for j in range(len(table_data[0]) - 1):
        table[0, j].set_facecolor("#2c5f8a")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(table_data) - 1):
        for j in range(len(table_data[0]) - 1):
            if i % 2 == 0: table[i+1, j].set_facecolor("#f0f4f8")

    ax.set_title("Table 1. Summary R-Factor Characteristics (RUSLE)\n"
                 "IMERG V07 (calibrated), 2001–2024, k = 0.082",
                 fontsize=12, fontweight="bold", pad=12)
    save(fig, "fig14_summary_table.png")


# ===================================================================
# FIG 15 — Bootstrap uncertainty
# ===================================================================
def fig15_bootstrap(means, stack):
    from scipy.stats import gaussian_kde

    mean_R = np.mean(means)
    # Bootstrap
    rng = np.random.default_rng(42)
    n = len(means)
    boot_means = np.array([rng.choice(means, n, replace=True).mean() for _ in range(50000)])
    ci = 0.90
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)

    # Spatial bootstrap std (5k)
    rng2 = np.random.default_rng(42)
    sums = np.zeros(stack.shape[1:])
    sq   = np.zeros(stack.shape[1:])
    for _ in range(5000):
        idx = rng2.integers(0, n, n)
        s = np.nanmean(stack[idx], axis=0)
        sums += s; sq += s**2
    mean_boot_sp = sums / 5000
    std_spatial = np.sqrt(sq / 5000 - mean_boot_sp**2)

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.hist(boot_means, bins=80, color=BLUE, alpha=0.7, edgecolor="none", density=True)
    ax1.axvline(mean_R, color="black", lw=2, label=f"Observed mean = {mean_R:.0f}")
    ax1.axvline(lo, color=RED, lw=1.5, ls="--", label=f"90% CI: [{lo:.0f}, {hi:.0f}]")
    ax1.axvline(hi, color=RED, lw=1.5, ls="--")
    kde = gaussian_kde(boot_means)
    xv = np.linspace(boot_means.min(), boot_means.max(), 300)
    ax1.plot(xv, kde(xv), color="navy", lw=1.5)
    ax1.set_xlabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax1.set_ylabel("Density")
    ax1.set_title(f"(a) Bootstrap CI of Mean R\nn_boot = 50,000, CI = 90%", fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(YEARS, means, color=plt.cm.YlOrRd(np.interp(means, [means.min(), means.max()], [0.25, 0.92])),
            edgecolor="gray", lw=0.4, alpha=0.85)
    ax2.axhline(mean_R, color="black", lw=1.5, ls="--", label=f"Mean = {mean_R:.0f}")
    ax2.fill_between(YEARS, lo, hi, alpha=0.15, color=BLUE, label="90% bootstrap CI")
    ax2.set_xlabel("Year"); ax2.set_ylabel("R-factor, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax2.set_title(f"(b) Interannual Dynamics\nCI = ±{(hi-lo)/2:.0f} ({(hi-lo)/2/mean_R*100:.1f}% of mean)",
                  fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.25)
    ax2.set_xticks([y for y in YEARS if y % 4 == 1])
    ax2.set_xticklabels([str(y) for y in YEARS if y % 4 == 1], rotation=45)

    p = os.path.join(DIR_K082, "R_imerg_2001.tif")
    with rasterio.open(p) as ds:
        t = ds.transform
        ext = [t.c, t.c + t.a * ds.width, t.f + t.e * ds.height, t.f]

    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(std_spatial, extent=ext, cmap="Oranges",
                    vmin=0, vmax=np.nanpercentile(std_spatial, 98),
                    interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax3, shrink=0.85, pad=0.02,
                 label="Bootstrap σ, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    ax3.set_title("(c) Spatial Sampling\nUncertainty (bootstrap σ)", fontweight="bold")
    ax3.set_xlabel("Longitude, °E"); ax3.set_ylabel("Latitude, °N")
    ax3.grid(alpha=0.2, lw=0.5)

    save(fig, "fig15_uncertainty_bootstrap.png")
    return (hi - lo) / 2   # sigma_samp


# ===================================================================
# FIG 16 — Uncertainty budget
# ===================================================================
def fig16_budget(u_sampling, u_calib, u_param, mean_R):
    labels = [
        "Climatic\n(Bootstrap, 90% CI / 2)",
        "Calibration\n(PBIAS median, α=1.5)",
        "Parametric\n(k=0.05 vs 0.082)",
    ]
    vals_abs = [u_sampling, u_calib, u_param]
    combined = np.sqrt(sum(v**2 for v in vals_abs))
    colors = [BLUE, RED, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    bars = axes[0].barh(labels, vals_abs, color=colors, alpha=0.8, edgecolor="gray", lw=0.5)
    axes[0].axvline(combined, color="black", lw=2, ls="--",
                    label=f"RSS total = {combined:.0f} MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    for bar, v in zip(bars, vals_abs):
        axes[0].text(v + 1, bar.get_y() + bar.get_height()/2,
                     f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Uncertainty, MJ·mm·ha⁻¹·h⁻¹·yr⁻¹")
    axes[0].set_title(f"(a) R-Factor Uncertainty Budget\nMean R = {mean_R:.0f}", fontweight="bold")
    axes[0].legend(fontsize=10); axes[0].grid(axis="x", alpha=0.3)

    pie_vals = [v**2 for v in vals_abs]
    wedges, texts, auto = axes[1].pie(
        pie_vals, labels=labels, autopct=lambda p: f"{p:.1f}%",
        colors=colors, startangle=90, pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", lw=1.5),
    )
    for t in auto: t.set_fontsize(11); t.set_fontweight("bold")
    axes[1].set_title("(b) Variance Contributions\n(quadrature sum σ²)", fontweight="bold")

    save(fig, "fig16_uncertainty_budget.png")


# ===================================================================
# FIG 17 — Spatial variogram
# ===================================================================
def fig17_variogram():
    with rasterio.open(MEAN_TIF) as src:
        data = src.read(1)
        transform = src.transform
        nodata = src.nodata or -9999

    rows, cols = np.where((data != nodata) & (data >= 0) & (~np.isnan(data)))
    values = data[rows, cols]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    coords = np.column_stack((xs, ys))

    if len(coords) > 2000:
        np.random.seed(42)
        idx = np.random.choice(len(coords), 2000, replace=False)
        coords = coords[idx]; values = values[idx]

    distances = pdist(coords)
    sq_diff = 0.5 * pdist(values.reshape(-1, 1), metric='sqeuclidean')

    max_dist = np.max(distances) / 2
    bins = np.linspace(0, max_dist, 21)
    lag_centers, semivariance = [], []
    for i in range(20):
        mask = (distances >= bins[i]) & (distances < bins[i+1])
        if np.any(mask):
            lag_centers.append((bins[i] + bins[i+1]) / 2)
            semivariance.append(np.mean(sq_diff[mask]))

    plt.figure(figsize=(9, 6))
    plt.plot(lag_centers, semivariance, 'bo-', linewidth=2, markersize=8)
    plt.title('Empirical Variogram of Long-Term Mean R-Factor\n(Spatial Coherence Assessment)')
    plt.xlabel('Lag Distance, degrees (°)')
    plt.ylabel('Semivariance')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.text(0.05, 0.95,
             "Increasing semivariance with distance confirms\n"
             "spatial autocorrelation (coherence) of pixels.\n"
             "The 0.1° pixelization reflects the physical\n"
             "geometry of precipitation, not random noise.",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig17_spatial_variogram.png")
    plt.savefig(path, dpi=DPI)
    plt.close()
    print("  saved:", path)


# ===================================================================
# FIG 18 — Extreme years PBIAS
# ===================================================================
def fig18_extreme_years():
    csv_files = glob.glob(os.path.join(CALIB_DIR, "*_calib.csv"))
    annual_metrics = []

    for fp in csv_files:
        try:
            df = pd.read_csv(fp)
            for col in ['date', 'time', 'datetime']:
                if col in df.columns:
                    df['year'] = df[col].astype(str).str[:4].astype(int)
                    break
            else:
                continue

            obs_col = sat_col = None
            if 'P_station_mm' in df.columns and 'P_sat_mm' in df.columns:
                obs_col, sat_col = 'P_station_mm', 'P_sat_mm'
            else:
                for o, s, c in [('precip','imerg','imerg_qm'), ('obs','raw','corr'),
                                 ('p_obs','p_raw','p_corr'), ('obs','sat','corr')]:
                    if o in df.columns and s in df.columns:
                        obs_col, sat_col = o, s; break
            if not obs_col: continue

            annual = df.groupby('year')[[obs_col, sat_col]].sum()
            for index, row in annual.iterrows():
                obs, sat = row[obs_col], row[sat_col]
                if obs > 0:
                    annual_metrics.append({'year': index, 'pbias': 100*(sat-obs)/obs})
        except: pass

    res_df = pd.DataFrame(annual_metrics)
    res_df = res_df[(res_df['year'] >= 2001) & (res_df['year'] <= 2024)]
    res_df = res_df[res_df['pbias'] > -99.0]
    median_pbias = res_df.groupby('year')['pbias'].median()

    plt.figure(figsize=(10, 6))
    plt.plot(median_pbias.index, median_pbias.values, 'b-o', label='Median PBIAS (%)')
    plt.axvline(x=2001, color='r', linestyle='--', alpha=0.7, label='Extreme year (2001)')
    plt.axvline(x=2007, color='m', linestyle='--', alpha=0.7, label='Extreme year (2007)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.title('IMERG Precipitation Percent Bias (PBIAS) by Year\nModel Stability Assessment in Anomalous Erosion Years')
    plt.xlabel('Year'); plt.ylabel('PBIAS (%)')
    plt.legend(); plt.grid(True, alpha=0.4)

    text_content = (
        f"PBIAS 2001: {median_pbias.get(2001, np.nan):.1f}%\n"
        f"PBIAS 2007: {median_pbias.get(2007, np.nan):.1f}%\n"
        f"Mean: {median_pbias.mean():.1f}%"
    )
    plt.text(0.05, 0.95, text_content, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig18_extreme_years_pbias.png")
    plt.savefig(path, dpi=DPI); plt.close()
    print("  saved:", path)


# ===================================================================
# FIG 19 — Structural break (Pettitt)
# ===================================================================
def fig19_structural_break():
    import pyhomogeneity as hg

    tif_files = sorted(glob.glob(os.path.join(DIR_K082, "*.tif")))
    records = []
    for tif in tif_files:
        year_str = os.path.basename(tif).split('_')[-1].replace('.tif', '')
        if not year_str.isdigit(): continue
        year = int(year_str)
        with rasterio.open(tif) as src:
            data = src.read(1)
            valid = data[data >= 0]
            if len(valid) > 0:
                records.append({'year': year, 'r_factor': np.mean(valid)})

    df = pd.DataFrame(records).sort_values('year').set_index('year')
    res = hg.pettitt_test(df['r_factor'])

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['r_factor'], marker='o', label='Domain-Mean R-Factor')

    cp = res.cp
    plt.axvline(x=df.index[cp], color='r', linestyle='--',
                label=f'Breakpoint (Pettitt): {df.index[cp]}')

    mean_before = df['r_factor'].iloc[:cp].mean()
    mean_after  = df['r_factor'].iloc[cp:].mean()
    plt.hlines(y=mean_before, xmin=df.index[0], xmax=df.index[cp]-1, color='g', linestyle='-',
               label=f'Mean (before {df.index[cp]}): {mean_before:.1f}')
    plt.hlines(y=mean_after, xmin=df.index[cp], xmax=df.index[-1], color='m', linestyle='-',
               label=f'Mean (from {df.index[cp]}): {mean_after:.1f}')

    if res.p <= 0.05:
        plt.title('Statistically Significant Structural Shift Detected (Pettitt Test)')
    else:
        plt.title(f'No Statistically Significant Shift Detected (p={res.p:.3f})')

    plt.xlabel('Year')
    plt.ylabel('R-factor (MJ·mm·ha⁻¹·h⁻¹·yr⁻¹)')
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig19_structural_break_pettitt.png")
    plt.savefig(path, dpi=DPI); plt.close()
    print("  saved:", path)


# ===================================================================
# FIG 20 — Teleconnections
# ===================================================================
def fig20_teleconnections():
    import urllib.request

    def fetch_clean_df(url):
        data = []
        with urllib.request.urlopen(url) as response:
            lines = response.read().decode('utf-8').splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 13 and parts[0].isdigit() and int(parts[0]) > 1900:
                    year = int(parts[0])
                    vals = [float(p) if float(p) != -99.90 else np.nan for p in parts[1:13]]
                    data.append([year] + vals)
        return pd.DataFrame(data)

    nao_df = fetch_clean_df("https://psl.noaa.gov/data/correlation/nao.data")
    scand_df = fetch_clean_df("https://psl.noaa.gov/data/correlation/scand.data")

    columns = ['Year'] + list(range(1, 13))
    nao_df.columns = columns; scand_df.columns = columns

    r_df = pd.read_csv(RFACTOR_CSV)

    nao_summer  = nao_df.set_index('Year').loc[2001:2024, 5:9].mean(axis=1)
    scand_summer = scand_df.set_index('Year').loc[2001:2024, 5:9].mean(axis=1)
    nao_annual  = nao_df.set_index('Year').loc[2001:2024, 1:12].mean(axis=1)
    scand_annual = scand_df.set_index('Year').loc[2001:2024, 1:12].mean(axis=1)

    analysis_df = r_df.set_index('year').copy()
    analysis_df['NAO_Summer'] = nao_summer
    analysis_df['SCAND_Summer'] = scand_summer
    analysis_df['NAO_Annual'] = nao_annual
    analysis_df['SCAND_Annual'] = scand_annual
    analysis_df = analysis_df.dropna()

    vars_to_test = ['NAO_Summer', 'SCAND_Summer', 'NAO_Annual', 'SCAND_Annual']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, var in enumerate(vars_to_test):
        r_p, p_p = sp_stats.pearsonr(analysis_df[var], analysis_df['r_factor'])
        r_s, p_s = sp_stats.spearmanr(analysis_df[var], analysis_df['r_factor'])

        ax = axes[i]
        ax.scatter(analysis_df[var], analysis_df['r_factor'], alpha=0.7)
        m, b = np.polyfit(analysis_df[var], analysis_df['r_factor'], 1)
        ax.plot(analysis_df[var], m * analysis_df[var] + b, color='red', alpha=0.5)
        ax.set_title(f'R-factor vs. {var}')
        ax.set_xlabel(f'{var} Index'); ax.set_ylabel('R-factor')
        ax.grid(alpha=0.3)
        textstr = '\n'.join((
            f'Pearson: $r$={r_p:.2f} ($p$={p_p:.3f})',
            f'Spearman: $r_s$={r_s:.2f} ($p$={p_s:.3f})'))
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig20_teleconnections_correlation.png")
    plt.savefig(path, dpi=DPI); plt.close()
    print("  saved:", path)


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("="*60)
    print("GENERATING ENGLISH FIGURES (300 DPI)")
    print("="*60)

    print("\nLoading k=0.082 raster stack...")
    stack, profile = load_annual_k082()
    means = domain_stat(stack, np.nanmean)
    mean_R = np.mean(means)
    print(f"Domain mean R = {mean_R:.1f}")

    print("\n[1/12] fig07: annual small multiples")
    fig07_small_multiples(stack, profile)

    print("[2/12] fig08: mean map + PDF + CDF")
    fig08_mean_pdf(stack, profile)

    print("[3/12] fig09: temporal dynamics")
    fig09_temporal(stack)

    print("[4/12] fig10: CV + trend maps")
    fig10_cv_trend(stack, profile)

    print("[5/12] fig11: decadal comparison")
    d1m, d2m, d3m = fig11_decadal(stack, profile)

    print("[6/12] fig12: percentile maps")
    fig12_percentile_maps(stack, profile)

    print("[7/12] fig13: literature comparison")
    fig13_lit_comparison(mean_R)

    print("[8/12] fig14: summary table")
    fig14_summary_table(means)

    print("[9/12] fig15: bootstrap uncertainty")
    sigma_samp = fig15_bootstrap(means, stack)

    print("[10/12] fig16: uncertainty budget")
    # Hardcoded calibration/param uncertainty (same values as original)
    sigma_calib = 23.5   # from PBIAS analysis
    sigma_param = 0.9    # from k ratio analysis
    fig16_budget(sigma_samp, sigma_calib, sigma_param, mean_R)

    print("[11/12] fig17: spatial variogram")
    fig17_variogram()

    print("[12/12] fig18: extreme years PBIAS")
    fig18_extreme_years()

    print("\n[EXTRA 1/2] fig19: structural break (Pettitt)")
    fig19_structural_break()

    print("[EXTRA 2/2] fig20: teleconnections")
    fig20_teleconnections()

    print("\n" + "="*60)
    print(f"DONE. All {len(os.listdir(FIG_DIR))} figures saved to {FIG_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
