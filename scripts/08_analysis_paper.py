# -*- coding: utf-8 -*-
"""
08_analysis_paper.py
Generates all figures for the comprehensive R-factor analysis paper (k=0.082).

Output: docs/figures/  (fig07_* through fig14_*)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

REPO    = os.path.dirname(os.path.dirname(__file__))
FIG_DIR = os.path.join(REPO, "docs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI    = 220
YEARS  = list(range(2001, 2025))
NODATA = 0.0

DIR_K082 = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\annual"
FNAME    = "R_imerg_{year}.tif"

BLUE  = "#2166ac"
RED   = "#d6604d"
GREEN = "#4dac26"
GRAY  = "#636363"

# ------------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------------
import rasterio

def load_annual_k082():
    import warnings
    stack = []
    profile = None
    for y in YEARS:
        p = os.path.join(DIR_K082, FNAME.format(year=y))
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
# FIG 07 — Annual small multiples (k=0.082)
# ===================================================================
def fig07_small_multiples(stack, profile):
    ext    = get_extent(profile)
    means  = domain_stat(stack, np.nanmean)
    vmin   = np.nanpercentile(stack, 2)
    vmax   = np.nanpercentile(stack, 98)

    ncols  = 6
    nrows  = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 12),
                              constrained_layout=True)

    cmap = plt.cm.YlOrRd
    for i, y in enumerate(YEARS):
        ax  = axes[i // ncols, i % ncols]
        im  = ax.imshow(stack[i], extent=ext, cmap=cmap,
                        vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(f"{y}\n{means[i]:.0f}", fontsize=8, fontweight="bold",
                     color="darkred" if means[i] > np.nanmean(means) else "navy")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)

    fig.colorbar(im, ax=axes, shrink=0.4, pad=0.01,
                 label="R (k=0.082), МДж·мм·га⁻¹·ч⁻¹·год⁻¹",
                 orientation="horizontal", aspect=40)
    fig.suptitle("Годовые карты R-фактора (k = 0.082), 2001–2024\n"
                 "IMERG V07 (калиброванный), 0.1°",
                 fontsize=14, fontweight="bold", y=1.01)
    save(fig, "fig07_annual_multiples.png")


# ===================================================================
# FIG 08 — Mean map + PDF + CDF (k=0.082)
# ===================================================================
def fig08_mean_pdf(stack, profile):
    with np.errstate(invalid="ignore"):
        mean_r = np.nanmean(stack, axis=0)

    ext = get_extent(profile)
    vals = mean_r[np.isfinite(mean_r)]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True,
                              gridspec_kw={"width_ratios": [2, 1, 1]})

    # --- Карта ---
    ax = axes[0]
    im = ax.imshow(mean_r, extent=ext, cmap="YlOrRd",
                   vmin=np.nanpercentile(mean_r, 2),
                   vmax=np.nanpercentile(mean_r, 98),
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                 label="МДж·мм·га⁻¹·ч⁻¹·год⁻¹")
    ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
    ax.set_title("Среднемноголетний R-фактор (2001–2024)\n$k = 0.082$ (RUSLE2)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2, lw=0.5)
    ax.text(0.02, 0.02,
            f"Mean = {np.mean(vals):.0f}\nMedian = {np.median(vals):.0f}\n"
            f"P5–P95 = {np.percentile(vals, 5):.0f}–{np.percentile(vals, 95):.0f}",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(fc="white", alpha=0.85, boxstyle="round,pad=0.3"))

    # --- PDF ---
    ax2 = axes[1]
    ax2.hist(vals, bins=40, color=RED, alpha=0.75, edgecolor="none", density=True)
    ax2.axvline(np.mean(vals), color="black", lw=1.5, ls="--", label=f"Mean={np.mean(vals):.0f}")
    ax2.axvline(np.median(vals), color=BLUE, lw=1.5, ls=":", label=f"Median={np.median(vals):.0f}")
    ax2.set_xlabel("R-фактор, МДж·мм·га⁻¹·ч⁻¹·год⁻¹"); ax2.set_ylabel("Плотность")
    ax2.set_title("Распределение значений\nсреднемноголетнего R", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # --- CDF ---
    ax3 = axes[2]
    sorted_v = np.sort(vals)
    cdf = np.arange(1, len(sorted_v)+1) / len(sorted_v)
    ax3.plot(sorted_v, cdf * 100, color=RED, lw=2)
    for p, ls in [(10, ":"), (25, "--"), (50, "-"), (75, "--"), (90, ":")]:
        v = np.percentile(sorted_v, p)
        ax3.axhline(p, color=GRAY, lw=0.8, ls=ls, alpha=0.6)
        ax3.axvline(v, color=GRAY, lw=0.8, ls=ls, alpha=0.6)
        ax3.text(v + 2, p + 1, f"P{p}={v:.0f}", fontsize=7, color=GRAY)
    ax3.set_xlabel("R-фактор, МДж·мм·га⁻¹·ч⁻¹·год⁻¹"); ax3.set_ylabel("CDF, %")
    ax3.set_title("Кумулятивная функция\nраспределения", fontsize=11, fontweight="bold")
    ax3.grid(alpha=0.3); ax3.set_ylim(0, 100)

    save(fig, "fig08_mean_pdf_cdf.png")
    return vals, mean_r


# ===================================================================
# FIG 09 — Temporal dynamics: bar+anomaly+cumulative
# ===================================================================
def fig09_temporal(stack):
    means   = domain_stat(stack, np.nanmean)
    medians = domain_stat(stack, np.nanmedian)
    p25     = domain_stat(stack, lambda x: np.percentile(x, 25))
    p75     = domain_stat(stack, lambda x: np.percentile(x, 75))
    p5      = domain_stat(stack, lambda x: np.percentile(x, 5))
    p95     = domain_stat(stack, lambda x: np.percentile(x, 95))

    years = np.array(YEARS, dtype=float)
    mu    = np.mean(means)
    sl, ic, r_, p_, _ = sp_stats.linregress(years, means)

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs  = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[3, 1.2, 1.2],
                             hspace=0.05)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])

    # Bar + envelopes
    colors = plt.cm.YlOrRd(np.interp(means, [means.min(), means.max()], [0.25, 0.92]))
    ax1.bar(YEARS, means, color=colors, edgecolor="gray", lw=0.4, alpha=0.88, zorder=3)
    ax1.fill_between(YEARS, p5, p95, alpha=0.08, color=BLUE, label="P5–P95")
    ax1.fill_between(YEARS, p25, p75, alpha=0.2, color=BLUE, label="P25–P75")
    ax1.plot(YEARS, medians, "o-", color=BLUE, ms=3, lw=1.2, alpha=0.7, label="Медиана")
    ax1.plot(years, sl * years + ic, "k--", lw=1.5,
             label=f"OLS: {sl:+.1f} ед./год (p={p_:.3f}, R²={r_**2:.3f})")
    ax1.axhline(mu, color=GRAY, lw=1.0, ls=":", alpha=0.7,
                label=f"Среднее: {mu:.0f}")

    # Annotate extremes
    imax = np.argmax(means); imin = np.argmin(means)
    for idx, clr, lbl in [(imax, "#a50026", "max"), (imin, "#313695", "min")]:
        ax1.annotate(f"{YEARS[idx]}: {means[idx]:.0f}",
                     xy=(YEARS[idx], means[idx]), xytext=(0, 18 if idx == imax else -28),
                     textcoords="offset points", ha="center", fontsize=9, fontweight="bold",
                     color=clr, arrowprops=dict(arrowstyle="-", color=clr, lw=0.7))

    ax1.set_ylabel("R-фактор, МДж·мм·га⁻¹·ч⁻¹·год⁻¹", fontsize=11)
    ax1.set_title("Межгодовая динамика R-фактора RUSLE (k = 0.082)\n"
                  "IMERG V07 (калиброванный), 2001–2024",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9, ncol=3, framealpha=0.9)
    ax1.grid(axis="y", alpha=0.25)
    ax1.set_xlim(2000.5, 2024.5)
    ax1.set_xticks([])

    # Anomaly bars
    anom = means - mu
    bc = [RED if a > 0 else BLUE for a in anom]
    ax2.bar(YEARS, anom, color=bc, edgecolor="gray", lw=0.3, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Аномалия R", fontsize=10)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_xlim(2000.5, 2024.5)
    ax2.set_xticks([])

    # Cumulative sum of anomalies
    cum = np.cumsum(anom)
    ax3.fill_between(YEARS, 0, cum, where=cum >= 0, color=RED, alpha=0.4)
    ax3.fill_between(YEARS, 0, cum, where=cum < 0, color=BLUE, alpha=0.4)
    ax3.plot(YEARS, cum, color="black", lw=1.5)
    ax3.axhline(0, color="black", lw=0.7)
    ax3.set_ylabel("Накопл. аномалия", fontsize=10)
    ax3.set_xlabel("Год", fontsize=11)
    ax3.grid(axis="y", alpha=0.25)
    ax3.set_xlim(2000.5, 2024.5)
    ax3.set_xticks(YEARS)
    ax3.set_xticklabels([str(y) if y % 2 == 1 else "" for y in YEARS], fontsize=8)

    save(fig, "fig09_temporal_dynamics.png")
    return means, mu, sl, p_


# ===================================================================
# FIG 10 — Spatial CV + pixel-wise trend
# ===================================================================
def fig10_cv_trend(stack, profile):
    ext  = get_extent(profile)
    mu   = np.nanmean(stack, axis=0)
    sig  = np.nanstd(stack, axis=0)
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
            if ok.sum() < 6:
                continue
            s, _, _, p, _ = sp_stats.linregress(years_arr[ok], ts[ok])
            slope_map[i, j] = s
            pval_map[i, j]  = p

    sig_mask = np.where(pval_map < 0.05, slope_map, np.nan)
    abs_max  = np.nanpercentile(np.abs(sig_mask[np.isfinite(sig_mask)]), 98) if np.any(np.isfinite(sig_mask)) else 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    # CV
    ax = axes[0]
    im = ax.imshow(cv, extent=ext, cmap="viridis",
                   vmin=np.nanpercentile(cv, 2), vmax=np.nanpercentile(cv, 98),
                   interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="CV, %")
    ax.set_title(f"Пространственный CV R-фактора (2001–2024)\nMean CV = {np.nanmean(cv):.0f}%",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
    ax.grid(alpha=0.2, lw=0.5)

    # Trend
    ax2 = axes[1]
    im2  = ax2.imshow(sig_mask, extent=ext, cmap="RdBu_r",
                      vmin=-abs_max, vmax=abs_max, interpolation="nearest", aspect="auto")
    # Gray insignificant background
    bg = np.where(np.isfinite(slope_map) & ~np.isfinite(sig_mask), 1.0, np.nan)
    ax2.imshow(bg, extent=ext, cmap="Greys", vmin=0, vmax=2,
               alpha=0.18, interpolation="nearest", aspect="auto")
    fig.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02,
                 label="Тренд R, МДж·мм·га⁻¹·ч⁻¹·год⁻¹/год")
    n_sig = int(np.sum(np.isfinite(sig_mask)))
    n_tot = int(np.sum(np.isfinite(slope_map)))
    ax2.set_title(f"Линейный тренд R-фактора (p < 0.05)\n{n_sig}/{n_tot} значимых пикселей ({n_sig/max(n_tot,1)*100:.0f}%)",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Долгота, °E"); ax2.set_ylabel("Широта, °N")
    ax2.grid(alpha=0.2, lw=0.5)

    save(fig, "fig10_cv_trend.png")
    return cv, slope_map, pval_map


# ===================================================================
# FIG 11 — Decadal comparison (2001–2008 vs 2009–2016 vs 2017–2024)
# ===================================================================
def fig11_decadal(stack, profile):
    ext = get_extent(profile)
    d1  = np.nanmean(stack[0:8],  axis=0)   # 2001–2008
    d2  = np.nanmean(stack[8:16], axis=0)   # 2009–2016
    d3  = np.nanmean(stack[16:],  axis=0)   # 2017–2024

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
        ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
        ax.grid(alpha=0.2, lw=0.5)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                     label="МДж·мм·га⁻¹·ч⁻¹·год⁻¹")

    # 4th: difference d3 - d1
    diff = d3 - d1
    amax = np.nanpercentile(np.abs(diff), 98)
    im4 = axes[3].imshow(diff, extent=ext, cmap="RdBu_r",
                          vmin=-amax, vmax=amax, interpolation="nearest", aspect="auto")
    axes[3].set_title(f"2017–2024 минус 2001–2008\nMean delta={np.nanmean(diff):.0f}",
                      fontsize=12, fontweight="bold")
    axes[3].set_xlabel("Долгота, °E"); axes[3].set_ylabel("Широта, °N")
    axes[3].grid(alpha=0.2, lw=0.5)
    fig.colorbar(im4, ax=axes[3], shrink=0.85, pad=0.02,
                 label="ΔR, МДж·мм·га⁻¹·ч⁻¹·год⁻¹")

    save(fig, "fig11_decadal.png")
    return np.nanmean(d1), np.nanmean(d2), np.nanmean(d3)


# ===================================================================
# FIG 12 — Percentile analysis: P50 vs P95 map + scatter
# ===================================================================
def fig12_percentile_maps(stack, profile):
    ext = get_extent(profile)
    p50 = np.nanpercentile(stack, 50, axis=0)
    p95 = np.nanpercentile(stack, 95, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(p50 > 0, p95 / p50, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for ax, data, cmap, label, title in [
        (axes[0], p50, "YlOrRd", "МДж·мм·га⁻¹·ч⁻¹·год⁻¹", "Медиана R (P50)"),
        (axes[1], p95, "YlOrRd", "МДж·мм·га⁻¹·ч⁻¹·год⁻¹", "95-й перцентиль R (P95)"),
        (axes[2], ratio, "plasma", "P95/P50", "Отношение P95/P50\n(мера правой асимметрии)"),
    ]:
        vn = np.nanpercentile(data, 2)
        vx = np.nanpercentile(data, 98)
        im = ax.imshow(data, extent=ext, cmap=cmap,
                       vmin=vn, vmax=vx, interpolation="nearest", aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label=label)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Долгота, °E"); ax.set_ylabel("Широта, °N")
        ax.grid(alpha=0.2, lw=0.5)
        v = data[np.isfinite(data)]
        ax.text(0.02, 0.02, f"Mean={np.mean(v):.0f}", transform=ax.transAxes,
                fontsize=9, va="bottom",
                bbox=dict(fc="white", alpha=0.85, boxstyle="round,pad=0.3"))

    save(fig, "fig12_percentile_maps.png")


# ===================================================================
# FIG 13 — World literature comparison (error bar chart)
# ===================================================================
def fig13_lit_comparison(mean_r082):
    LIT = [
        # label, mean, lo, hi, source, marker_style
        ("Татарстан\n(Ларионов 1993)",         115,  70,  160, "♦"),
        ("Евр. Россия, зона B\n(Панов et al. 2020)",  160, 100, 220, "♦"),
        ("Данное иссл. (k=0.082)",             mean_r082, None, None, "★"),
        ("Вост. Европа (ср.)\n(Ballabio et al. 2017)",  300, 150, 500, "○"),
        ("Центр. Европа\n(Panagos et al. 2015)",        500, 300, 800, "○"),
        ("Зап. Европа\n(Panagos et al. 2015)",          700, 300,1500, "○"),
        ("Средиземноморье\n(Panagos et al. 2015)",     1200, 500,3000, "○"),
        ("Сев.-Зап. Китай\n(Xu et al. 2022)",           100,  40, 200, "△"),
        ("Korea (KIM и др. 2021)",              2500,1800, 3500, "△"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    y_pos = np.arange(len(LIT))
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

    ax.set_yticks(y_pos)
    ax.set_yticklabels([l[0] for l in LIT], fontsize=10)
    ax.set_xlabel("R-фактор, МДж·мм·га⁻¹·ч⁻¹·год⁻¹", fontsize=12)
    ax.set_title("Сопоставление R-фактора с мировыми данными\n"
                 "Данное исследование выделено зелёным",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, 4200)

    # Legend
    from matplotlib.patches import Patch
    legend_elem = [
        Patch(color="#d73027", alpha=0.8, label="Российские данные"),
        Patch(color="#1a9641", alpha=0.9, label="Данное исследование"),
        Patch(color="#4575b4", alpha=0.8, label="Европейские данные"),
        Patch(color="#fdae61", alpha=0.8, label="Азиатские данные"),
    ]
    ax.legend(handles=legend_elem, fontsize=9, loc="lower right")

    save(fig, "fig13_lit_comparison.png")


# ===================================================================
# FIG 14 — Summary statistics table as figure
# ===================================================================
def fig14_summary_table(means):
    years = np.array(YEARS)
    sl, ic, rv, pv, _ = sp_stats.linregress(years.astype(float), means)

    rows = []
    for yr, m in zip(YEARS, means):
        rows.append((yr, f"{m:.1f}"))

    table_data = [
        ["Показатель", "Значение", "Единицы"],
        ["Приборная база", "IMERG V07 Final", "—"],
        ["Период", "2001–2024", "24 года"],
        ["Пространств. разрешение", "0.1° (~11 км)", "—"],
        ["Формула кинетической энергии", "e=0.29(1−0.72e^{−0.082i})", "(Foster 2003)"],
        ["Среднее R по домену", f"{np.mean(means):.1f}", "МДж·мм·га⁻¹·ч⁻¹·год⁻¹"],
        ["Медиана R по домену", f"{np.median(means):.1f}", "—"],
        ["Минимум (год)", f"{np.min(means):.1f} ({YEARS[np.argmin(means)]})", "—"],
        ["Максимум (год)", f"{np.max(means):.1f} ({YEARS[np.argmax(means)]})", "—"],
        ["Межгодовой CV", f"{np.std(means)/np.mean(means)*100:.1f}%", "—"],
        ["OLS тренд", f"{sl:+.2f} ед./год", f"p={pv:.3f}"],
        ["Доля значимого тренда (p<0.05)", "< 5% домена", "—"],
        ["Пространств. диапазон (P5–P95)", "~70–430", "МДж·мм·га⁻¹·ч⁻¹·год⁻¹"],
        ["Отношение R082/R05", "1.166 ± 0.003", "—"],
    ]

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    ax.axis("off")

    col_widths = [0.42, 0.33, 0.25]
    table = ax.table(
        cellText=[r[1:] for r in table_data[1:]],
        colLabels=table_data[0][1:],
        rowLabels=[r[0] for r in table_data[1:]],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.7)

    # Color header
    for j in range(len(table_data[0]) - 1):
        table[0, j].set_facecolor("#2c5f8a")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternating rows
    for i in range(len(table_data) - 1):
        for j in range(len(table_data[0]) - 1):
            if i % 2 == 0:
                table[i+1, j].set_facecolor("#f0f4f8")

    ax.set_title("Таблица 1. Сводные характеристики R-фактора RUSLE\nIMERG V07 (калиброванный), 2001–2024, k = 0.082",
                 fontsize=12, fontweight="bold", pad=12)

    save(fig, "fig14_summary_table.png")


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("Loading k=0.082 stack...")
    stack, profile = load_annual_k082()
    means = domain_stat(stack, np.nanmean)

    print("[1/8] fig07: annual small multiples")
    fig07_small_multiples(stack, profile)

    print("[2/8] fig08: mean map + PDF + CDF")
    vals, mean_r = fig08_mean_pdf(stack, profile)

    print("[3/8] fig09: temporal dynamics")
    means, mu, sl, pv = fig09_temporal(stack)

    print("[4/8] fig10: CV + trend maps")
    cv, slope_map, pval_map = fig10_cv_trend(stack, profile)

    print("[5/8] fig11: decadal comparison")
    d1m, d2m, d3m = fig11_decadal(stack, profile)
    print(f"  Decade means: {d1m:.1f} / {d2m:.1f} / {d3m:.1f}")

    print("[6/8] fig12: percentile maps")
    fig12_percentile_maps(stack, profile)

    print("[7/8] fig13: literature comparison")
    fig13_lit_comparison(np.mean(means))

    print("[8/8] fig14: summary table")
    fig14_summary_table(means)

    print("\nDone. All figures in docs/figures/")
    print(f"Domain mean R = {np.mean(means):.1f}")
    print(f"Decade 2001-08: {d1m:.1f}  |  2009-16: {d2m:.1f}  |  2017-24: {d3m:.1f}")
    return means, d1m, d2m, d3m


if __name__ == "__main__":
    main()
