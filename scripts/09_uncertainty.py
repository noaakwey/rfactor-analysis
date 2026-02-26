# -*- coding: utf-8 -*-
"""
09_uncertainty.py
Three-component uncertainty analysis for R-factor (k=0.082):
  1. Sampling/climate uncertainty — bootstrap on 24 annual rasters
  2. Calibration uncertainty   — analytical PBIAS propagation from station data
  3. Parametric uncertainty    — k coefficient (0.05 vs 0.082, already known)

Outputs:
  docs/figures/fig15_uncertainty_bootstrap.png
  docs/figures/fig16_uncertainty_budget.png
  output/tables/uncertainty_summary.csv
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats
import pandas as pd

REPO     = r"D:\Cache\Yandex.Disk\РАЗРАБОТКА\code\rfactor-analysis"
FIG_DIR  = os.path.join(REPO, "docs", "figures")
TAB_DIR  = os.path.join(REPO, "output", "tables")

DIR_K082 = r"D:\Cache\Yandex.Disk\РНФ25-28\Осадки\rfactor_imerg_k082\annual"
PBIAS_CSV = r"D:\Cache\Yandex.Disk\РАЗРАБОТКА\code\imerg2meteo_calib\docs\station_annual_pbias_summary_imerg.csv"
COMPARE_CSV = os.path.join(REPO, "output", "tables", "compare_k05_k082.csv")

YEARS = list(range(2001, 2025))
DPI   = 220
BLUE  = "#2166ac"; RED = "#d6604d"; GREEN = "#1a9641"

# ------------------------------------------------------------------ #
# LOAD DATA
# ------------------------------------------------------------------ #
import rasterio

def load_stack():
    stack = []
    for y in YEARS:
        p = os.path.join(DIR_K082, f"R_imerg_{y}.tif")
        with rasterio.open(p) as ds:
            band = ds.read(1).astype(np.float64)
        band[band == 0] = np.nan
        stack.append(band)
    return np.array(stack)

def domain_means(stack):
    return np.array([np.nanmean(stack[i]) for i in range(len(stack))])


# ------------------------------------------------------------------ #
# COMPONENT 1:  Sampling / climate uncertainty — bootstrap
# ------------------------------------------------------------------ #
def bootstrap_uncertainty(means, n_boot=50000, ci=0.90):
    """Bootstrap CI on multi-year mean from 24 annual values."""
    rng = np.random.default_rng(42)
    n   = len(means)
    boot_means = np.array([rng.choice(means, n, replace=True).mean()
                            for _ in range(n_boot)])
    lo = np.percentile(boot_means, (1-ci)/2 * 100)
    hi = np.percentile(boot_means, (1+ci)/2 * 100)
    return boot_means, lo, hi


def bootstrap_spatial(stack, n_boot=5000):
    """Pixel-wise bootstrap: std of bootstrapped pixel means."""
    rng  = np.random.default_rng(42)
    n    = stack.shape[0]
    sums = np.zeros(stack.shape[1:])
    sq   = np.zeros(stack.shape[1:])
    for _ in range(n_boot):
        idx  = rng.integers(0, n, n)
        s    = np.nanmean(stack[idx], axis=0)
        sums += s
        sq   += s**2
    mean_boot = sums / n_boot
    std_boot  = np.sqrt(sq / n_boot - mean_boot**2)
    return std_boot


# ------------------------------------------------------------------ #
# COMPONENT 2:  Calibration uncertainty from PBIAS
# ------------------------------------------------------------------ #
def calibration_uncertainty(means_r082):
    """
    Analytical propagation of calibration PBIAS into R-factor uncertainty.

    R = sum_events(E * I30)
    If precipitation is scaled by factor (1 + epsilon), where epsilon ~ PBIAS/100:
      - E scales approximately linearly with precip (sum of e(i)*p)
      - I30 scales approximately linearly with precip
      => R scales ~ (1+epsilon)^alpha, where alpha in [1, 2]
    
    Conservative (upper bound): alpha = 1.5 (geometric mean of linear and quadratic)
    The uncertainty in R from calibration PBIAS:
      sigma_R_calib / R = alpha * sigma_PBIAS / 100
    """
    df = pd.read_csv(PBIAS_CSV)
    # Use corrected abs median PBIAS across stations as uncertainty
    pbias_vals = df["annual_pbias_corr_abs_med"].values
    # Filter out extreme outliers (stations with |PBIAS| > 50% are edge cases)
    pbias_vals = pbias_vals[pbias_vals < 50]
    
    pbias_median = np.median(pbias_vals)
    pbias_p84    = np.percentile(pbias_vals, 84)   # 1-sigma equivalent for lognormal
    pbias_p95    = np.percentile(pbias_vals, 95)
    
    alpha = 1.5   # conservative exponent
    sigma_rel_median = alpha * pbias_median / 100
    sigma_rel_p84    = alpha * pbias_p84    / 100
    sigma_rel_p95    = alpha * pbias_p95    / 100
    
    mean_R = np.mean(means_r082)
    sigma_calib_median = sigma_rel_median * mean_R
    sigma_calib_p84    = sigma_rel_p84    * mean_R
    
    return {
        "pbias_median": pbias_median,
        "pbias_p84":    pbias_p84,
        "pbias_p95":    pbias_p95,
        "sigma_rel_median": sigma_rel_median * 100,   # in %
        "sigma_rel_p84":    sigma_rel_p84    * 100,
        "sigma_calib_median": sigma_calib_median,
        "sigma_calib_p84":    sigma_calib_p84,
        "pbias_vals":   pbias_vals,
    }


# ------------------------------------------------------------------ #
# COMPONENT 3:  Parametric (k) uncertainty
# ------------------------------------------------------------------ #
def parametric_uncertainty(means_r082):
    df  = pd.read_csv(COMPARE_CSV)
    df  = df[df["year"] != "MEAN"]
    ratio = df["ratio"].astype(float)
    mean_R = np.mean(means_r082)
    # sigma_k = std of ratio * mean_R / mean_ratio (relative contribution)
    sigma_ratio = ratio.std()
    mean_ratio  = ratio.mean()
    sigma_param = sigma_ratio / mean_ratio * mean_R
    return {
        "ratio_mean": mean_ratio,
        "ratio_std":  sigma_ratio,
        "sigma_param": sigma_param,
        "sigma_rel":   sigma_ratio / mean_ratio * 100,
    }


# ------------------------------------------------------------------ #
# FIG 15 — Bootstrap sampling uncertainty
# ------------------------------------------------------------------ #
def fig15_bootstrap(means, boot_means, lo, hi, ci, std_spatial):
    fig = plt.figure(figsize=(15, 5), constrained_layout=True)
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    mean_R = np.mean(means)

    # --- (a) Bootstrap distribution of domain mean ---
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(boot_means, bins=80, color=BLUE, alpha=0.7, edgecolor="none", density=True)
    ax1.axvline(mean_R, color="black", lw=2, label=f"Observed mean = {mean_R:.0f}")
    ax1.axvline(lo, color=RED, lw=1.5, ls="--", label=f"{int(ci*100)}% CI: [{lo:.0f}, {hi:.0f}]")
    ax1.axvline(hi, color=RED, lw=1.5, ls="--")
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(boot_means)
    xv  = np.linspace(boot_means.min(), boot_means.max(), 300)
    ax1.plot(xv, kde(xv), color="navy", lw=1.5)
    ax1.set_xlabel("R-фактор, МДж·мм·га⁻¹·ч⁻¹·год⁻¹")
    ax1.set_ylabel("Плотность")
    ax1.set_title(f"(a) Bootstrap DI среднего R\nn_boot = 50 000, CI = {int(ci*100)}%",
                  fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- (b) Observed annual means + CI band ---
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(YEARS, means, color=plt.cm.YlOrRd(np.interp(means, [means.min(), means.max()], [0.25, 0.92])),
            edgecolor="gray", lw=0.4, alpha=0.85)
    ax2.axhline(mean_R, color="black", lw=1.5, ls="--", label=f"Среднее = {mean_R:.0f}")
    ax2.fill_between(YEARS, lo, hi, alpha=0.15, color=BLUE,
                     label=f"{int(ci*100)}% bootstrap CI")
    ax2.set_xlabel("Год")
    ax2.set_ylabel("R-фактор, МДж·мм·га⁻¹·ч⁻¹·год⁻¹")
    ax2.set_title(f"(b) Межгодовая динамика\nCI = ±{(hi-lo)/2:.0f} ({(hi-lo)/2/mean_R*100:.1f}% от среднего)",
                  fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)
    ax2.set_xticks([y for y in YEARS if y % 4 == 1])
    ax2.set_xticklabels([str(y) for y in YEARS if y % 4 == 1], rotation=45)

    # --- (c) Spatial map of bootstrap std ---
    import rasterio as rio
    p = os.path.join(DIR_K082, "R_imerg_2001.tif")
    with rio.open(p) as ds:
        t = ds.transform
        ext = [t.c, t.c + t.a * ds.width, t.f + t.e * ds.height, t.f]

    ax3 = fig.add_subplot(gs[2])
    im  = ax3.imshow(std_spatial, extent=ext, cmap="Oranges",
                     vmin=0, vmax=np.nanpercentile(std_spatial, 98),
                     interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax3, shrink=0.85, pad=0.02,
                 label="Bootstrap σ, МДж·мм·га⁻¹·ч⁻¹·год⁻¹")
    ax3.set_title(f"(c) Пространственная выборочная\nнеопределённость (bootstrap σ)",
                  fontweight="bold")
    ax3.set_xlabel("Долгота, °E"); ax3.set_ylabel("Широта, °N")
    ax3.grid(alpha=0.2, lw=0.5)

    path = os.path.join(FIG_DIR, "fig15_uncertainty_bootstrap.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved:", path)


# ------------------------------------------------------------------ #
# FIG 16 — Uncertainty budget
# ------------------------------------------------------------------ #
def fig16_budget(u_sampling, u_calib, u_param, mean_R):
    labels = [
        "Климатическая\n(Bootstrap, 90% CI / 2)",
        "Калибровочная\n(PBIAS median, alpha=1.5)",
        "Параметрическая\n(k=0.05 vs 0.082)",
    ]
    vals_abs  = [u_sampling, u_calib, u_param]
    vals_rel  = [v / mean_R * 100 for v in vals_abs]
    combined  = np.sqrt(sum(v**2 for v in vals_abs))  # quadrature sum

    colors = [BLUE, RED, GREEN]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Absolute
    bars = axes[0].barh(labels, vals_abs, color=colors, alpha=0.8, edgecolor="gray", lw=0.5)
    axes[0].axvline(combined, color="black", lw=2, ls="--",
                    label=f"RSS total = {combined:.0f} МДж·мм·га⁻¹·ч⁻¹·год⁻¹")
    for bar, v in zip(bars, vals_abs):
        axes[0].text(v + 1, bar.get_y() + bar.get_height()/2,
                     f"{v:.1f}", va="center", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Неопределённость, МДж·мм·га⁻¹·ч⁻¹·год⁻¹")
    axes[0].set_title(f"(a) Бюджет неопределённости R-фактора\nСреднее R = {mean_R:.0f}",
                      fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="x", alpha=0.3)

    # Relative (pie)
    pie_vals  = [v**2 for v in vals_abs]    # variance contributions
    pie_total = sum(pie_vals)
    wedges, texts, auto = axes[1].pie(
        pie_vals,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%",
        colors=colors,
        startangle=90,
        pctdistance=0.65,
        wedgeprops=dict(edgecolor="white", lw=1.5),
    )
    for t in auto:
        t.set_fontsize(11); t.set_fontweight("bold")
    axes[1].set_title(f"(b) Доли дисперсии\n(квадратурная сумма σ²)",
                      fontweight="bold")

    path = os.path.join(FIG_DIR, "fig16_uncertainty_budget.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved:", path)


# ------------------------------------------------------------------ #
# SAVE TABLE
# ------------------------------------------------------------------ #
def save_table(mean_R, lo, hi, sigma_samp, u_calib_dict, u_param_dict):
    combined = np.sqrt(sigma_samp**2 + u_calib_dict["sigma_calib_median"]**2 + u_param_dict["sigma_param"]**2)
    rows = [
        ["Базовое значение R (среднее)", f"{mean_R:.1f}", "МДж·мм·га⁻¹·ч⁻¹·год⁻¹"],
        ["90% bootstrap CI (нижняя граница)", f"{lo:.1f}", "—"],
        ["90% bootstrap CI (верхняя граница)", f"{hi:.1f}", "—"],
        ["Полуширина bootstrap CI", f"{(hi-lo)/2:.1f} ({(hi-lo)/2/mean_R*100:.1f}%)", "—"],
        ["PBIAS станций (медиана корр.)", f"{u_calib_dict['pbias_median']:.1f}%", "по 201 станции"],
        ["σ_калибр (медиана, alpha=1.5)", f"{u_calib_dict['sigma_calib_median']:.1f} ({u_calib_dict['sigma_rel_median']:.1f}%)", "—"],
        ["σ_калибр (P84, alpha=1.5)", f"{u_calib_dict['sigma_calib_p84']:.1f} ({u_calib_dict['sigma_rel_p84']:.1f}%)", "—"],
        ["σ_k (std ratio 0.082/0.05)", f"{u_param_dict['sigma_param']:.1f} ({u_param_dict['sigma_rel']:.2f}%)", "—"],
        ["RSS неопределённость (1σ)", f"{combined:.1f} ({combined/mean_R*100:.1f}%)", "квадратурная сумма"],
    ]
    df = pd.DataFrame(rows, columns=["Показатель", "Значение", "Примечание"])
    path = os.path.join(TAB_DIR, "uncertainty_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("  CSV:", path)
    return combined


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
def main():
    print("Loading k=0.082 stack...")
    stack = load_stack()
    means = domain_means(stack)
    mean_R = np.mean(means)
    print(f"Domain mean R = {mean_R:.1f}")

    print("\n[1/4] Bootstrap sampling uncertainty (50k draws)...")
    boot_means, lo, hi = bootstrap_uncertainty(means, n_boot=50000, ci=0.90)
    sigma_samp = (hi - lo) / 2
    print(f"  90% CI: [{lo:.1f}, {hi:.1f}]  half-width = {sigma_samp:.1f} ({sigma_samp/mean_R*100:.1f}%)")

    print("\n[2/4] Spatial bootstrap std (5k draws)...")
    std_spatial = bootstrap_spatial(stack, n_boot=5000)
    print("  Mean spatial std = %.1f" % np.nanmean(std_spatial))

    print("\n[3/4] Calibration uncertainty from PBIAS...")
    u_calib = calibration_uncertainty(means)
    print("  PBIAS median = %.1f%%, sigma_R = %.1f (%.1f%%)" % (
        u_calib['pbias_median'], u_calib['sigma_calib_median'], u_calib['sigma_rel_median']))

    print("\n[4/4] Parametric uncertainty (k)...")
    u_param = parametric_uncertainty(means)
    print("  ratio std = %.4f, sigma_R = %.1f (%.2f%%)" % (
        u_param['ratio_std'], u_param['sigma_param'], u_param['sigma_rel']))

    print("\nFig 15: bootstrap figure...")
    fig15_bootstrap(means, boot_means, lo, hi, 0.90, std_spatial)

    print("Fig 16: uncertainty budget...")
    fig16_budget(sigma_samp, u_calib["sigma_calib_median"], u_param["sigma_param"], mean_R)

    print("\nSaving table...")
    combined = save_table(mean_R, lo, hi, sigma_samp, u_calib, u_param)

    print("\n=== UNCERTAINTY SUMMARY ===")
    print("  R (mean)             = %.1f" % mean_R)
    print("  sigma_sampling (90%%) = %.1f (%.1f%%)" % (sigma_samp, sigma_samp/mean_R*100))
    print("  sigma_calib (median) = %.1f (%.1f%%)" % (u_calib['sigma_calib_median'], u_calib['sigma_rel_median']))
    print("  sigma_param (k)      = %.1f (%.2f%%)" % (u_param['sigma_param'], u_param['sigma_rel']))
    print("  sigma_RSS (total 1s) = %.1f (%.1f%%)" % (combined, combined/mean_R*100))


if __name__ == "__main__":
    main()
