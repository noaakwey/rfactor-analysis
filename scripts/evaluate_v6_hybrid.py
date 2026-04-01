from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.v6_hybrid import (  # noqa: E402
    apply_station_models_to_target_year,
    extract_pixel_series,
    fit_station_models_for_year,
    load_aws_series,
    load_biomet_series,
    load_calibration_tables,
)


RAW_RE = re.compile(r"IMERG_V07_P30min_mmh_(\d{4})_(Q\d)_permanent\.tif$", re.IGNORECASE)
ANNUAL_RE = re.compile(r"IMERG_V07_P30min_mmh_(\d{4})_.*\.tif$", re.IGNORECASE)
METHOD_ORDER = ["raw", "v1", "v6"]
METHOD_COLORS = {"raw": "#8f8f8f", "v1": "#2c7fb8", "v6": "#d95f02", "ground": "#1b9e77"}
SEASON_COLORS = {"DJF": "#5e3c99", "MAM": "#1b9e77", "JJA": "#d95f02", "SON": "#7570b3"}
SITE_MARKERS = {"aws310": "o", "biomet": "s"}
AWS310_COORDS = {"lon": 48.81, "lat": 55.84}
BIOMET_COORDS = {"lon": 49.2802, "lat": 55.2694}


def _repo_default_paths() -> dict[str, str]:
    sibling = ROOT.parent / "imerg2meteo_calib"
    return {
        "calib_dir": str(sibling / "output" / "calib_imerg"),
        "v5_dir": str(sibling / "output" / "imerg_rfactor_calib_v5_year_anchor"),
        "v6_dir": str(ROOT / "output" / "v6_imerg_calibrated"),
        "diagnostics_dir": str(ROOT / "output" / "v6_diagnostics"),
        "aws_csv": str(ROOT / "output" / "aws310_pluvio_v2.csv"),
        "biomet_csv": str(Path.home() / "Downloads" / "Biomet01_12_2024-31_12_2025.csv"),
    }


def _find_default_raw_zip() -> Optional[str]:
    root = Path(r"D:\Cache\Yandex.Disk")
    if not root.exists():
        return None

    candidates: List[Path] = []
    for child in root.iterdir():
        if not child.is_dir() or "25-28" not in child.name:
            continue
        try:
            candidates.extend(child.rglob("IMERG_RFACTOR_ANNUAL-*.zip"))
        except OSError:
            continue
    if not candidates:
        return None
    return str(sorted(candidates)[-1])


def parse_args() -> argparse.Namespace:
    defaults = _repo_default_paths()
    parser = argparse.ArgumentParser(description="Evaluate IMERG v6 hybrid calibration.")
    parser.add_argument("--calib-dir", default=defaults["calib_dir"])
    parser.add_argument("--v5-dir", default=defaults["v5_dir"])
    parser.add_argument("--v6-dir", default=defaults["v6_dir"])
    parser.add_argument("--raw-zip", default=_find_default_raw_zip())
    parser.add_argument("--diagnostics-dir", default=defaults["diagnostics_dir"])
    parser.add_argument("--aws-csv", default=defaults["aws_csv"])
    parser.add_argument("--biomet-csv", default=defaults["biomet_csv"])
    parser.add_argument("--year-start", type=int, default=2001)
    parser.add_argument("--year-end", type=int, default=2024)
    parser.add_argument("--half-window", type=int, default=7)
    return parser.parse_args()


def calc_pbias(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs)
    if not np.any(mask):
        return np.nan
    denom = float(np.sum(obs[mask]))
    if denom == 0.0:
        return np.nan
    return 100.0 * float(np.sum(sim[mask] - obs[mask])) / denom


def calc_kge(sim: np.ndarray, obs: np.ndarray) -> float:
    sim = np.asarray(sim, dtype=np.float64)
    obs = np.asarray(obs, dtype=np.float64)
    mask = np.isfinite(sim) & np.isfinite(obs)
    sim = sim[mask]
    obs = obs[mask]
    if sim.size < 2 or obs.size < 2:
        return np.nan
    obs_std = float(np.std(obs))
    sim_std = float(np.std(sim))
    obs_mean = float(np.mean(obs))
    sim_mean = float(np.mean(sim))
    if obs_std == 0.0 or obs_mean == 0.0:
        return np.nan
    r = float(np.corrcoef(obs, sim)[0, 1])
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    return 1.0 - float(np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def erosivity_proxy_3h(series_mm: np.ndarray, dt_h: float = 3.0) -> float:
    arr = np.asarray(series_mm, dtype=np.float64)
    return float(np.sum(arr * (arr / dt_h)))


def group_raw_tifs_by_year(root_dir: str) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    for path in sorted(Path(root_dir).rglob("*.tif")):
        match = RAW_RE.search(path.name)
        if not match:
            continue
        grouped.setdefault(int(match.group(1)), []).append(str(path))
    return grouped


def group_annual_tifs_by_year(root_dir: str) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    root = Path(root_dir)
    if not root.exists():
        return grouped
    for path in sorted(root.glob("*.tif")):
        match = ANNUAL_RE.search(path.name)
        if not match:
            continue
        grouped.setdefault(int(match.group(1)), []).append(str(path))
    return grouped


def _wet_quantile(series_mm: pd.Series, dt_hours: float = 0.5, q: float = 99.0) -> float:
    wet = series_mm[series_mm > 0.05]
    if wet.empty:
        return 0.0
    return float(np.percentile(wet.to_numpy(dtype=np.float64) / dt_hours, q))


def _max_3h_total(series_mm: pd.Series) -> float:
    if series_mm.empty:
        return 0.0
    return float(series_mm.resample("3h").sum().max())


def unit_energy_brown_foster(i_mm_h: np.ndarray) -> np.ndarray:
    arr = np.asarray(i_mm_h, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    return 0.29 * (1.0 - 0.72 * np.exp(-0.05 * arr))


def find_events(rain_mm: np.ndarray, dt_hours: float = 0.5, separation_hours: float = 6.0, sep_max_rain_mm: float = 1.27) -> List[tuple[int, int]]:
    rain_mm = np.asarray(rain_mm, dtype=np.float64)
    wet_idx = np.flatnonzero(rain_mm > 0.0)
    if wet_idx.size == 0:
        return []

    sep_steps = int(round(separation_hours / dt_hours))
    events: List[tuple[int, int]] = []
    start = int(wet_idx[0])
    last_wet = int(wet_idx[0])

    for idx in wet_idx[1:]:
        idx = int(idx)
        gap = idx - last_wet
        if gap >= sep_steps:
            gap_rain = float(rain_mm[last_wet + 1 : idx].sum())
            if gap_rain < sep_max_rain_mm:
                events.append((start, last_wet))
                start = idx
        last_wet = idx

    events.append((start, last_wet))
    return events


def compute_r_factor(series_mm: pd.Series, dt_hours: float = 0.5) -> float:
    if series_mm.empty:
        return 0.0

    series = series_mm.resample("30min").sum().fillna(0.0)
    rain = series.to_numpy(dtype=np.float64)
    intensity = rain / dt_hours
    energy = unit_energy_brown_foster(intensity) * rain

    total_r = 0.0
    for i0, i1 in find_events(rain, dt_hours=dt_hours):
        p_event = float(rain[i0 : i1 + 1].sum())
        i30 = float(intensity[i0 : i1 + 1].max())
        if p_event < 12.7 and i30 < 25.4:
            continue
        total_r += float(energy[i0 : i1 + 1].sum()) * i30
    return total_r


def compute_station_metrics(year_df: pd.DataFrame, sim_col: str) -> dict[str, float]:
    obs = year_df["P_station_mm"].to_numpy(dtype=np.float64)
    sim = year_df[sim_col].to_numpy(dtype=np.float64)

    daily = (
        year_df.assign(date=year_df["datetime"].dt.floor("D"))
        .groupby("date", as_index=False)[["P_station_mm", sim_col]]
        .sum()
    )
    wet_obs = obs > 0.1
    wet_sim = sim > 0.1
    hit = int(np.sum(wet_obs & wet_sim))
    miss = int(np.sum(wet_obs & ~wet_sim))
    false_alarm = int(np.sum(~wet_obs & wet_sim))
    csi = hit / max(hit + miss + false_alarm, 1)

    return {
        "annual_total_obs_mm": float(np.sum(obs)),
        "annual_total_sim_mm": float(np.sum(sim)),
        "annual_total_pbias_pct": calc_pbias(sim, obs),
        "daily_kge": calc_kge(daily[sim_col].to_numpy(dtype=np.float64), daily["P_station_mm"].to_numpy(dtype=np.float64)),
        "proxy_r_obs": erosivity_proxy_3h(obs),
        "proxy_r_sim": erosivity_proxy_3h(sim),
        "proxy_r_pbias_pct": calc_pbias(
            np.array([erosivity_proxy_3h(sim)], dtype=np.float64),
            np.array([erosivity_proxy_3h(obs)], dtype=np.float64),
        ),
        "p99_obs_mmh": float(np.percentile(obs[obs > 0.05] / 3.0, 99)) if np.any(obs > 0.05) else 0.0,
        "p99_sim_mmh": float(np.percentile(sim[sim > 0.05] / 3.0, 99)) if np.any(sim > 0.05) else 0.0,
        "wet_csi": float(csi),
        "wet_hit": hit,
        "wet_miss": miss,
        "wet_false_alarm": false_alarm,
        "n_steps": int(len(year_df)),
    }


def run_station_cross_validation(
    station_tables: Mapping[int, pd.DataFrame],
    year_start: int,
    year_end: int,
    half_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[dict[str, object]] = []

    for year in tqdm(range(year_start, year_end + 1), desc="station-cv"):
        year_models = fit_station_models_for_year(station_tables, target_year=year, half_window_years=half_window)
        for wmo, model in year_models.items():
            station_df = station_tables[wmo]
            year_df = apply_station_models_to_target_year(station_df, target_year=year, station_models=model)
            if year_df.empty:
                continue

            method_cols = {"raw": "P_sat_mm", "v1": "P_corrected_mm", "v6": "P_v6_mm"}
            for method, col in method_cols.items():
                if col not in year_df.columns or year_df[col].isna().all():
                    continue
                metrics = compute_station_metrics(year_df, col)
                rows.append(
                    {
                        "year": year,
                        "wmo_index": int(wmo),
                        "method": method,
                        "blend_alpha": float(model.blend_alpha),
                        **metrics,
                    }
                )

    metrics_df = pd.DataFrame(rows)
    summary_df = (
        metrics_df.groupby("method", as_index=False)
        .agg(
            n_station_years=("wmo_index", "count"),
            annual_total_pbias_median=("annual_total_pbias_pct", "median"),
            annual_total_pbias_mean=("annual_total_pbias_pct", "mean"),
            daily_kge_median=("daily_kge", "median"),
            proxy_r_pbias_median=("proxy_r_pbias_pct", "median"),
            proxy_r_pbias_mean=("proxy_r_pbias_pct", "mean"),
            wet_csi_median=("wet_csi", "median"),
            p99_sim_mmh_median=("p99_sim_mmh", "median"),
            p99_obs_mmh_median=("p99_obs_mmh", "median"),
        )
        .sort_values("method")
        .reset_index(drop=True)
    )
    yearly_df = (
        metrics_df.groupby(["year", "method"], as_index=False)
        .agg(
            annual_total_pbias_median=("annual_total_pbias_pct", "median"),
            proxy_r_pbias_median=("proxy_r_pbias_pct", "median"),
            daily_kge_median=("daily_kge", "median"),
            wet_csi_median=("wet_csi", "median"),
            n_station_years=("wmo_index", "count"),
        )
        .sort_values(["year", "method"])
        .reset_index(drop=True)
    )
    return metrics_df, summary_df, yearly_df


def plot_station_verification(metrics_df: pd.DataFrame, yearly_df: pd.DataFrame, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    for ax, metric, title in [
        (axes[0], "annual_total_pbias_pct", "Annual Total PBIAS"),
        (axes[1], "proxy_r_pbias_pct", "Proxy-R PBIAS"),
    ]:
        data = [metrics_df.loc[metrics_df["method"] == method, metric].dropna().to_numpy(dtype=np.float64) for method in METHOD_ORDER]
        box = ax.boxplot(data, tick_labels=METHOD_ORDER, patch_artist=True, showfliers=False)
        for patch, method in zip(box["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.7)
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    for method in METHOD_ORDER:
        sub = yearly_df[yearly_df["method"] == method]
        ax.plot(sub["year"], sub["proxy_r_pbias_median"], marker="o", linewidth=1.8, markersize=3.5, color=METHOD_COLORS[method], label=method)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Median Proxy-R PBIAS by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("%")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_peak_diagnostics(training_csv: str, meta_csv: str, out_path: str) -> None:
    if not os.path.exists(training_csv) or not os.path.exists(meta_csv):
        return
    train_df = pd.read_csv(training_csv)
    meta_df = pd.read_csv(meta_csv)
    meta_map = dict(zip(meta_df["key"], meta_df["value"]))
    templates = eval(meta_map.get("templates", "{}"), {"__builtins__": {}})

    fig = plt.figure(figsize=(14, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1)
    for season, color in SEASON_COLORS.items():
        for site, marker in SITE_MARKERS.items():
            sub = train_df[(train_df["season"] == season) & (train_df["site"] == site)]
            if sub.empty:
                continue
            ax1.scatter(
                sub["raw_peak_share"],
                sub["gamma_star"],
                c=color,
                marker=marker,
                s=42,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.5,
                label=f"{season}-{site}",
            )
    ax1.set_xlabel("Raw 3h peak share")
    ax1.set_ylabel("Fitted sharpening gamma")
    ax1.set_title("Peak-Model Training Windows")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(1, 2, 2)
    slots = np.arange(1, 7)
    for season, color in SEASON_COLORS.items():
        weights = np.asarray(templates.get(season, [1 / 6.0] * 6), dtype=np.float64)
        ax2.plot(slots, weights, marker="o", linewidth=2.0, color=color, label=season)
    ax2.set_xticks(slots)
    ax2.set_xlabel("30-min slot within 3h window")
    ax2.set_ylabel("Mean weight")
    ax2.set_title("Seasonal Intra-Window Templates")
    ax2.grid(alpha=0.25)
    ax2.legend(frameon=False)

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_highfreq_product_metrics(
    raw_zip: Optional[str],
    v5_dir: str,
    v6_dir: str,
    aws_csv: str,
    biomet_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    site_specs: List[tuple[str, dict[str, float], pd.Series]] = []
    if os.path.exists(aws_csv):
        site_specs.append(("aws310", AWS310_COORDS, load_aws_series(aws_csv)))
    if os.path.exists(biomet_csv):
        site_specs.append(("biomet", BIOMET_COORDS, load_biomet_series(biomet_csv)))
    if not site_specs:
        return pd.DataFrame(), pd.DataFrame()

    raw_groups: Dict[int, List[str]] = {}
    tmp_root: Optional[str] = None
    if raw_zip and os.path.exists(raw_zip):
        tmp_root = tempfile.mkdtemp(prefix="imerg_v6_eval_")
        with zipfile.ZipFile(raw_zip, "r") as zf:
            zf.extractall(tmp_root)
        raw_groups = group_raw_tifs_by_year(tmp_root)

    try:
        v5_groups = group_annual_tifs_by_year(v5_dir)
        v6_groups = group_annual_tifs_by_year(v6_dir)

        product_rows: List[dict[str, object]] = []
        pair_rows: List[dict[str, object]] = []

        for site_name, coords, ground_series in site_specs:
            if ground_series.empty:
                continue

            overlap_years = sorted(set(int(y) for y in ground_series.index.year))
            series_by_product: Dict[str, pd.Series] = {"ground": ground_series}
            if raw_groups:
                series_by_product["raw"] = extract_pixel_series(raw_groups, lon=coords["lon"], lat=coords["lat"], years=overlap_years)
            if v5_groups:
                series_by_product["v5"] = extract_pixel_series(v5_groups, lon=coords["lon"], lat=coords["lat"], years=overlap_years)
            if v6_groups:
                series_by_product["v6"] = extract_pixel_series(v6_groups, lon=coords["lon"], lat=coords["lat"], years=overlap_years)

            available_years = sorted({int(y) for s in series_by_product.values() for y in s.index.year.unique()})
            for year in available_years:
                year_series: Dict[str, pd.Series] = {}
                for product, series in series_by_product.items():
                    sub = series[series.index.year == year].copy()
                    if sub.empty:
                        continue
                    sub = sub.resample("30min").sum().fillna(0.0)
                    year_series[product] = sub

                    product_rows.append(
                        {
                            "site": site_name,
                            "year": year,
                            "product": product,
                            "annual_total_mm": float(sub.sum()),
                            "p99_30min_mmh": _wet_quantile(sub, dt_hours=0.5, q=99.0),
                            "max_3h_total_mm": _max_3h_total(sub),
                            "annual_r": compute_r_factor(sub, dt_hours=0.5),
                        }
                    )

                ground_year = year_series.get("ground")
                if ground_year is None:
                    continue
                ground_3h = ground_year.resample("3h").sum().fillna(0.0)
                for product in ("raw", "v5", "v6"):
                    sim_year = year_series.get(product)
                    if sim_year is None:
                        continue
                    idx = ground_year.index.union(sim_year.index)
                    obs = ground_year.reindex(idx, fill_value=0.0)
                    sim = sim_year.reindex(idx, fill_value=0.0)
                    obs_3h = ground_3h.reindex(ground_3h.index.union(sim.resample("3h").sum().index), fill_value=0.0)
                    sim_3h = sim.resample("3h").sum().reindex(obs_3h.index, fill_value=0.0)
                    wet_obs = obs_3h > 0.3
                    wet_sim = sim_3h > 0.3
                    hit = int(np.sum(wet_obs & wet_sim))
                    miss = int(np.sum(wet_obs & ~wet_sim))
                    false_alarm = int(np.sum(~wet_obs & wet_sim))

                    pair_rows.append(
                        {
                            "site": site_name,
                            "year": year,
                            "product": product,
                            "ground_annual_total_mm": float(obs.sum()),
                            "ground_annual_r": compute_r_factor(obs, dt_hours=0.5),
                            "annual_total_pbias_pct": calc_pbias(sim.to_numpy(dtype=np.float64), obs.to_numpy(dtype=np.float64)),
                            "annual_r_pbias_pct": calc_pbias(
                                np.array([compute_r_factor(sim, dt_hours=0.5)], dtype=np.float64),
                                np.array([compute_r_factor(obs, dt_hours=0.5)], dtype=np.float64),
                            ),
                            "p99_bias_pct": calc_pbias(
                                np.array([_wet_quantile(sim, dt_hours=0.5)], dtype=np.float64),
                                np.array([_wet_quantile(obs, dt_hours=0.5)], dtype=np.float64),
                            ),
                            "wet_3h_csi": hit / max(hit + miss + false_alarm, 1),
                            "hit": hit,
                            "miss": miss,
                            "false_alarm": false_alarm,
                        }
                    )

        return pd.DataFrame(product_rows), pd.DataFrame(pair_rows)
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)


def plot_highfreq_summary(product_df: pd.DataFrame, out_path: str) -> None:
    if product_df.empty:
        return
    pivot_r = product_df.pivot_table(index=["site", "year"], columns="product", values="annual_r", aggfunc="first")
    pivot_p99 = product_df.pivot_table(index=["site", "year"], columns="product", values="p99_30min_mmh", aggfunc="first")
    groups = list(pivot_r.index)
    if not groups:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    width = 0.18
    x = np.arange(len(groups), dtype=np.float64)
    products = [p for p in ["ground", "raw", "v5", "v6"] if p in pivot_r.columns or p in pivot_p99.columns]

    for i, product in enumerate(products):
        offset = (i - (len(products) - 1) / 2.0) * width
        axes[0].bar(x + offset, [pivot_r.get(product, pd.Series(index=groups, dtype=float)).get(idx, np.nan) for idx in groups], width=width, color=METHOD_COLORS.get(product, "#444444"), label=product, alpha=0.85)
        axes[1].bar(x + offset, [pivot_p99.get(product, pd.Series(index=groups, dtype=float)).get(idx, np.nan) for idx in groups], width=width, color=METHOD_COLORS.get(product, "#444444"), label=product, alpha=0.85)

    labels = [f"{site}-{year}" for site, year in groups]
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_title("Annual R at High-Frequency Sites")
    axes[0].set_ylabel("MJ mm ha-1 h-1 yr-1")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_title("99th Percentile 30-min Intensity")
    axes[1].set_ylabel("mm h-1")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.diagnostics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_tables = load_calibration_tables(args.calib_dir)
    metrics_df, summary_df, yearly_df = run_station_cross_validation(
        station_tables=station_tables,
        year_start=args.year_start,
        year_end=args.year_end,
        half_window=args.half_window,
    )
    metrics_df.to_csv(out_dir / "v6_station_year_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(out_dir / "v6_station_method_summary.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(out_dir / "v6_station_year_summary.csv", index=False, encoding="utf-8-sig")
    plot_station_verification(metrics_df, yearly_df, str(out_dir / "v6_station_verification.png"))

    plot_peak_diagnostics(
        training_csv=str(out_dir / "v6_peak_model_training.csv"),
        meta_csv=str(out_dir / "v6_peak_model_meta.csv"),
        out_path=str(out_dir / "v6_peak_model_diagnostics.png"),
    )

    product_df, pair_df = build_highfreq_product_metrics(
        raw_zip=args.raw_zip,
        v5_dir=args.v5_dir,
        v6_dir=args.v6_dir,
        aws_csv=args.aws_csv,
        biomet_csv=args.biomet_csv,
    )
    if not product_df.empty:
        product_df.to_csv(out_dir / "v6_highfreq_product_metrics.csv", index=False, encoding="utf-8-sig")
    if not pair_df.empty:
        pair_df.to_csv(out_dir / "v6_highfreq_pair_metrics.csv", index=False, encoding="utf-8-sig")
    plot_highfreq_summary(product_df, str(out_dir / "v6_highfreq_summary.png"))

    print("\n=== Station CV summary ===")
    print(summary_df.to_string(index=False))
    if not pair_df.empty:
        pair_valid = pair_df[(pair_df["ground_annual_total_mm"] >= 25.0) & (pair_df["ground_annual_r"] > 0.0)].copy()
        print("\n=== High-frequency site pair summary ===")
        print(
            pair_valid.groupby("product", as_index=False)
            .agg(
                annual_total_pbias_median=("annual_total_pbias_pct", "median"),
                annual_r_pbias_median=("annual_r_pbias_pct", "median"),
                p99_bias_median=("p99_bias_pct", "median"),
                wet_3h_csi_median=("wet_3h_csi", "median"),
            )
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
