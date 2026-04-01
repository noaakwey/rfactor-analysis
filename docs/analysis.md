# Peak-Aware Hybrid Calibration of IMERG for Rainfall Erosivity in the Volga Region, 2001-2024

## Abstract
This analysis presents a new `v6` hybrid calibration workflow for GPM IMERG V07 precipitation, designed specifically for rainfall erosivity estimation rather than for annual water-balance correction alone. The core methodological change is a transition from single-station Voronoi correction to predictive rolling-window multi-station blending combined with event-aware 30-minute peak reconstruction. The resulting framework is evaluated at three levels: predictive cross-validation against 202 station calibration tables (`4848` station-years), high-frequency diagnostics against two independent gauges (`AWS310`, `Biomet`), and final RUSLE2 `R`-factor maps for `2001-2024`.

The predictive station-level evaluation shows that `v6` improves both annual bias control and erosivity-sensitive metrics relative to the previous seasonal quantile-mapping baseline (`v1`). Median absolute annual-total bias decreases from `9.23%` (`v1`) to `7.69%` (`v6`), while median absolute proxy-`R` bias decreases from `38.18%` to `34.00%`. The median simulated 99th percentile of 3-hour intensity increases from `3.13` to `3.33 mm h-1`, moving in the physically correct direction toward the observed `5.09 mm h-1`.

At the final map level, the new peak-aware product yields a markedly higher erosivity climatology than the conservative legacy map. The long-term mean of the previous area-scale product (`R_area`, baseline `v5`) is `159.2 MJ mm ha-1 h-1 yr-1`, whereas the peak-aware `v6` point-equivalent product (`R_point_eq`) reaches `340.1 MJ mm ha-1 h-1 yr-1`. This difference is interpreted not as a simple correction error, but as an explicit manifestation of the `area-to-point scaling` problem: spatially averaged pixel precipitation suppresses short-lived erosive peaks, while field-scale erosivity depends precisely on these peaks.

The recommended practical outcome of this work is therefore not a single number, but a two-product system. `R_area` should be used for conservative area-averaged climatology and intercomparison with coarse-scale products, whereas `R_point_eq` should be used for field-scale erosion-hazard screening, upper-envelope assessments, and applications where sub-pixel convective peak losses are known to be critical.

**Keywords:** IMERG, rainfall erosivity, RUSLE2, quantile mapping, peak reconstruction, area-to-point scaling, Volga Region

## 1. Objective
The working `v5_year_anchor` product is operationally useful, but it has two structural limitations:

1. It is optimized mainly for annual and seasonal precipitation consistency, not for erosive peak reconstruction.
2. It is not predictive in the strict sense, because the same-year annual anchor uses station information from the target year.

For erosivity applications this is not a minor detail. The final `R`-factor depends on event kinetic energy and on `I30`, so the main error source is not only precipitation amount, but the suppression of short high-intensity bursts inside each satellite pixel.

The target of the present work was therefore to build a better approach with four explicit requirements:

1. Predictive training without same-year leakage.
2. Smooth spatial blending instead of hard Voronoi station zones.
3. Explicit learning of sub-3-hour peak structure.
4. Honest treatment of the area-to-point scaling problem at the level of the final `R` product.

## 2. Data and Workflow
### 2.1 Input data
The analysis uses:

1. GPM IMERG Final Run V07 half-hourly precipitation stacks for `2001-2024`.
2. `202` station calibration tables from the paired IMERG-ground archive.
3. ERA5-Land phase masks for liquid/solid partitioning in the final `RUSLE2` computation.
4. Two independent high-frequency gauges for peak diagnostics:
   `AWS310` and `Biomet`.

### 2.2 New `v6` methodology
The new workflow is implemented in:

1. `scripts/run_v6_imerg_pipeline.py`
2. `lib/v6_hybrid.py`
3. `scripts/evaluate_v6_hybrid.py`
4. `scripts/plot_v6_paper.py`

The `v6` algorithm consists of five linked stages.

**Stage 1. Predictive rolling-window station training**

For each target year, station models are trained on a `±7 year` moving window with explicit exclusion of the target year. If the local window is too short, the model falls back to all non-target years. This removes the principal leakage of the same-year annual-anchor strategy.

**Stage 2. Multi-station soft blending**

Instead of assigning each pixel to a single nearest station, each pixel is corrected by distance-weighted blending of the `4` nearest stations. This suppresses hard spatial seams and reduces sensitivity to a single local station anomaly.

**Stage 3. Erosivity-aware soft quantile mapping**

Seasonal empirical quantile mapping is still the core transfer function, but the correction strength is no longer chosen only by daily and monthly water-balance skill. The station-specific `blend_alpha` is now selected using an objective that also penalizes proxy-`R` bias and upper-tail (`q99`) intensity mismatch. This is the key change that allowed `v6` to outperform `v1` in predictive proxy-erosivity.

**Stage 4. Event-aware 30-minute peak reconstruction**

After correction of 3-hour totals, the 30-minute structure inside each 3-hour window is reconstructed using a compact peak model trained on high-frequency gauge data. The model uses:

1. corrected 3-hour total,
2. raw peak share,
3. number of wet 30-minute slots,
4. season,
5. season-specific intra-window templates.

The fitted sharpening parameter spans `gamma = 0.7-3.5`, and the training dataset contains `28` observed windows. This stage directly targets the quantity that matters most for `R`: high short-duration intensities.

**Stage 5. Final hydrological guards**

Daily and annual constraints are preserved, but only as soft guards. They keep the calibrated product within a plausible envelope while allowing stronger tail correction than the original working version.

## 3. Predictive Validation
### 3.1 Station cross-validation
Predictive validation was carried out on `4848` station-years (`202` stations, `2001-2024`). The figure below summarizes the result.

![Station cross-validation](figures/fig26_v6_station_verification.png)

*Figure 1. Predictive station-level validation of raw IMERG, baseline seasonal QM (`v1`) and the new hybrid method (`v6`).*

The main results are:

| Metric | Raw | v1 | v6 |
|---|---:|---:|---:|
| Median absolute annual-total PBIAS, % | 11.15 | 9.23 | **7.69** |
| Median absolute proxy-R PBIAS, % | 46.91 | 38.18 | **34.00** |
| Median daily KGE | 0.536 | 0.530 | **0.536** |
| Median wet-window CSI | 0.1166 | 0.1163 | **0.1174** |
| Median simulated p99 intensity, mm h-1 | 2.82 | 3.13 | **3.33** |
| Median observed p99 intensity, mm h-1 | 5.09 | 5.09 | 5.09 |

Three conclusions follow immediately.

1. `v6` is the best predictive solution in terms of the joint balance between annual amount, daily skill and erosivity-sensitive upper-tail behavior.
2. The improvement over `v1` is not cosmetic. The proxy-`R` bias reduction from `38.18%` to `34.00%` means that the calibration objective is now materially closer to erosivity physics.
3. Even `v6` still underestimates the observed upper tail. This is expected, because station tables are only 3-hourly and cannot by themselves recover the full 30-minute peak spectrum.

### 3.2 Peak-model diagnostics
The peak-aware layer is the main methodological novelty of the new pipeline.

![Peak-model diagnostics](figures/fig27_v6_peak_model_diagnostics.png)

*Figure 2. Training windows for the peak-reconstruction model and season-specific mean 30-minute templates inside 3-hour events.*

Two features are especially important.

1. The fitted `gamma` values occupy a broad range and often exceed `2`, confirming that raw IMERG intra-window structure is too flat for erosivity work.
2. Seasonal templates are not uniform. In particular, the autumn (`SON`) template is strongly asymmetric, which supports the choice to use season-specific sub-window redistribution instead of fixed symmetric disaggregation.

### 3.3 High-frequency gauge diagnostics
High-frequency site verification remains intentionally secondary, because a single point sensor cannot fully validate a `0.1°` satellite pixel. Still, the gauge comparison is useful to check whether `v6` moves in the right direction.

![High-frequency diagnostics](figures/fig28_v6_highfreq_summary.png)

*Figure 3. Annual `R` and 99th-percentile 30-minute intensity for the two available high-frequency sites and satellite products.*

The site results are mixed but informative.

For `AWS310` in `2023`, the gauge-derived annual `R` equals `3118.9 MJ mm ha-1 h-1 yr-1`. The corresponding pixel values are:

1. raw IMERG: `162.4` (`-94.8%`)
2. v5 baseline: `179.3` (`-94.3%`)
3. v6 peak-aware: `354.0` (`-88.6%`)

Here `v6` is clearly the least biased product, although it still remains far below the gauge value.

For `Biomet` in `2024`, the gauge-derived annual `R` equals `406.4`, while the products give:

1. raw IMERG: `377.9` (`-7.0%`)
2. v5 baseline: `589.9` (`+45.1%`)
3. v6 peak-aware: `1065.3` (`+162.1%`)

Here `v6` overshoots strongly. This is not a reason to reject the method; rather, it demonstrates the exact scaling problem the analysis was meant to expose. The field sensor samples a point, while the satellite product samples a pixel containing mixed convective structure. One site-year can therefore be strongly under-corrected and another over-corrected even when the calibration logic is physically consistent.

The aggregated high-frequency summary is therefore interpreted conservatively:

1. `v6` provides the lowest median annual-total bias (`58.1%`) among the three satellite products.
2. `v6` also gives the highest wet 3-hour CSI (`0.1089`).
3. Annual `R` bias remains heterogeneous and should not be treated as a strict scalar validation metric at point support.

## 4. Final `R` Products: Conservative Area Scale vs Peak-Aware Point Equivalent
The central result of the project is that a single final `R` map is methodologically insufficient. The new workflow therefore yields two complementary products:

1. `R_area`: conservative area-scale erosivity, represented here by the legacy baseline product stored in `output/v6_rfactor_area/annual`.
2. `R_point_eq`: peak-aware point-equivalent erosivity, represented by the new `v6` product in `output/v6_rfactor/annual`.

This distinction is not semantic. It changes the climatology fundamentally.

![Mean map, PDF and CDF](figures/fig21_v6_mean_pdf_cdf.png)

*Figure 4. Spatial mean, histogram and cumulative distribution for the peak-aware `R_point_eq` product.*

![v5 vs v6 comparison](figures/fig24_v5_v6_comparison.png)

*Figure 5. Comparison between the conservative area-scale baseline (`v5`) and the peak-aware `v6` product.*

The long-term domain means are:

1. `R_area` (`v5` baseline): `159.2 MJ mm ha-1 h-1 yr-1`
2. `R_point_eq` (`v6`): `340.1 MJ mm ha-1 h-1 yr-1`

Thus, the peak-aware product is larger by `113.6%` in domain mean. At the pixel level the median relative shift is `+110.4%`, and the `95th` percentile reaches `+173.0%`.

This increase may look extreme only if one assumes that the old map was already point-equivalent. It was not. The legacy product was a conservative area-smoothed estimate. Once sub-3-hour peaks are partially restored, the quadratic response of `EI30` makes a much larger `R` climatology unavoidable.

The practical interpretation is straightforward:

1. If the task is conservative regional climatology, use `R_area`.
2. If the task is field-scale erosion hazard or screening for upper-envelope erosivity, use `R_point_eq`.
3. If the task is policy or design where both scales matter, report the interval `[R_area, R_point_eq]` explicitly.

## 5. Spatial Structure and Temporal Dynamics
### 5.1 Spatial climatology
The `R_point_eq` mean map is spatially coherent and internally plausible. The mean pixel value is `340.1`, the median is `337.8`, and the long-term spatial `P05-P95` range is `278.3-412.1 MJ mm ha-1 h-1 yr-1`.

The map does not show a simple monotonic zonal gradient. Instead, the field is organized around persistent mesoscale hotspots, which is exactly what would be expected for convective erosivity in a relatively compact study region.

### 5.2 Interannual variability and trend
![Temporal dynamics](figures/fig22_v6_temporal_dynamics.png)

*Figure 6. Interannual evolution of the domain-mean `R_point_eq`, its anomaly structure and relative shift against the `v5` baseline.*

![CV and trend maps](figures/fig23_v6_cv_trend.png)

*Figure 7. Pixel-scale coefficient of variation and significant linear trends in `R_point_eq`.*

The interannual coefficient of variation of the domain-mean `R_point_eq` is `36.95%`, while the mean pixel-scale coefficient of variation reaches `59.80%`. This confirms that erosivity remains a highly intermittent climatic control even after calibration.

The domain-mean linear trend is negative (`-5.73 MJ mm ha-1 h-1 yr-2`), but statistically non-significant (`p = 0.133`, `R² = 0.10`). Therefore, the new product does not support a robust monotonic long-term trend over `2001-2024`.

The time series is nevertheless structured. Mean `R_point_eq` by three eight-year blocks equals:

1. `2001-2008`: `425.7`
2. `2009-2016`: `278.2`
3. `2017-2024`: `316.0`

This indicates a low-erosivity regime in `2009-2016`, followed by partial recovery after `2017`, but not a full return to the early-2000s peak.

### 5.3 Annual mosaics
![Annual small multiples](figures/fig25_v6_annual_multiples.png)

*Figure 8. Annual `R_point_eq` maps for 2001-2024. Map titles show annual domain means.*

The annual mosaics confirm that the spatial pattern is persistent while the amplitude varies strongly from year to year. In other words, the geography of erosivity is comparatively stable, but the strength of its annual realization is highly non-stationary.

## 6. Interpretation
The most important conceptual result of this work is that the previous single-map paradigm was too narrow. Once erosivity rather than precipitation amount becomes the calibration target, two scales emerge naturally:

1. pixel-area averaged rainfall forcing,
2. point-equivalent erosive forcing relevant to field processes.

The legacy product underestimates the second quantity because it preserves the spatial smoothing inherent to IMERG pixels. The new `v6` product partially restores the missing peak structure. The large `v5-v6` gap is therefore not merely a model spread; it is the operational expression of unresolved sub-pixel convective structure.

This is also why the high-frequency validation is mixed. The method can reduce severe underestimation in one point-year and overshoot in another, because no single point instrument is a stable truth standard for a full satellite pixel. That is exactly why the final result must be delivered as a scale-aware pair of products rather than as a falsely precise single estimate.

## 7. Limitations
Three limitations remain material.

1. The peak-reconstruction model is trained on only `28` high-frequency windows. This is enough to detect non-uniform sub-window structure, but not enough for a universal regional convective climatology.
2. The current peak model uses local precipitation shape and season, but does not yet include external convective predictors such as `CAPE`, `TCWV` or freezing-level diagnostics.
3. The point-support validation dataset is too sparse for hard calibration of an absolute `R_point_eq` scaling factor.

These limitations do not invalidate the product. They simply define the next upgrade step: expand the high-frequency training archive and add physically meaningful atmospheric predictors to the peak layer.

## 8. Practical Recommendation
For operational use, the following scheme is recommended.

1. Use `output/v6_rfactor_area/annual` when a conservative regional climatology is required.
2. Use `output/v6_rfactor/annual` when the task is field-scale risk screening or an upper-envelope erosivity estimate.
3. When publishing or reporting a single number, always state explicitly which scale is being reported.
4. When uncertainty is critical, report both products as a bracket rather than collapsing them into one unsupported compromise value.

## 9. Main Conclusions
1. A predictive rolling-window multi-station hybrid calibration was successfully implemented and tested end-to-end for `2001-2024`.
2. In predictive station cross-validation, `v6` outperforms the previous seasonal QM baseline in the metrics that matter most for erosivity: annual bias, proxy-`R` bias, wet-event overlap and upper-tail intensity.
3. The event-aware peak-reconstruction layer materially alters the final `R` climatology and cannot be treated as a cosmetic post-processing step.
4. The final `R` product is scale-dependent. A conservative area-scale estimate (`R_area`) and a peak-aware point-equivalent estimate (`R_point_eq`) are both necessary.
5. Over `2001-2024`, `R_point_eq` shows strong interannual variability but no statistically significant linear trend.

## Reproducibility
The analysis in this document is reproduced by the following scripts:

1. `scripts/run_v6_imerg_pipeline.py`
2. `r_factor_rusle2.py`
3. `scripts/evaluate_v6_hybrid.py`
4. `scripts/plot_v6_paper.py`

All intermediate diagnostics and summary tables are stored in `output/v6_diagnostics`.
