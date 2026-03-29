import numpy as np
import pandas as pd
import scipy.stats as stats

def get_season(month):
    if month in [12, 1, 2]: return 'DJF'
    elif month in [3, 4, 5]: return 'MAM'
    elif month in [6, 7, 8]: return 'JJA'
    else: return 'SON'

def compute_quantiles(series, q_range):
    if len(series) == 0: return np.zeros_like(q_range)
    return np.quantile(series, q_range)

def apply_qm(val, q_sat, q_station, slope_extrap, p_th, tail_power=1.0):
    if isinstance(val, (int, float, np.float64, np.float32, np.int64)):
        return _apply_qm_scalar(val, q_sat, q_station, slope_extrap, p_th, tail_power)
    
    val_arr = np.asarray(val)
    res = np.zeros_like(val_arr, dtype=float)
    pos_mask = val_arr > p_th
    if not np.any(pos_mask): return res
    
    # Vectorized check for quantiles
    res[pos_mask] = [_apply_qm_scalar(v, q_sat, q_station, slope_extrap, p_th, tail_power) for v in val_arr[pos_mask]]
    return res

def _apply_qm_scalar(v, q_sat, q_station, slope_extrap, p_th, tail_power=1.0):
    if v <= p_th: return 0.0
    if v <= q_sat[0]:
        dx = q_sat[0] - p_th
        return q_station[0] * ((v - p_th) / dx) if dx > 0 else q_station[0]
            
    if v >= q_sat[-1]:
        linear_incr = slope_extrap * (v - q_sat[-1])
        if tail_power != 1.0 and q_sat[-1] > 0:
            inflation = (v / q_sat[-1])**(tail_power - 1.0)
            return q_station[-1] + linear_incr * inflation
        return q_station[-1] + linear_incr
    
    idx = np.searchsorted(q_sat, v)
    if idx == 0: return q_station[0] 
    x0, x1 = q_sat[idx-1], q_sat[idx]
    y0, y1 = q_station[idx-1], q_station[idx]
    return y0 + (y1 - y0) * ((v - x0) / (x1 - x0)) if x1 != x0 else y0

def fit_season_models(train_df, seasons=('DJF', 'MAM', 'JJA', 'SON'), num_q=1000, min_samples=30, vf_clip=(0.7, 1.3)):
    q_levels = np.linspace(0.0, 1.0, num_q + 2)[1:-1]
    models = {}
    for season in seasons:
        sd = train_df[train_df['season'] == season]
        p_sat = sd['P_sat_mm'].values
        p_st = sd['P_station_mm'].values
        if len(p_sat) < min_samples or len(p_st) < min_samples: continue
        
        q_sat = np.quantile(p_sat, q_levels)
        q_st = np.quantile(p_st, q_levels)
        
        slope = (q_st[-1] - q_st[-5]) / (q_sat[-1] - q_sat[-5]) if (q_sat[-1] - q_sat[-5]) > 0 else 1.0
        
        # Calculate Volume Factor (VF)
        train_corr = apply_qm(p_sat, q_sat, q_st, slope, p_th=0.0, tail_power=1.0)
        mean_st = np.mean(p_st)
        mean_corr = np.mean(train_corr)
        vf = np.clip(mean_st / mean_corr, vf_clip[0], vf_clip[1]) if mean_corr > 0 else 1.0
        
        models[season] = {'q_sat': q_sat, 'q_st': q_st, 'slope': slope, 'p_th': 0.0, 'vf': vf}
    return models

def apply_season_models(df, models, tail_power=1.0):
    corrected = df['P_sat_mm'].to_numpy(dtype=float, copy=True)
    for season, params in models.items():
        mask = (df['season'] == season).to_numpy()
        if not np.any(mask): continue
        vals = df.loc[mask, 'P_sat_mm']
        mapped = apply_qm(vals, params['q_sat'], params['q_st'], params['slope'], params['p_th'], tail_power)
        corrected[mask] = np.maximum(mapped, 0.0) * params['vf']
    return corrected
