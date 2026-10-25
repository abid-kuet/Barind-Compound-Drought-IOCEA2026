from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gamma as gammafn
import yaml

ROOT = Path(__file__).resolve().parents[2]


def calc_fao56_ra(doy, lat_deg):
    # FAO-56 Eq. 21 extraterrestrial radiation converted to mm/day (divide by 2.45)
    lat_rad = np.deg2rad(lat_deg)
    dec = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    ws_arg = float(np.clip(-np.tan(lat_rad) * np.tan(dec), -1.0, 1.0))
    ws = np.arccos(ws_arg)
    dr = 1.0 + 0.033 * np.cos(2.0 * np.pi * doy / 365.0)
    ra_mj = (24.0 * 60.0 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(dec) + np.cos(lat_rad) * np.cos(dec) * np.sin(ws)
    )
    return ra_mj / 2.45


def hargreaves_daily(tmin, tmax, tmean, ra):
    dtr = max(tmax - tmin, 0.1)
    return max(0.0023 * ra * (tmean + 17.8) * np.sqrt(dtr), 0.0)


def thornthwaite_monthly(tmean_m, lat_deg):
    lat_rad = np.deg2rad(lat_deg)
    tm = tmean_m.clip(lower=0.0)
    m_clim = tm.groupby(tm.index.month).mean().clip(lower=0.0)
    heat_idx = np.sum((m_clim / 5.0) ** 1.514)
    a = (6.75e-7 * heat_idx ** 3) - (7.71e-5 * heat_idx ** 2) + (1.792e-2 * heat_idx) + 0.49239

    vals = {}
    for dt, val in tm.items():
        dec = 0.409 * np.sin((2.0 * np.pi * dt.dayofyear / 365.0) - 1.39)
        ws_arg = float(np.clip(-np.tan(lat_rad) * np.tan(dec), -1.0, 1.0))
        ws = np.arccos(ws_arg)
        daylen = (24.0 / np.pi) * ws
        raw_pet = 16.0 * ((10.0 * max(val, 0.0) / heat_idx) ** a) if heat_idx > 0 else 0.0
        vals[dt] = raw_pet * (daylen / 12.0) * (dt.days_in_month / 30.0)
    return pd.Series(vals)


def fit_fisk_pwm(x):
    # Unbiased probability-weighted moments for 3-parameter log-logistic (Vicente-Serrano et al. 2010)
    x = np.sort(np.asarray(x, float))
    n = len(x)
    f_pos = (np.arange(1, n + 1) - 0.35) / n
    w0 = x.mean()
    w1 = float(((1.0 - f_pos) * x).mean())
    w2 = float(((1.0 - f_pos) ** 2 * x).mean())

    denom = 6.0 * w1 - w0 - 6.0 * w2
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return None
    beta = (2.0 * w1 - w0) / denom
    if not np.isfinite(beta) or beta <= 1.0:
        return None
    g1, g2 = gammafn(1.0 + 1.0 / beta), gammafn(1.0 - 1.0 / beta)
    alpha = (w0 - 2.0 * w1) * beta / (g1 * g2)
    if not np.isfinite(alpha) or alpha <= 0:
        return None
    gam = w0 - alpha * g1 * g2
    if not np.isfinite(gam) or gam >= x.min():
        return None
    return {'dist': 'fisk', 'c': float(beta), 'loc': float(gam), 'scale': float(alpha)}


def fit_month_dist(cal_vals, dist_type):
    cal_vals = np.asarray(cal_vals, float)
    cal_vals = cal_vals[np.isfinite(cal_vals)]
    if len(cal_vals) < 5:
        return None
    if dist_type == 'gamma':
        pos = cal_vals[cal_vals > 0]
        if len(pos) < 5:
            return None
        a, _, b = stats.gamma.fit(pos, floc=0)
        q0 = float((cal_vals == 0).mean())
        return {'dist': 'gamma', 'a': a, 'b': b, 'q0': q0}
    if dist_type == 'fisk':
        par = fit_fisk_pwm(cal_vals)
        if par is not None:
            return par
    return {'dist': 'norm', 'mu': float(cal_vals.mean()), 'sd': float(cal_vals.std(ddof=1))}


def eval_standardized_index(series, params):
    if params is None:
        return pd.Series(index=series.index, dtype=float)
    if params['dist'] == 'gamma':
        q0 = params['q0']
        cdf = q0 + (1.0 - q0) * stats.gamma.cdf(series.values, params['a'], loc=0, scale=params['b'])
    elif params['dist'] == 'fisk':
        cdf = stats.fisk.cdf(series.values, params['c'], loc=params['loc'], scale=params['scale'])
    else:
        cdf = stats.norm.cdf(series.values, params['mu'], params['sd'])
    cdf = np.clip(cdf, 1e-4, 1.0 - 1e-4)
    return pd.Series(stats.norm.ppf(cdf), index=series.index)


def compute_rdi_series(monthly_df, cal_start, cal_end, scale=6, fitted=None):
    p_roll = monthly_df['Precipitation'].rolling(scale).sum()
    pet_roll = monthly_df['PET'].rolling(scale).sum()
    ratio = (p_roll / pet_roll.clip(lower=1.0)).clip(lower=1e-3)
    y_log = np.log(ratio)

    if fitted is None:
        fitted = {}
        for m in range(1, 13):
            sub = y_log[(y_log.index.year >= cal_start) & (y_log.index.year <= cal_end) & (y_log.index.month == m)].dropna()
            fitted[m] = (float(sub.mean()), float(sub.std(ddof=1)))

    out = pd.Series(index=monthly_df.index, dtype=float)
    for m in range(1, 13):
        mask = monthly_df.index.month == m
        mu, sd = fitted[m]
        out.loc[mask] = (y_log.loc[mask] - mu) / (sd if sd > 0 else 1.0)
    return out, fitted


def add_drought_indices(monthly_df, cal_start, cal_end, scales, fitted=None, rdi_fitted=None):
    out = monthly_df.copy()
    if fitted is None:
        fitted = {}
        for s in scales:
            p_r = monthly_df['Precipitation'].rolling(s).sum()
            wb_r = monthly_df['WaterBalance'].rolling(s).sum()
            spi_p, spei_p = [], []
            for m in range(1, 13):
                cp = p_r[(p_r.index.year >= cal_start) & (p_r.index.year <= cal_end) & (p_r.index.month == m)].dropna()
                cwb = wb_r[(wb_r.index.year >= cal_start) & (wb_r.index.year <= cal_end) & (wb_r.index.month == m)].dropna()
                spi_p.append(fit_month_dist(cp, 'gamma'))
                spei_p.append(fit_month_dist(cwb, 'fisk'))
            fitted[s] = {'SPI': spi_p, 'SPEI': spei_p}

    for s in scales:
        p_r = monthly_df['Precipitation'].rolling(s).sum()
        wb_r = monthly_df['WaterBalance'].rolling(s).sum()
        spi_s = pd.Series(index=monthly_df.index, dtype=float)
        spei_s = pd.Series(index=monthly_df.index, dtype=float)
        for m in range(1, 13):
            mask = monthly_df.index.month == m
            spi_s.loc[mask] = eval_standardized_index(p_r[mask], fitted[s]['SPI'][m - 1])
            spei_s.loc[mask] = eval_standardized_index(wb_r[mask], fitted[s]['SPEI'][m - 1])
        out[f'SPI_{s}'] = spi_s
        out[f'SPEI_{s}'] = spei_s

    out['RDI_6'], rdi_fitted = compute_rdi_series(out, cal_start, cal_end, 6, rdi_fitted)
    return out, fitted, rdi_fitted


def aggregate_daily_to_monthly(df_daily, lat_deg):
    pet_vals = [
        hargreaves_daily(row['Tmin'], row['Tmax'], 0.5 * (row['Tmax'] + row['Tmin']), calc_fao56_ra(idx.dayofyear, lat_deg))
        for idx, row in df_daily.iterrows()
    ]
    pet_s = pd.Series(pet_vals, index=df_daily.index)
    m = pd.DataFrame({
        'Precipitation': df_daily['Precipitation'].resample('MS').sum(),
        'Tmax': df_daily['Tmax'].resample('MS').mean(),
        'Tmin': df_daily['Tmin'].resample('MS').mean(),
        'PET': pet_s.resample('MS').sum(),
    })
    m['Tmean'] = 0.5 * (m['Tmax'] + m['Tmin'])
    m['WaterBalance'] = m['Precipitation'] - m['PET']
    return m


def summarize_window_stats(df_m, df_d):
    ann_p = df_m['Precipitation'].groupby(df_m.index.year).sum()
    ann_pet = df_m['PET'].groupby(df_m.index.year).sum()
    ann_wb = df_m['WaterBalance'].groupby(df_m.index.year).sum()
    ann_tx = df_m['Tmax'].groupby(df_m.index.year).mean()
    ann_tn = df_m['Tmin'].groupby(df_m.index.year).mean()
    ann_tm = df_m['Tmean'].groupby(df_m.index.year).mean()
    ann_dtr = (df_m['Tmax'] - df_m['Tmin']).groupby(df_m.index.year).mean()

    jjas = df_m[df_m.index.month.isin([6, 7, 8, 9])]
    mam = df_m[df_m.index.month.isin([3, 4, 5])]
    ma = df_m[df_m.index.month.isin([3, 4])]

    jjas_p = jjas['Precipitation'].groupby(jjas.index.year).sum()
    mam_p = mam['Precipitation'].groupby(mam.index.year).sum()
    ma_tx = ma['Tmax'].groupby(ma.index.year).mean()

    d_mam = df_d[df_d.index.month.isin([3, 4, 5])]
    hot_mam = (d_mam['Tmax'] > 35.0).groupby(d_mam.index.year).sum()
    hot_ann = (df_d['Tmax'] > 35.0).groupby(df_d.index.year).sum()

    return {
        'P_mean': float(ann_p.mean()), 'P_sd': float(ann_p.std()),
        'JJAS_P_mean': float(jjas_p.mean()), 'JJAS_P_sd': float(jjas_p.std()),
        'MAM_P_mean': float(mam_p.mean()), 'MAM_P_sd': float(mam_p.std()),
        'PET_mean': float(ann_pet.mean()), 'PET_sd': float(ann_pet.std()),
        'WB_mean': float(ann_wb.mean()), 'WB_sd': float(ann_wb.std()),
        'Tmax_mean': float(ann_tx.mean()), 'Tmax_sd': float(ann_tx.std()),
        'Tmin_mean': float(ann_tn.mean()), 'Tmin_sd': float(ann_tn.std()),
        'Tmean_mean': float(ann_tm.mean()), 'Tmean_sd': float(ann_tm.std()),
        'DTR_mean': float(ann_dtr.mean()), 'DTR_sd': float(ann_dtr.std()),
        'MarApr_Tmax_mean': float(ma_tx.mean()), 'MarApr_Tmax_sd': float(ma_tx.std()),
        'HotDays35_MAM_yr': float(hot_mam.mean()),
        'HotDays35_Ann_yr': float(hot_ann.mean()),
        'SPEI3_le_m1p5': float((df_m['SPEI_3'].dropna() <= -1.5).mean() * 100.0),
        'SPEI6_le_m1p0': float((df_m['SPEI_6'].dropna() <= -1.0).mean() * 100.0),
        'SPEI6_le_m1p5': float((df_m['SPEI_6'].dropna() <= -1.5).mean() * 100.0),
        'SPEI6_le_m2p0': float((df_m['SPEI_6'].dropna() <= -2.0).mean() * 100.0),
        'SPEI12_le_m1p5': float((df_m['SPEI_12'].dropna() <= -1.5).mean() * 100.0),
        'SPI6_le_m1p5': float((df_m['SPI_6'].dropna() <= -1.5).mean() * 100.0),
        'RDI6_le_m1p5': float((df_m['RDI_6'].dropna() <= -1.5).mean() * 100.0),
    }


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    lat = cfg['domain']['centroid_lat']
    y0, y1 = cfg['periods']['baseline']
    scales = cfg['drought_indices']['scales']

    out_root = ROOT / cfg['paths']['output_dir']
    bc_dir = out_root / 'bias_corrected'
    idx_dir = out_root / 'indices'
    mon_dir = out_root / 'monthly'
    tab_dir = out_root / 'tables'
    for d in [idx_dir, mon_dir, tab_dir]:
        d.mkdir(parents=True, exist_ok=True)

    era5_d = pd.read_csv(ROOT / cfg['paths']['era5_daily'], parse_dates=['date'])
    era5_d = era5_d.rename(columns={'PrecipData': 'Precipitation', 'TMaxData': 'Tmax', 'TMinData': 'Tmin'})
    era5_d = era5_d.set_index('date').sort_index()
    era5_d = era5_d[~era5_d.index.duplicated()][['Precipitation', 'Tmax', 'Tmin']]

    era5_m = aggregate_daily_to_monthly(era5_d, lat)
    era5_m, _, _ = add_drought_indices(era5_m, y0, y1, scales)
    era5_m.to_csv(idx_dir / 'ERA5_daily.csv')

    # Hargreaves-Samani vs daylength-adjusted Thornthwaite sensitivity
    th_pet = thornthwaite_monthly(era5_m['Tmean'], lat)
    sens = pd.DataFrame({
        'Precipitation': era5_m['Precipitation'],
        'PET_Hargreaves': era5_m['PET'],
        'PET_Thornthwaite': th_pet,
        'PET_Hargreaves_minus_Thornthwaite': era5_m['PET'] - th_pet,
    })
    sens.to_csv(tab_dir / 'pet_sensitivity_monthly.csv')

    for gcm in cfg['cmip6']['candidate_models']:
        hist_d = pd.read_csv(bc_dir / f'{gcm}_historical_BC.csv', parse_dates=['Date']).set_index('Date').sort_index()
        hm = aggregate_daily_to_monthly(hist_d, lat)
        hm_cal = hm[(hm.index.year >= y0) & (hm.index.year <= y1)]
        hm_cal[['Precipitation', 'Tmax', 'Tmin']].to_csv(mon_dir / f'{gcm}_monthly.csv')

        hm_idx, fitted_par, rdi_par = add_drought_indices(hm_cal, y0, y1, scales)
        hm_idx.to_csv(idx_dir / f'{gcm}_historical_drought_indices.csv')

        for scen in ['ssp245', 'ssp585']:
            fut_d = pd.read_csv(bc_dir / f'{gcm}_{scen}_BC.csv', parse_dates=['Date']).set_index('Date').sort_index()
            fm = aggregate_daily_to_monthly(fut_d, lat)
            fm_idx, _, _ = add_drought_indices(fm, y0, y1, scales, fitted=fitted_par, rdi_fitted=rdi_par)
            fm_idx.to_csv(idx_dir / f'{gcm}_{scen}_drought_indices.csv')

    # Build multi-window climate & drought summary across Top-5 ensemble
    summary_rows = []
    for gcm in cfg['cmip6']['selected_top5']:
        hm = pd.read_csv(idx_dir / f'{gcm}_historical_drought_indices.csv', parse_dates=['Date']).set_index('Date')
        hd = pd.read_csv(bc_dir / f'{gcm}_historical_BC.csv', parse_dates=['Date']).set_index('Date')
        hd = hd[(hd.index.year >= y0) & (hd.index.year <= y1)]
        summary_rows.append({'GCM': gcm, 'Scenario': 'historical', 'Window': f'{y0}-{y1}', **summarize_window_stats(hm, hd)})

        for scen in ['ssp245', 'ssp585']:
            fm = pd.read_csv(idx_dir / f'{gcm}_{scen}_drought_indices.csv', parse_dates=['Date']).set_index('Date')
            fd = pd.read_csv(bc_dir / f'{gcm}_{scen}_BC.csv', parse_dates=['Date']).set_index('Date')
            for w_label, (w0, w1) in [('2015-2100', cfg['periods']['future_full']), ('2071-2100', cfg['periods']['future_late'])]:
                sub_m = fm[(fm.index.year >= w0) & (fm.index.year <= w1)]
                sub_d = fd[(fd.index.year >= w0) & (fd.index.year <= w1)]
                summary_rows.append({'GCM': gcm, 'Scenario': scen, 'Window': w_label, **summarize_window_stats(sub_m, sub_d)})

    pd.DataFrame(summary_rows).to_csv(tab_dir / 'comprehensive_climate_drought_summary.csv', index=False)
    print("Completed PET, drought indices, and seasonal/window summaries.")


if __name__ == '__main__':
    main()
