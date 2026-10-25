from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
VARS = ['Precipitation', 'Tmax', 'Tmin']
IS_MULT = {'Precipitation': True, 'Tmax': False, 'Tmin': False}
WET_THRESH = 0.1
MAX_DELTA = 10.0
N_ROTATIONS = 30
RNG_SEED = 42


def inv_ecdf(sorted_x, tau):
    n = len(sorted_x)
    pp = (np.arange(1, n + 1) - 0.5) / n
    return np.interp(tau, pp, sorted_x)


def nonexceedance_prob(x):
    return (rankdata(x, method='average') - 0.5) / len(x)


def qdm_precip(obs_h, mod_h, mod_p, thresh=WET_THRESH):
    # Separate wet-day frequency adaptation (Themessl et al. 2012) from ratio QDM intensity
    o_wet = np.sort(obs_h[obs_h >= thresh])
    h_wet = np.sort(mod_h[mod_h >= thresh])
    if len(o_wet) < 5 or len(h_wet) < 5:
        return mod_p.copy()

    f_o = float((obs_h < thresh).mean())
    f_h = float((mod_h < thresh).mean())
    f_p = float((mod_p < thresh).mean())

    n = len(mod_p)
    wet = mod_p >= thresh
    n_wet_mod = int(wet.sum())

    f_target = float(np.clip(f_o * (f_p / f_h) if f_h > 0 else f_o, 0.0, 0.98))
    n_dry_target = int(round(f_target * n))
    n_wet_target = n - n_dry_target
    n_promote = max(0, n_wet_target - n_wet_mod)
    tau_lo = n_promote / n_wet_target if n_wet_target > 0 else 0.0

    out = np.zeros(n, dtype=float)
    if n_wet_mod > 0:
        tau_w = nonexceedance_prob(mod_p[wet])
        q_ow = inv_ecdf(o_wet, tau_lo + tau_w * (1.0 - tau_lo))
        q_hw = inv_ecdf(h_wet, tau_w)
        denom = np.where(q_hw < 1e-3, 1e-3, q_hw)
        delta = np.clip(mod_p[wet] / denom, 0.0, MAX_DELTA)
        out[wet] = np.clip(q_ow * delta, 0.0, None)

    if n_promote > 0:
        dry_idx = np.where(~wet)[0]
        n_promote = min(n_promote, len(dry_idx))
        order = dry_idx[np.argsort(-mod_p[dry_idx], kind='stable')][:n_promote]
        taus = (np.arange(1, n_promote + 1) - 0.5) / n_promote * tau_lo
        out[order] = inv_ecdf(o_wet, taus)[::-1]
    elif n_wet_target < n_wet_mod:
        n_drop = n_wet_mod - n_wet_target
        wet_idx = np.where(wet)[0]
        order = wet_idx[np.argsort(out[wet_idx], kind='stable')][:n_drop]
        out[order] = 0.0

    return out


def qdm_1d(obs_h, mod_h, mod_p, multiplicative=False):
    obs_h = np.asarray(obs_h, float)
    mod_h = np.asarray(mod_h, float)
    mod_p = np.asarray(mod_p, float)
    if len(obs_h) < 5 or len(mod_h) < 5 or len(mod_p) == 0:
        return mod_p.copy()
    if multiplicative:
        return qdm_precip(obs_h, mod_h, mod_p)

    tau = nonexceedance_prob(mod_p)
    q_oh = inv_ecdf(np.sort(obs_h), tau)
    q_mh = inv_ecdf(np.sort(mod_h), tau)
    return q_oh + (mod_p - q_mh)


def random_rotation_matrix(k, rng):
    mat = rng.standard_normal((k, k))
    q, r = np.linalg.qr(mat)
    return q * np.sign(np.diag(r))


def mbcn_core(obs, mod_h, mod_p, n_iter=N_ROTATIONS, seed=RNG_SEED):
    obs = np.asarray(obs, float)
    mod_h = np.asarray(mod_h, float)
    mod_p = np.asarray(mod_p, float)
    k = obs.shape[1]
    mult_flags = [IS_MULT[v] for v in VARS]

    qdm_h = np.column_stack([qdm_1d(obs[:, j], mod_h[:, j], mod_h[:, j], mult_flags[j]) for j in range(k)])
    qdm_p = np.column_stack([qdm_1d(obs[:, j], mod_h[:, j], mod_p[:, j], mult_flags[j]) for j in range(k)])
    if len(mod_p) < 10 or len(obs) < 10:
        return qdm_p

    mu = obs.mean(axis=0)
    sd = obs.std(axis=0)
    sd[sd == 0] = 1.0
    o_std = (obs - mu) / sd
    h_std = (qdm_h - mu) / sd
    p_std = (qdm_p - mu) / sd

    rng = np.random.default_rng(seed)
    for _ in range(n_iter):
        rot = random_rotation_matrix(k, rng)
        o_r, h_r, p_r = o_std @ rot, h_std @ rot, p_std @ rot
        h_next = np.empty_like(h_r)
        p_next = np.empty_like(p_r)
        for j in range(k):
            h_next[:, j] = qdm_1d(o_r[:, j], h_r[:, j], h_r[:, j], False)
            p_next[:, j] = qdm_1d(o_r[:, j], h_r[:, j], p_r[:, j], False)
        h_std = h_next @ rot.T
        p_std = p_next @ rot.T

    out = np.empty_like(qdm_p)
    for j in range(k):
        ranks = np.argsort(np.argsort(p_std[:, j]))
        out[:, j] = np.sort(qdm_p[:, j])[ranks]
    return out


def apply_mbcn_monthly(obs_df, mod_h_df, mod_p_df):
    chunks = []
    for m in range(1, 13):
        o = obs_df[obs_df.index.month == m][VARS].values
        h = mod_h_df[mod_h_df.index.month == m][VARS].values
        p_idx = mod_p_df.index[mod_p_df.index.month == m]
        p = mod_p_df.loc[p_idx, VARS].values
        if len(p) == 0:
            continue
        if len(o) < 10 or len(h) < 10:
            chunks.append(pd.DataFrame(p, index=p_idx, columns=VARS))
            continue
        corr = mbcn_core(o, h, p, seed=RNG_SEED + m)
        chunks.append(pd.DataFrame(corr, index=p_idx, columns=VARS))

    out = pd.concat(chunks).sort_index()
    out['Precipitation'] = out['Precipitation'].clip(lower=0.0)
    return out


def calc_kge(sim, obs):
    sim = np.asarray(sim, float)
    obs = np.asarray(obs, float)
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def resample_monthly(df):
    return pd.DataFrame({
        'Precipitation': df['Precipitation'].resample('MS').sum(),
        'Tmax': df['Tmax'].resample('MS').mean(),
        'Tmin': df['Tmin'].resample('MS').mean(),
    })


def evaluate_skill(gcm, stage, mod_d, obs_d, obs_m, include_ks=False):
    common_d = mod_d.index.intersection(obs_d.index)
    md = mod_d.loc[common_d]
    mm = resample_monthly(md)
    common_m = mm.index.intersection(obs_m.index)
    mm, om = mm.loc[common_m], obs_m.loc[common_m]

    row = {'GCM': gcm, 'Stage': stage}
    for v in VARS:
        row[f'{v}_KGE_monthly'] = float(calc_kge(mm[v], om[v]))
        row[f'{v}_bias_monthly'] = float(mm[v].mean() - om[v].mean())
    row['P_annual_mm'] = float(mm['Precipitation'].groupby(mm.index.year).sum().mean())
    row['P_maxdaily_mm'] = float(md['Precipitation'].max())
    row['P_drydayfrac'] = float((md['Precipitation'] == 0).mean())
    row['spearman_P_Tmax'] = float(pd.Series(md['Precipitation']).corr(md['Tmax'], method='spearman'))

    if include_ks:
        for v in VARS:
            ks_stat, _ = stats.ks_2samp(md[v].dropna(), obs_d.loc[common_d, v].dropna())
            row[f'{v}_KS_daily'] = float(ks_stat)
    return row


def load_era5_reference(path):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.rename(columns={'PrecipData': 'Precipitation', 'TMaxData': 'Tmax', 'TMinData': 'Tmin'})
    df = df.set_index('date').sort_index()
    return df[~df.index.duplicated()][VARS]


def load_gcm_sheet(gcm, sheet, cmip6_dir):
    f = Path(cmip6_dir) / f'{gcm}_Rajshahi.xlsx'
    df = pd.ExcelFile(f).parse(sheet)
    df['Date'] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day))
    df = df.set_index('Date').sort_index()
    df = df.rename(columns={'PrecipData': 'Precipitation', 'TMaxData': 'Tmax', 'TMinData': 'Tmin'})
    return df[VARS]


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    y_start, y_end = cfg['periods']['baseline']
    cal_s0, cal_s1 = cfg['periods']['split_cal']
    val_s0, val_s1 = cfg['periods']['split_val']

    out_root = ROOT / cfg['paths']['output_dir']
    bc_dir = out_root / 'bias_corrected'
    tab_dir = out_root / 'tables'
    bc_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    era5 = load_era5_reference(ROOT / cfg['paths']['era5_daily'])
    era5_cal = era5[(era5.index.year >= y_start) & (era5.index.year <= y_end)]
    era5_cal_m = resample_monthly(era5_cal)

    era5_split_cal = era5_cal[era5_cal.index.year <= cal_s1]
    era5_split_val = era5_cal[era5_cal.index.year >= val_s0]
    era5_split_val_m = resample_monthly(era5_split_val)

    val_rows, split_rows, trend_rows = [], [], []

    for gcm in cfg['cmip6']['candidate_models']:
        raw_hist = load_gcm_sheet(gcm, 'historical', ROOT / cfg['paths']['cmip6_dir'])
        raw_cal = raw_hist[(raw_hist.index.year >= y_start) & (raw_hist.index.year <= y_end)]
        val_rows.append(evaluate_skill(gcm, 'raw', raw_cal, era5_cal, era5_cal_m))

        # Split-sample out-of-sample check (1986-2000 calibration -> 2001-2014 validation)
        r_scal = raw_hist[(raw_hist.index.year >= cal_s0) & (raw_hist.index.year <= cal_s1)]
        r_sval = raw_hist[(raw_hist.index.year >= val_s0) & (raw_hist.index.year <= val_s1)]
        bc_sval = apply_mbcn_monthly(era5_split_cal, r_scal, r_sval)
        split_rows.append(evaluate_skill(gcm, 'raw_oos', r_sval, era5_split_val, era5_split_val_m, include_ks=True))
        split_rows.append(evaluate_skill(gcm, 'MBCn_oos', bc_sval, era5_split_val, era5_split_val_m, include_ks=True))

        # Direct 1986-2014 self-calibrated reference annual mean
        bc_self = apply_mbcn_monthly(era5_cal, raw_cal, raw_cal)
        self_cal_ann = float(bc_self['Precipitation'].resample('YS').sum().mean())

        raw_cal_ann = float(raw_cal['Precipitation'].resample('YS').sum().mean())
        bc_cal_ann = None
        bc_cal_tmax = None

        for scen in ['historical', 'ssp245', 'ssp585']:
            target = raw_hist if scen == 'historical' else load_gcm_sheet(gcm, scen, ROOT / cfg['paths']['cmip6_dir'])
            corrected = apply_mbcn_monthly(era5_cal, raw_cal, target)
            corrected.to_csv(bc_dir / f'{gcm}_{scen}_BC.csv', index_label='Date')

            if scen == 'historical':
                bc_cal = corrected[(corrected.index.year >= y_start) & (corrected.index.year <= y_end)]
                val_rows.append(evaluate_skill(gcm, 'MBCn', bc_cal, era5_cal, era5_cal_m))
                bc_cal_ann = float(bc_cal['Precipitation'].resample('YS').sum().mean())
                bc_cal_tmax = float(bc_cal['Tmax'].mean())
            else:
                c_ann = float(corrected['Precipitation'].resample('YS').sum().mean())
                r_ann = float(target['Precipitation'].resample('YS').sum().mean())
                trend_rows.append({
                    'GCM': gcm,
                    'Scenario': scen,
                    'P_ratio_raw': r_ann / raw_cal_ann,
                    'P_ratio_corrected': c_ann / bc_cal_ann,
                    'P_cal_self_mm': self_cal_ann,
                    'P_ratio_vs_selfcal': c_ann / self_cal_ann,
                    'dTmax_raw': float(target['Tmax'].mean() - raw_cal['Tmax'].mean()),
                    'dTmax_corrected': float(corrected['Tmax'].mean() - bc_cal_tmax),
                    'P_annual_corrected': c_ann,
                    'N_days': len(corrected),
                })

    pd.DataFrame(val_rows).to_csv(tab_dir / 'bias_correction_validation.csv', index=False)
    pd.DataFrame(split_rows).to_csv(tab_dir / 'mbcn_split_sample_validation.csv', index=False)
    pd.DataFrame(trend_rows).to_csv(tab_dir / 'bias_correction_trend_check.csv', index=False)
    print("Completed MBCn bias correction and split-sample validation.")


if __name__ == '__main__':
    main()
