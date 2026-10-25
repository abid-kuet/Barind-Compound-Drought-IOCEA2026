from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import yaml

ROOT = Path(__file__).resolve().parents[2]


def pseudo_obs(x):
    return stats.rankdata(x, method='average') / (len(x) + 1.0)


def gauss_nll(rho, u, v):
    rho = float(np.ravel(rho)[0])
    if abs(rho) >= 0.999:
        return 1e10
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    den = 1.0 - rho ** 2
    return 0.5 * len(u) * np.log(den) + 0.5 * np.sum((rho ** 2 * (z1 ** 2 + z2 ** 2) - 2.0 * rho * z1 * z2) / den)


def clayton_nll(theta, u, v):
    t = float(np.ravel(theta)[0])
    if t <= 1e-6:
        return 1e10
    s = u ** (-t) + v ** (-t) - 1.0
    if np.any(s <= 0):
        return 1e10
    logc = np.log1p(t) - (t + 1.0) * (np.log(u) + np.log(v)) - (1.0 / t + 2.0) * np.log(s)
    return -np.sum(logc) if np.all(np.isfinite(logc)) else 1e10


def gumbel_nll(theta, u, v):
    t = float(np.ravel(theta)[0])
    if t < 1.0:
        return 1e10
    if abs(t - 1.0) < 1e-9:
        return 0.0
    x, y = -np.log(u), -np.log(v)
    a_val = (x ** t + y ** t) ** (1.0 / t)
    logc = -a_val - np.log(u) - np.log(v) + (t - 1.0) * (np.log(x) + np.log(y)) + (1.0 - 2.0 * t) * np.log(a_val) + np.log(a_val + t - 1.0)
    return -np.sum(logc) if np.all(np.isfinite(logc)) else 1e10


def frank_nll(theta, u, v):
    t = float(np.ravel(theta)[0])
    if abs(t) < 1e-5:
        return 0.0
    num = t * (1.0 - np.exp(-t)) * np.exp(-t * (u + v))
    den = ((1.0 - np.exp(-t)) - (1.0 - np.exp(-t * u)) * (1.0 - np.exp(-t * v))) ** 2
    if np.any(den <= 0) or np.any(num <= 0):
        return 1e10
    logc = np.log(num) - np.log(den)
    return -np.sum(logc) if np.all(np.isfinite(logc)) else 1e10


def joe_nll(theta, u, v):
    t = float(np.ravel(theta)[0])
    if t < 1.0:
        return 1e10
    if abs(t - 1.0) < 1e-9:
        return 0.0
    ub, vb = 1.0 - u, 1.0 - v
    k_val = ub ** t + vb ** t - (ub * vb) ** t
    if np.any(k_val <= 0):
        return 1e10
    logc = (1.0 / t - 2.0) * np.log(k_val) + (t - 1.0) * (np.log(ub) + np.log(vb)) + np.log(t - 1.0 + k_val)
    return -np.sum(logc) if np.all(np.isfinite(logc)) else 1e10


def fit_copula_families(u, v, families):
    specs = {
        'gaussian': (gauss_nll, [0.2], [(-0.99, 0.99)]),
        'clayton': (clayton_nll, [0.5], [(1e-4, 20.0)]),
        'gumbel': (gumbel_nll, [1.3], [(1.0, 20.0)]),
        'frank': (frank_nll, [1.5], [(-30.0, 30.0)]),
        'joe': (joe_nll, [1.3], [(1.0, 20.0)]),
    }
    fits = []
    for fam in families:
        fn, x0, bnds = specs[fam]
        res = minimize(fn, x0, args=(u, v), bounds=bnds, method='L-BFGS-B')
        nll = float(res.fun)
        aic = 2.0 * 1.0 + 2.0 * nll
        fits.append({'family': fam, 'theta': float(res.x[0]), 'nll': nll, 'aic': aic})
    fits.sort(key=lambda d: d['aic'])
    return fits[0], fits


def eval_copula_cdf(family, theta, u, v):
    u = float(np.clip(u, 1e-6, 1.0 - 1e-6))
    v = float(np.clip(v, 1e-6, 1.0 - 1e-6))
    fam = family.lower()
    if fam == 'gaussian':
        z1, z2 = stats.norm.ppf(u), stats.norm.ppf(v)
        return float(stats.multivariate_normal.cdf([z1, z2], mean=[0.0, 0.0], cov=[[1.0, theta], [theta, 1.0]]))
    if fam == 'clayton':
        return float((u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta))
    if fam == 'gumbel':
        return float(np.exp(-(((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1.0 / theta))))
    if fam == 'frank':
        num = (np.exp(-theta * u) - 1.0) * (np.exp(-theta * v) - 1.0)
        return float(-np.log(1.0 + num / (np.exp(-theta) - 1.0)) / theta)
    if fam == 'joe':
        ub, vb = 1.0 - u, 1.0 - v
        return float(1.0 - (ub ** theta + vb ** theta - (ub * vb) ** theta) ** (1.0 / theta))
    return u * v


def kendall_pval_neff(tau, n_eff):
    if n_eff <= 4:
        return 1.0
    var_tau = (2.0 * (2.0 * n_eff + 5.0)) / (9.0 * n_eff * (n_eff - 1.0))
    z = abs(tau) / np.sqrt(var_tau)
    return float(2.0 * stats.norm.sf(z))


def count_episodes(ev_bool):
    ev = np.asarray(ev_bool, dtype=int)
    if len(ev) == 0:
        return 0, 0.0
    runs = np.diff(np.concatenate(([0], ev, [0])))
    starts = np.where(runs == 1)[0]
    ends = np.where(runs == -1)[0]
    n_ep = len(starts)
    return n_ep, (float(np.mean(ends - starts)) if n_ep > 0 else 0.0)


def block_bootstrap_ci(ev_bool, block_len=12, n_boot=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    ind = np.asarray(ev_bool, dtype=float)
    n = len(ind)
    n_blocks = int(np.ceil(n / block_len))
    starts = np.arange(0, n - block_len + 1)
    rps = []
    for _ in range(n_boot):
        idx = rng.choice(starts, size=n_blocks, replace=True)
        samp = np.concatenate([ind[i:i + block_len] for i in idx])[:n]
        k = samp.sum()
        if k > 0:
            rps.append((n / 12.0) / k)
    if not rps:
        return np.nan, np.nan
    return float(np.percentile(rps, 2.5)), float(np.percentile(rps, 97.5))


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    y0, y1 = cfg['periods']['baseline']
    f0, f1 = cfg['periods']['future_full']
    top5 = cfg['cmip6']['selected_top5']
    families = cfg['copula']['families']
    block_len = cfg['copula']['block_bootstrap_len']
    n_boot = cfg['copula']['n_boot']

    out_root = ROOT / cfg['paths']['output_dir']
    idx_dir = out_root / 'indices'
    tab_dir = out_root / 'tables'
    tab_dir.mkdir(parents=True, exist_ok=True)

    era5 = pd.read_csv(idx_dir / 'ERA5_daily.csv', parse_dates=['date']).set_index('date')
    era5 = era5[(era5.index.year >= y0) & (era5.index.year <= y1)]
    tmax_p90_fixed = float(era5['Tmax'].quantile(0.90))
    tmax_p90_month = era5.groupby(era5.index.month)['Tmax'].quantile(0.90).to_dict()
    tmax_mean_month = era5.groupby(era5.index.month)['Tmax'].mean().to_dict()

    hist_t90_gcm = {}
    for gcm in top5:
        h = pd.read_csv(idx_dir / f'{gcm}_historical_drought_indices.csv', parse_dates=['Date']).set_index('Date')
        h = h[(h.index.year >= y0) & (h.index.year <= y1)][['Tmax', 'SPEI_6']].dropna()
        hist_t90_gcm[gcm] = float(np.quantile(h['Tmax'], 0.90))

    rng = np.random.default_rng(42)
    diag_rows, fam_rows = [], []

    for scen in ['historical', 'ssp245', 'ssp585']:
        gcm_frames = []
        for gcm in top5:
            df = pd.read_csv(idx_dir / f'{gcm}_{scen}_drought_indices.csv', parse_dates=['Date']).set_index('Date')
            df = df[(df.index.year >= y0) & (df.index.year <= y1)] if scen == 'historical' else df[(df.index.year >= f0) & (df.index.year <= f1)]
            df = df[['Tmax', 'SPEI_6']].dropna().copy()
            df['Tmax_anom'] = [r['Tmax'] - tmax_mean_month[dt.month] for dt, r in df.iterrows()]
            df['Hot_Fixed'] = df['Tmax'] > tmax_p90_fixed
            df['Hot_CalMonth'] = [r['Tmax'] > tmax_p90_month[dt.month] for dt, r in df.iterrows()]
            gcm_frames.append((gcm, df))

            u = pseudo_obs(df['Tmax'].values)
            v = pseudo_obs(-df['SPEI_6'].values)
            best, all_fits = fit_copula_families(u, v, families)
            for f_item in all_fits:
                fam_rows.append({'GCM': gcm, 'Scenario': scen, 'Family': f_item['family'], 'Theta': f_item['theta'], 'AIC': f_item['aic'], 'Selected': f_item['family'] == best['family']})

            tau, p_tau = stats.kendalltau(u, v)
            r1 = float(pd.Series(df['SPEI_6'].values).autocorr(lag=1))
            n_eff = len(df) * (1.0 - r1) / (1.0 + r1)
            p_neff = kendall_pval_neff(float(tau), n_eff)

            u_an = pseudo_obs(df['Tmax_anom'].values)
            best_an, _ = fit_copula_families(u_an, v, families)
            tau_an, p_an = stats.kendalltau(u_an, v)

            ev_fix = df['Hot_Fixed'] & (df['SPEI_6'] <= -1.5)
            ev_cm = df['Hot_CalMonth'] & (df['SPEI_6'] <= -1.5)
            df_mam = df[df.index.month.isin([3, 4, 5])]
            ev_mam = (df_mam['Tmax'] > tmax_p90_fixed) & (df_mam['SPEI_6'] <= -1.0)

            rp_fix = (len(df) / 12.0) / ev_fix.sum() if ev_fix.sum() > 0 else np.inf
            ci_lo, ci_hi = block_bootstrap_ci(ev_fix.values, block_len=block_len, n_boot=n_boot, rng=rng)
            n_ep, dur_ep = count_episodes(ev_fix.values)
            rp_cm = (len(df) / 12.0) / ev_cm.sum() if ev_cm.sum() > 0 else np.inf

            t90_g = hist_t90_gcm[gcm]
            u0 = float(np.clip((df['Tmax'] <= t90_g).mean(), 0.001, 0.999))
            v0 = float(np.clip((-df['SPEI_6'] <= 1.5).mean(), 0.001, 0.999))
            c_uv = eval_copula_cdf(best['family'], best['theta'], u0, v0)
            rp_param = (1.0 / 12.0) / max(1.0 - u0 - v0 + c_uv, 1e-6)

            diag_rows.append({
                'GCM': gcm, 'Scenario': scen, 'N_Months': len(df),
                'Lag1_r1_SPEI6': r1, 'N_eff': n_eff,
                'Copula_Family': best['family'], 'Theta': best['theta'],
                'Kendall_Tau': float(tau), 'Tau_pvalue': float(p_tau), 'Tau_pvalue_Neff': p_neff,
                'AIC': best['aic'], 'AIC_per_N': best['aic'] / len(df),
                'N_Events_Fixed': int(ev_fix.sum()), 'N_Episodes_Declustered': n_ep, 'Mean_Episode_Dur_mo': dur_ep,
                'RP_Fixed_yr': rp_fix, 'RP_Parametric_Copula_yr': rp_param,
                'CI_Block_Lo': ci_lo, 'CI_Block_Hi': ci_hi,
                'Deseas_Family': best_an['family'], 'Deseas_Theta': best_an['theta'],
                'Deseas_Tau': float(tau_an), 'Deseas_Tau_p': float(p_an),
                'N_Events_CalMonth': int(ev_cm.sum()), 'RP_CalMonth_yr': rp_cm,
                'MAM_Compound_Pct': float(ev_mam.mean() * 100.0),
            })

        # Pooled ensemble across the 5 GCMs (block bootstrap resampled strictly within each GCM)
        pooled = pd.concat([d for _, d in gcm_frames], axis=0)
        u_p = pseudo_obs(pooled['Tmax'].values)
        v_p = pseudo_obs(-pooled['SPEI_6'].values)
        best_p, all_p = fit_copula_families(u_p, v_p, families)
        for f_item in all_p:
            fam_rows.append({'GCM': 'ENSEMBLE', 'Scenario': scen, 'Family': f_item['family'], 'Theta': f_item['theta'], 'AIC': f_item['aic'], 'Selected': f_item['family'] == best_p['family']})

        tau_p, p_tau_p = stats.kendalltau(u_p, v_p)
        r1_list = [float(pd.Series(d['SPEI_6'].values).autocorr(lag=1)) for _, d in gcm_frames]
        neff_tot = float(sum(len(d) * (1.0 - rg) / (1.0 + rg) for (_, d), rg in zip(gcm_frames, r1_list)))
        p_neff_p = kendall_pval_neff(float(tau_p), neff_tot)

        u_an_p = pseudo_obs(pooled['Tmax_anom'].values)
        best_an_p, _ = fit_copula_families(u_an_p, v_p, families)
        tau_an_p, p_an_p = stats.kendalltau(u_an_p, v_p)

        ev_fix_p = pooled['Hot_Fixed'] & (pooled['SPEI_6'] <= -1.5)
        ev_cm_p = pooled['Hot_CalMonth'] & (pooled['SPEI_6'] <= -1.5)
        mam_p = pooled[pooled.index.month.isin([3, 4, 5])]
        ev_mam_p = (mam_p['Tmax'] > tmax_p90_fixed) & (mam_p['SPEI_6'] <= -1.0)

        rp_fix_p = (len(pooled) / 12.0) / ev_fix_p.sum()
        n_ep_p, dur_ep_p = count_episodes(ev_fix_p.values)
        rp_cm_p = (len(pooled) / 12.0) / ev_cm_p.sum()

        u0_p = float(np.clip(np.mean([(d['Tmax'] <= hist_t90_gcm[g]).mean() for g, d in gcm_frames]), 0.001, 0.999))
        v0_p = float(np.clip(np.mean([(-d['SPEI_6'] <= 1.5).mean() for _, d in gcm_frames]), 0.001, 0.999))
        c_uv_p = eval_copula_cdf(best_p['family'], best_p['theta'], u0_p, v0_p)
        rp_param_p = (1.0 / 12.0) / max(1.0 - u0_p - v0_p + c_uv_p, 1e-6)

        rps_boot = []
        for _ in range(n_boot):
            tot_k, tot_n = 0.0, 0
            for g, d in gcm_frames:
                ind_g = ((d['Tmax'] > hist_t90_gcm[g]) & (d['SPEI_6'] <= -1.5)).values.astype(float)
                ng = len(ind_g)
                nb = int(np.ceil(ng / block_len))
                idx = rng.choice(np.arange(0, ng - block_len + 1), size=nb, replace=True)
                samp = np.concatenate([ind_g[i:i + block_len] for i in idx])[:ng]
                tot_k += samp.sum()
                tot_n += ng
            if tot_k > 0:
                rps_boot.append((tot_n / 12.0) / tot_k)
        ci_lo_p, ci_hi_p = float(np.percentile(rps_boot, 2.5)), float(np.percentile(rps_boot, 97.5))

        diag_rows.append({
            'GCM': 'ENSEMBLE', 'Scenario': scen, 'N_Months': len(pooled),
            'Lag1_r1_SPEI6': float(np.mean(r1_list)), 'N_eff': neff_tot,
            'Copula_Family': best_p['family'], 'Theta': best_p['theta'],
            'Kendall_Tau': float(tau_p), 'Tau_pvalue': float(p_tau_p), 'Tau_pvalue_Neff': p_neff_p,
            'AIC': best_p['aic'], 'AIC_per_N': best_p['aic'] / len(pooled),
            'N_Events_Fixed': int(ev_fix_p.sum()), 'N_Episodes_Declustered': n_ep_p, 'Mean_Episode_Dur_mo': dur_ep_p,
            'RP_Fixed_yr': rp_fix_p, 'RP_Parametric_Copula_yr': rp_param_p,
            'CI_Block_Lo': ci_lo_p, 'CI_Block_Hi': ci_hi_p,
            'Deseas_Family': best_an_p['family'], 'Deseas_Theta': best_an_p['theta'],
            'Deseas_Tau': float(tau_an_p), 'Deseas_Tau_p': float(p_an_p),
            'N_Events_CalMonth': int(ev_cm_p.sum()), 'RP_CalMonth_yr': rp_cm_p,
            'MAM_Compound_Pct': float(ev_mam_p.mean() * 100.0),
        })

    pd.DataFrame(fam_rows).to_csv(tab_dir / 'copula_family_selection.csv', index=False)
    pd.DataFrame(diag_rows).to_csv(tab_dir / 'copula_enhanced_diagnostics.csv', index=False)
    print("Completed bivariate copula modeling and return period diagnostics.")


if __name__ == '__main__':
    main()
