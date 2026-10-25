from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import yaml

ROOT = Path(__file__).resolve().parents[2]


def calc_kge(sim, obs):
    r = np.corrcoef(sim, obs)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def run_topsis(matrix, weights, impacts):
    norms = np.max(np.abs(matrix), axis=0)
    norms[norms == 0] = 1.0
    weighted = (matrix / norms) * weights

    ideal_best = np.zeros(matrix.shape[1])
    ideal_worst = np.zeros(matrix.shape[1])
    for j, sign in enumerate(impacts):
        if sign == '+':
            ideal_best[j] = weighted[:, j].max()
            ideal_worst[j] = weighted[:, j].min()
        else:
            ideal_best[j] = weighted[:, j].min()
            ideal_worst[j] = weighted[:, j].max()

    d_best = np.array([euclidean(row, ideal_best) for row in weighted])
    d_worst = np.array([euclidean(row, ideal_worst) for row in weighted])
    return d_worst / (d_best + d_worst + 1e-12)


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    y0, y1 = cfg['periods']['baseline']
    out_root = ROOT / cfg['paths']['output_dir']
    mon_dir = out_root / 'monthly'
    tab_dir = out_root / 'tables'
    tab_dir.mkdir(parents=True, exist_ok=True)

    era5 = pd.read_csv(out_root / 'indices' / 'ERA5_daily.csv', parse_dates=['date']).set_index('date')
    era5 = era5[(era5.index.year >= y0) & (era5.index.year <= y1)]
    o_p = era5['Precipitation'].values
    o_tx = era5['Tmax'].values
    o_tn = era5['Tmin'].values
    o_tm = 0.5 * (o_tx + o_tn)
    era5_p_mean = float(o_p.mean())

    rows = []
    for gcm in cfg['cmip6']['candidate_models']:
        f = mon_dir / f'{gcm}_monthly.csv'
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=['Date']).set_index('Date')
        df = df[(df.index.year >= y0) & (df.index.year <= y1)]

        s_p = df['Precipitation'].values
        s_tx = df['Tmax'].values
        s_tn = df['Tmin'].values
        s_tm = 0.5 * (s_tx + s_tn)

        p_bias = float(np.mean(s_p - o_p))
        tm_bias = float(np.mean(s_tm - o_tm))
        rows.append({
            'Model': gcm,
            'P_KGE': float(calc_kge(s_p, o_p)),
            'P_Bias': p_bias,
            'P_Bias_abs': abs(p_bias),
            'P_Bias_mm_mo': p_bias,
            'P_Bias_pct': (p_bias / era5_p_mean) * 100.0,
            'P_r': float(np.corrcoef(s_p, o_p)[0, 1]),
            'Tmax_KGE': float(calc_kge(s_tx, o_tx)),
            'Tmin_KGE': float(calc_kge(s_tn, o_tn)),
            'Tmean_Bias': tm_bias,
            'Tmean_Bias_abs': abs(tm_bias),
        })

    df_rk = pd.DataFrame(rows)
    crit = cfg['topsis']['criteria']
    weights = np.array(cfg['topsis']['weights'], dtype=float)
    impacts = cfg['topsis']['impacts']

    df_rk['TOPSIS_Score'] = run_topsis(df_rk[crit].values, weights, impacts)
    df_rk['Rank'] = df_rk['TOPSIS_Score'].rank(ascending=False).astype(int)
    df_rk = df_rk.sort_values('Rank').reset_index(drop=True)

    df_rk.to_csv(tab_dir / 'gcm_topsis_ranking_complete.csv', index=False)
    print(df_rk[['Rank', 'Model', 'TOPSIS_Score', 'P_KGE', 'P_Bias_mm_mo', 'Tmax_KGE', 'Tmin_KGE']].to_string(index=False))


if __name__ == '__main__':
    main()
