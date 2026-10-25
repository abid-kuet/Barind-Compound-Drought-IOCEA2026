from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def classify_state(val, thresholds):
    if val >= thresholds[0]:
        return 0
    if val >= thresholds[1]:
        return 1
    if val >= thresholds[2]:
        return 2
    return 3


def estimate_chain(spei_series, thresholds, state_names, n_boot=1000, conf=0.95):
    states = spei_series.apply(lambda x: classify_state(x, thresholds)).astype(int)
    pairs = pd.DataFrame({'s0': states, 's1': states.shift(-1)}).dropna()

    k = len(state_names)
    s0 = pairs['s0'].values.astype(int)
    s1 = pairs['s1'].values.astype(int)

    counts = np.bincount(s0 * k + s1, minlength=k ** 2).reshape(k, k)
    counts_df = pd.DataFrame(counts, index=state_names, columns=state_names)
    row_totals = counts_df.sum(axis=1)
    probs_df = counts_df.div(row_totals, axis=0).fillna(0.0)

    n = len(pairs)
    boot_arr = np.zeros((n_boot, k, k))
    for b in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        bc = np.bincount(s0[idx] * k + s1[idx], minlength=k ** 2).reshape(k, k).astype(float)
        bs = bc.sum(axis=1)
        bs[bs == 0] = 1.0
        boot_arr[b] = bc / bs[:, np.newaxis]

    alpha = 0.5 * (1.0 - conf)
    ci_lo = pd.DataFrame(np.percentile(boot_arr, alpha * 100.0, axis=0), index=state_names, columns=state_names)
    ci_hi = pd.DataFrame(np.percentile(boot_arr, (1.0 - alpha) * 100.0, axis=0), index=state_names, columns=state_names)
    return counts_df, probs_df, ci_lo, ci_hi, int(row_totals.sum())


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    np.random.seed(cfg['xgboost']['seed'])
    out_root = ROOT / cfg['paths']['output_dir']
    idx_dir = out_root / 'indices'
    tab_dir = out_root / 'tables'
    tab_dir.mkdir(parents=True, exist_ok=True)

    top5 = cfg['cmip6']['selected_top5']
    wmo_t = cfg['markov']['wmo_thresholds']
    usdm_t = cfg['markov']['usdm_thresholds']
    labels = cfg['markov']['state_names']
    n_boot = cfg['markov']['n_boot']
    conf = cfg['markov']['conf_level']

    trans_rows, sum_rows, usdm_rows = [], [], []

    for gcm in top5:
        for scen in ['historical', 'ssp245', 'ssp585']:
            f = idx_dir / f'{gcm}_{scen}_drought_indices.csv'
            if not f.exists():
                continue
            df = pd.read_csv(f, parse_dates=['Date']).set_index('Date')
            spei = df['SPEI_6'].dropna()

            cnt, pr, clo, chi, n_tr = estimate_chain(spei, wmo_t, labels, n_boot=n_boot, conf=conf)
            for i, fs in enumerate(labels):
                for j, ts in enumerate(labels):
                    trans_rows.append({
                        'GCM': gcm, 'Scenario': scen,
                        'From_State': fs, 'To_State': ts,
                        'Count': int(cnt.iloc[i, j]),
                        'Probability': float(pr.iloc[i, j]),
                        'CI_Lower': float(clo.iloc[i, j]),
                        'CI_Upper': float(chi.iloc[i, j]),
                    })

            sum_rows.append({
                'GCM': gcm, 'Scenario': scen, 'N_Transitions': n_tr,
                'Normal_to_Severe': float(pr.loc['Normal', 'Severe']),
                'Normal_to_Severe_CI_Lower': float(clo.loc['Normal', 'Severe']),
                'Normal_to_Severe_CI_Upper': float(chi.loc['Normal', 'Severe']),
                'Severe_to_Severe': float(pr.loc['Severe', 'Severe']),
                'Severe_to_Severe_CI_Lower': float(clo.loc['Severe', 'Severe']),
                'Severe_to_Severe_CI_Upper': float(chi.loc['Severe', 'Severe']),
                'N_Severe_Origin': int(cnt.loc['Severe'].sum()),
                'N_Normal_Origin': int(cnt.loc['Normal'].sum()),
            })

            _, pr_u, _, _, _ = estimate_chain(spei, usdm_t, labels, n_boot=100, conf=conf)
            usdm_rows.append({
                'GCM': gcm, 'Scenario': scen,
                'NonDrought_to_ModSev': float(pr_u.loc['Normal', 'Moderate'] + pr_u.loc['Normal', 'Severe']),
                'NonDrought_to_Severe': float(pr_u.loc['Normal', 'Severe']),
                'Mild_to_Severe': float(pr_u.loc['Mild', 'Severe']),
                'Moderate_to_Severe': float(pr_u.loc['Moderate', 'Severe']),
                'Severe_to_Severe': float(pr_u.loc['Severe', 'Severe']),
            })

    pd.DataFrame(trans_rows).to_csv(tab_dir / 'markov_transitions.csv', index=False)
    pd.DataFrame(sum_rows).to_csv(tab_dir / 'markov_summary.csv', index=False)
    pd.DataFrame(usdm_rows).to_csv(tab_dir / 'markov_usdm_sensitivity.csv', index=False)
    print("Completed Markov chain state transition analysis.")


if __name__ == '__main__':
    main()
