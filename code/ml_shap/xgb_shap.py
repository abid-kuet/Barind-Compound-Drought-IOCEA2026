from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap
import yaml

ROOT = Path(__file__).resolve().parents[2]


def make_lagged_features(df, lags=6, include_autoreg=True):
    df = df.copy()
    df['DTR'] = df['Tmax'] - df['Tmin']
    met_cols = ['Precipitation', 'Tmax', 'Tmin', 'DTR', 'PET']

    feats = {}
    for col in met_cols:
        for lag in range(1, lags + 1):
            feats[f'{col}_lag{lag}'] = df[col].shift(lag)

    if include_autoreg:
        for lag in range(1, lags + 1):
            feats[f'SPEI_6_lag{lag}'] = df['SPEI_6'].shift(lag)

    feats['sin_month'] = np.sin(2.0 * np.pi * df.index.month / 12.0)
    feats['cos_month'] = np.cos(2.0 * np.pi * df.index.month / 12.0)

    X = pd.DataFrame(feats, index=df.index)
    y = df['SPEI_6']
    y_pers = df['SPEI_6'].shift(1)
    valid = X.notna().all(axis=1) & y.notna() & y_pers.notna()
    return X[valid], y[valid], y_pers[valid]


def run_purged_cv(X, y, y_pers, xgb_params, n_splits=5, purge_months=6):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(X))

    y_true_all, y_pred_all, y_pers_all = [], [], []
    shap_blocks, va_idx_blocks, fold_records = [], [], []

    for k, (tr, va) in enumerate(splits, start=1):
        # Drop last 6 months of training fold so 6-month SPEI windows don't overlap with validation start
        tr_clean = tr[:-purge_months] if len(tr) > purge_months + 24 else tr
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X.iloc[tr_clean], y.iloc[tr_clean])

        pred = model.predict(X.iloc[va])
        pers = y_pers.iloc[va].values
        truth = y.iloc[va].values

        y_true_all.append(truth)
        y_pred_all.append(pred)
        y_pers_all.append(pers)

        fold_records.append({
            'Fold': k,
            'Train_Start': str(X.index[tr_clean[0]].date()),
            'Train_End': str(X.index[tr_clean[-1]].date()),
            'Val_Start': str(X.index[va[0]].date()),
            'Val_End': str(X.index[va[-1]].date()),
            'N_Train': len(tr_clean),
            'N_Val': len(va),
            'R2_XGB': float(r2_score(truth, pred)),
            'R2_Persistence': float(r2_score(truth, pers)),
            'MAE_XGB': float(mean_absolute_error(truth, pred)),
            'MAE_Persistence': float(mean_absolute_error(truth, pers)),
        })

        explainer = shap.TreeExplainer(model)
        shap_blocks.append(np.asarray(explainer.shap_values(X.iloc[va])))
        va_idx_blocks.append(va)

    yt = np.concatenate(y_true_all)
    yp = np.concatenate(y_pred_all)
    y_p = np.concatenate(y_pers_all)
    sv_all = np.vstack(shap_blocks)
    y_oos = y.iloc[np.concatenate(va_idx_blocks)]

    imp_glob = pd.Series(np.abs(sv_all).mean(axis=0), index=X.columns)
    m_mod = y_oos.values <= -1.0
    m_sev = y_oos.values <= -1.5
    imp_t10 = pd.Series(np.abs(sv_all[m_mod]).mean(axis=0), index=X.columns) if m_mod.sum() >= 3 else imp_glob
    imp_t15 = pd.Series(np.abs(sv_all[m_sev]).mean(axis=0), index=X.columns) if m_sev.sum() >= 3 else imp_t10

    tr_last, _ = splits[-1]
    tr_last_clean = tr_last[:-purge_months] if len(tr_last) > purge_months + 24 else tr_last
    m_last = xgb.XGBRegressor(**xgb_params)
    m_last.fit(X.iloc[tr_last_clean], y.iloc[tr_last_clean])
    r2_tr = float(r2_score(y.iloc[tr_last_clean], m_last.predict(X.iloc[tr_last_clean])))

    metrics = {
        'R2_cv_pooled': float(r2_score(yt, yp)),
        'RMSE_cv_pooled': float(np.sqrt(mean_squared_error(yt, yp))),
        'MAE_cv_pooled': float(mean_absolute_error(yt, yp)),
        'R2_cv_mean': float(np.mean([f['R2_XGB'] for f in fold_records])),
        'R2_cv_std': float(np.std([f['R2_XGB'] for f in fold_records])),
        'R2_persistence_pooled': float(r2_score(yt, y_p)),
        'MAE_persistence_pooled': float(mean_absolute_error(yt, y_p)),
        'R2_train_lastfold': r2_tr,
    }
    return metrics, fold_records, imp_glob, imp_t10, imp_t15


def aggregate_by_variable(imp_series):
    def base_name(col):
        if col in ('sin_month', 'cos_month'):
            return 'AnnualCycle'
        return col.rsplit('_lag', 1)[0]

    grp = imp_series.groupby(imp_series.index.map(base_name)).sum()
    return grp / grp.sum() if grp.sum() > 0 else grp


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    y0, y1 = cfg['periods']['baseline']
    f0, f1 = cfg['periods']['future_full']
    top5 = cfg['cmip6']['selected_top5']
    lags = cfg['xgboost']['lags']
    n_folds = cfg['xgboost']['cv_folds']
    purge = cfg['xgboost']['purge_months']

    out_root = ROOT / cfg['paths']['output_dir']
    idx_dir = out_root / 'indices'
    tab_dir = out_root / 'tables'
    tab_dir.mkdir(parents=True, exist_ok=True)

    params = dict(
        **cfg['xgboost']['params'],
        random_state=cfg['xgboost']['seed'],
        n_jobs=4,
        objective='reg:squarederror',
        verbosity=0,
    )

    perf_rows, shap_rows, fold_rows = [], [], []

    for gcm in top5:
        for scen in ['historical', 'ssp245', 'ssp585']:
            df = pd.read_csv(idx_dir / f'{gcm}_{scen}_drought_indices.csv', parse_dates=['Date']).set_index('Date').sort_index()
            df = df[(df.index.year >= y0) & (df.index.year <= y1)] if scen == 'historical' else df[(df.index.year >= f0) & (df.index.year <= f1)]

            Xa, ya, ypa = make_lagged_features(df, lags=lags, include_autoreg=True)
            ma, folds_a, ga, t10a, t15a = run_purged_cv(Xa, ya, ypa, params, n_splits=n_folds, purge_months=purge)

            Xb, yb, ypb = make_lagged_features(df, lags=lags, include_autoreg=False)
            mb, _, gb, t10b, t15b = run_purged_cv(Xb, yb, ypb, params, n_splits=n_folds, purge_months=purge)

            if gcm == top5[0]:
                for fr in folds_a:
                    fold_rows.append({'Scenario': scen, **fr})

            perf_rows.append({
                'GCM': gcm, 'Scenario': scen, 'N_Samples': len(Xa),
                'ModelA_R2_cv': ma['R2_cv_pooled'],
                'ModelA_RMSE_cv': ma['RMSE_cv_pooled'],
                'ModelA_MAE_cv': ma['MAE_cv_pooled'],
                'ModelA_R2_mean': ma['R2_cv_mean'],
                'ModelA_R2_std': ma['R2_cv_std'],
                'Persistence_R2': ma['R2_persistence_pooled'],
                'Persistence_MAE': ma['MAE_persistence_pooled'],
                'ModelB_MetOnly_R2_cv': mb['R2_cv_pooled'],
                'ModelB_MetOnly_MAE_cv': mb['MAE_cv_pooled'],
                'ModelA_Train_R2': ma['R2_train_lastfold'],
            })

            for tag, ig, it10, it15 in [('ModelA_Autoreg', ga, t10a, t15a), ('ModelB_MetOnly', gb, t10b, t15b)]:
                sg, st10, st15 = aggregate_by_variable(ig), aggregate_by_variable(it10), aggregate_by_variable(it15)
                for v in sg.index:
                    shap_rows.append({
                        'GCM': gcm, 'Scenario': scen, 'Model_Config': tag,
                        'Variable': v,
                        'SHAP_Share_Global': float(sg[v]),
                        'SHAP_Share_Tail_le_m1p0': float(st10[v]),
                        'SHAP_Share_Tail_le_m1p5': float(st15[v]),
                    })

    df_shap = pd.DataFrame(shap_rows)
    pd.DataFrame(perf_rows).to_csv(tab_dir / 'ml_purged_cv_and_persistence.csv', index=False)
    df_shap.to_csv(tab_dir / 'ml_shap_global_vs_tail.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(tab_dir / 'ml_purged_fold_schedule.csv', index=False)

    # Grouped collinearity summary (Thermal+Evaporative vs Moisture Supply vs Autoregressive Memory)
    thermal_set = {'Tmax', 'Tmin', 'DTR', 'PET'}
    df_shap['Predictor_Group'] = df_shap['Variable'].map(
        lambda v: 'Thermal_Evaporative_Group (Tmax+Tmin+DTR+PET)' if v in thermal_set
        else ('Moisture_Supply_Group (Precipitation)' if v == 'Precipitation'
              else ('Autoregressive_Memory_Group (SPEI_6_lags)' if v == 'SPEI_6' else 'Annual_Cycle_Group'))
    )
    grp = df_shap.groupby(['Model_Config', 'Scenario', 'GCM', 'Predictor_Group'])[['SHAP_Share_Global', 'SHAP_Share_Tail_le_m1p5']].sum().reset_index()
    grp_ens = grp.groupby(['Model_Config', 'Scenario', 'Predictor_Group'])[['SHAP_Share_Global', 'SHAP_Share_Tail_le_m1p5']].agg(['mean', 'std']).reset_index()
    grp_ens.columns = ['Model_Config', 'Scenario', 'Predictor_Group', 'Global_Mean', 'Global_SD', 'Tail_Mean', 'Tail_SD']
    grp_ens.to_csv(tab_dir / 'ml_shap_grouped_collinearity.csv', index=False)
    print("Completed purged blocked-CV XGBoost and TreeSHAP attribution.")


if __name__ == '__main__':
    main()
