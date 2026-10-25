from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import yaml

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = yaml.safe_load(open(ROOT / 'config.yaml', encoding='utf-8'))
    obs_path = ROOT / cfg['paths']['bmd_stations']
    era5_path = ROOT / cfg['paths']['era5_daily']
    tab_dir = ROOT / cfg['paths']['output_dir'] / 'tables'
    tab_dir.mkdir(parents=True, exist_ok=True)

    df_bmd = pd.ExcelFile(obs_path).parse('Daily_Combined')
    df_bmd['Date'] = pd.to_datetime(df_bmd['Date'])
    df_bmd = df_bmd.set_index('Date').sort_index()
    df_bmd = df_bmd[(df_bmd.index.year >= 1986) & (df_bmd.index.year <= 2012)]

    era5 = pd.read_csv(era5_path, parse_dates=['date']).set_index('date').sort_index()
    era5 = era5[~era5.index.duplicated()]
    era5 = era5[(era5.index.year >= 1986) & (era5.index.year <= 2012)]

    bmd_m = df_bmd[['Rajshahi_mm', 'Bogura_mm', 'Ishurdi_mm']].resample('MS').sum(min_count=20)
    bmd_m['BMD_3Station_Mean'] = bmd_m[['Rajshahi_mm', 'Bogura_mm', 'Ishurdi_mm']].mean(axis=1)
    era5_m = era5['PrecipData'].resample('MS').sum()

    stations = [
        ('Rajshahi_mm', 'Rajshahi (Western Barind Core)'),
        ('Ishurdi_mm', 'Ishurdi (Southern Ganges Terrace)'),
        ('Bogura_mm', 'Bogura (Eastern Karatoya-Jamuna Margin)'),
        ('BMD_3Station_Mean', '3-Station Gauge Mean'),
    ]

    rows = []
    for col, label in stations:
        idx = bmd_m[col].dropna().index.intersection(era5_m.index)
        s_gauge = bmd_m.loc[idx, col]
        s_era = era5_m.loc[idx]

        ann = s_gauge.groupby(s_gauge.index.year).sum()
        ann = ann[s_gauge.groupby(s_gauge.index.year).count() == 12]

        clim_g = s_gauge.groupby(s_gauge.index.month).mean()
        clim_e = s_era.groupby(s_era.index.month).mean()

        rows.append({
            'Station': label,
            'N_Valid_Months': len(idx),
            'Annual_Mean_mm': float(ann.mean()),
            'Annual_SD_mm': float(ann.std()),
            'Monthly_r_vs_ERA5': float(np.corrcoef(s_gauge, s_era)[0, 1]),
            'Monthly_Spearman_vs_ERA5': float(stats.spearmanr(s_gauge, s_era).statistic),
            'Climatology_r_vs_ERA5': float(np.corrcoef(clim_g, clim_e)[0, 1]),
            'Monthly_RMSE_mm': float(np.sqrt(np.mean((s_era - s_gauge) ** 2))),
        })

    out = pd.DataFrame(rows)
    out.to_csv(tab_dir / 'bmd_vs_era5_validation.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
