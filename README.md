# Barind-Compound-Drought-IOCEA2026

Analysis scripts and summary outputs for the IOCEA 2026 paper:
*Compound Hot-Dry Extremes and Multi-Scalar Drought Projections Under CMIP6 Scenarios in Northwestern Bangladesh: Multi-Criteria Model Screening and Explainable Machine Learning Attribution* (Anower Hossain Abid and Debbendu Saha).

### Folders
- `code/bias_correction/`: `mbcn.py` (daily MBCn bias correction and split-sample check), `bmd_validation.py` (ERA5 vs. BMD station rain gauge comparison)
- `code/drought_indices/`: `calc_indices.py` (Hargreaves-Samani and Thornthwaite PET, SPI, SPEI-3/6/12, RDI-6), `topsis.py` (6-criteria TOPSIS ranking of the 13 CMIP6 models)
- `code/copula_markov/`: `copula.py` (bivariate copula fitting, effective sample size, and block-bootstrap return periods), `markov.py` (first-order Markov transition matrices)
- `code/ml_shap/`: `xgb_shap.py` (1-month-ahead XGBoost with 6-month purge buffer and TreeSHAP attribution)
- `code/figures/`: `plot_figures.py` (plots Figures 1 to 5 from the CSV tables)
- `data/`: grid coordinates (`Rajshahi_grid_points.csv`) and digitized division boundary (`study_area_boundary_digitised.csv`)
- `outputs/tables/` and `outputs/figures/`: pre-computed CSV tables and 300 DPI figures

### Running
Install dependencies from `requirements.txt` (`pip install -r requirements.txt`), then run:

```bash
python code/figures/plot_figures.py
```

or run the full pipeline from raw daily files:

```bash
python code/run_all.py
```
