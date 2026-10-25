from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parent
for sub in ['bias_correction', 'drought_indices', 'copula_markov', 'ml_shap', 'figures']:
    sys.path.insert(0, str(CODE_DIR / sub))

import mbcn
import bmd_validation
import calc_indices
import topsis
import copula
import markov
import xgb_shap
import plot_figures

if __name__ == '__main__':
    mbcn.main()
    bmd_validation.main()
    calc_indices.main()
    topsis.main()
    copula.main()
    markov.main()
    xgb_shap.main()
    plot_figures.main()
