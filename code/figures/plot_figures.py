from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.8,
    "axes.titlesize": 9.6,
    "axes.labelsize": 8.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,
})

STATIONS = {
    "Rajshahi": (88.70, 24.37),
    "Ishurdi": (89.05, 24.13),
    "Bogura": (89.37, 24.85),
}

SCEN_COLOR = {
    "historical": "#4a4a4a",
    "ssp245": "#1f77b4",
    "ssp585": "#d62728",
}

SCEN_LABEL = {
    "historical": "Historical (1986-2014)",
    "ssp245": "SSP2-4.5 (2015-2100)",
    "ssp585": "SSP5-8.5 (2015-2100)",
}


def plot_figure_1(cfg, tab_dir, idx_dir, fig_dir):
    gp_path = REPO_ROOT / cfg["domain"]["grid_points_file"]
    bnd_path = REPO_ROOT / cfg["domain"]["boundary_file"]
    gp = pd.read_csv(gp_path)
    bnd = pd.read_csv(bnd_path) if bnd_path.exists() else None

    lat0 = cfg["domain"]["centroid_lat"]
    lon0 = cfg["domain"]["centroid_lon"]
    cell = cfg["domain"]["grid_resolution_deg"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.8, 5.0), gridspec_kw={"width_ratios": [1.05, 1.0]}
    )

    if bnd is not None:
        ax1.fill(bnd.longitude, bnd.latitude, facecolor="#eef4e8", edgecolor="none", zorder=1)
        ax1.plot(
            bnd.longitude, bnd.latitude, color="#4d4d4d", lw=1.3, zorder=4,
            label="Rajshahi Division Boundary"
        )

    for _, r in gp.iterrows():
        ax1.add_patch(
            Rectangle(
                (r.longitude - cell / 2, r.latitude - cell / 2),
                cell, cell, facecolor="#c6dbef", alpha=0.65,
                edgecolor="#2171b5", linewidth=0.7, zorder=2
            )
        )

    ax1.scatter(
        gp.longitude, gp.latitude, s=16, c="#08306b", zorder=3,
        label=f"ERA5 / CMIP6 Grid Nodes (n={len(gp)})"
    )
    ax1.scatter(
        [lon0], [lat0], marker="*", s=240, c="#e6550d", edgecolor="k",
        linewidth=0.6, zorder=5,
        label=f"Domain Centroid ({lat0:.4f}°N, {lon0:.4f}°E)"
    )

    for name, (lon, lat) in STATIONS.items():
        ax1.scatter([lon], [lat], marker="^", s=90, c="white", edgecolor="k", linewidth=1.0, zorder=6)
        ax1.annotate(name, (lon, lat), textcoords="offset points", xytext=(6, 5), fontsize=8, zorder=7)
    ax1.scatter([], [], marker="^", s=90, c="white", edgecolor="k", label="BMD Rain Gauge")

    ax1.set_xticks(np.arange(88.0, 90.01, 0.5))
    ax1.set_yticks(np.arange(23.5, 25.51, 0.5))
    ax1.set_xlim(87.85, 89.95)
    ax1.set_ylim(23.60, 25.40)
    ax1.set_aspect(1.0 / np.cos(np.deg2rad(lat0)))
    ax1.set_xlabel("Longitude (°E)")
    ax1.set_ylabel("Latitude (°N)")
    ax1.set_title("(a) Study Domain & 26 ERA5/CMIP6 0.25° Grid Cells\nRajshahi Division, Northwestern Bangladesh", fontweight="bold")
    ax1.legend(loc="lower left", fontsize=7.2, framealpha=0.92)

    era5_file = idx_dir / "ERA5_daily.csv"
    if era5_file.exists():
        era5_m = pd.read_csv(era5_file, index_col=0, parse_dates=True)
        cs, ce = cfg["periods"]["baseline"]
        e = era5_m[(era5_m.index.year >= cs) & (era5_m.index.year <= ce)]
        clim = e.groupby(e.index.month).mean()
        months = np.arange(1, 13)
        ax2.bar(months, clim["Precipitation"], color="#6baed6", edgecolor="#2171b5", label="Precipitation")
        ax2.plot(months, clim["PET"], "o-", color="#d62728", ms=4, label="PET (Hargreaves-Samani)")
        ax2.plot(months, clim["WaterBalance"], "s--", color="#2ca02c", ms=4, label="P - PET")
        ax2.axhline(0, color="k", lw=0.6)
        ax2.set_xticks(months)
        ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax2.set_xlabel("Month")
        ax2.set_ylabel("Monthly Flux (mm month⁻¹)")
        ax2.set_title("(b) Baseline Annual Hydro-Climatic Cycle (1986-2014)\nPrecipitation, PET, and Climatic Water Balance", fontweight="bold")
        ax2.legend(loc="upper left", fontsize=7.5)
        fig.savefig(fig_dir / "Fig1_study_area.png", dpi=300)
    plt.close(fig)


def plot_figure_2(cfg, tab_dir, fig_dir):
    val = pd.read_csv(tab_dir / "bias_correction_validation.csv")
    spl = pd.read_csv(tab_dir / "mbcn_split_sample_validation.csv")
    rk = pd.read_csv(tab_dir / "gcm_topsis_ranking_complete.csv").sort_values("TOPSIS_Score")
    top5 = cfg["cmip6"]["selected_top5"]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    a1, a2, a3, a4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    piv_p = val.pivot(index="GCM", columns="Stage", values="Precipitation_KGE_monthly").sort_values("MBCn")
    piv_oos = spl.pivot(index="GCM", columns="Stage", values="Precipitation_KGE_monthly")
    y = np.arange(len(piv_p))
    h = 0.27

    a1.barh(y - h, piv_p["raw"], h, color="#bdbdbd", edgecolor="k", lw=0.4, label="Raw GCM (1986-2014)")
    a1.barh(y, piv_p["MBCn"], h, color="#3182bd", edgecolor="k", lw=0.4, label="MBCn Full (1986-2014)")
    a1.barh(y + h, piv_oos.loc[piv_p.index, "MBCn_oos"], h, color="#2ca02c", edgecolor="k", lw=0.4, label="MBCn Split-Sample Val (2001-2014)")
    a1.set_yticks(y)
    a1.set_yticklabels(piv_p.index, fontsize=7.8)
    a1.set_xlabel("Monthly Precipitation KGE")
    a1.set_title("(a) Precipitation Monthly KGE: Raw vs. MBCn\n(Full Baseline & Out-of-Sample Split Validation)", fontweight="bold")
    a1.set_xlim(0.25, 0.92)
    a1.legend(fontsize=7.3, loc="lower right")

    piv_rho = val.pivot(index="GCM", columns="Stage", values="spearman_P_Tmax").loc[piv_p.index]
    a2.barh(y - 0.18, piv_rho["raw"], 0.36, color="#bdbdbd", edgecolor="k", lw=0.4, label="Raw GCM")
    a2.barh(y + 0.18, piv_rho["MBCn"], 0.36, color="#3182bd", edgecolor="k", lw=0.4, label="After MBCn")
    a2.axvline(0.298, color="#d62728", ls="--", lw=1.3, label="ERA5 Reference (0.298)")
    a2.set_yticks(y)
    a2.set_yticklabels(piv_p.index, fontsize=7.8)
    a2.set_xlabel(r"Daily Inter-Variable Spearman $\rho(P,\,T_{\max})$")
    a2.set_title("(b) Multivariate Dependence Restoration\n(Orthogonal Rotation Transfer Against ERA5)", fontweight="bold")
    a2.legend(fontsize=7.3, loc="upper left")

    top5_set = set(top5)
    cols_c = ["#d62728" if m in top5_set else "#9ecae1" for m in rk["Model"]]
    y_rk = np.arange(len(rk))
    a3.barh(y_rk, rk["TOPSIS_Score"], color=cols_c, edgecolor="k", lw=0.45)
    a3.set_yticks(y_rk)
    a3.set_yticklabels(rk["Model"], fontsize=7.8)
    a3.set_xlabel("TOPSIS Relative Closeness Coefficient ($C_i$)")
    a3.set_title("(c) Six-Criteria Unstandardized TOPSIS Ranking\n(Red = Selected Top-5 Ensemble; Equal Weights $w_j = 1/6$)", fontweight="bold")
    for yi, v in zip(y_rk, rk["TOPSIS_Score"]):
        a3.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=7.2)
    a3.set_xlim(0, 1.08)

    crit = ["P_KGE", "P_Bias_abs", "P_r", "Tmax_KGE", "Tmin_KGE", "Tmean_Bias_abs"]
    crit_labels = ["Rain KGE (+)", "|Rain Bias| (-)", "Rain r (+)", "Tmax KGE (+)", "Tmin KGE (+)", "|Tmean Bias| (-)"]
    sub = rk.set_index("Model")[crit].copy()
    disp = sub.copy()
    for col, imp in zip(crit, ["+", "-", "+", "+", "+", "-"]):
        cmin, cmax = sub[col].min(), sub[col].max()
        if imp == "+":
            disp[col] = (sub[col] - cmin) / (cmax - cmin + 1e-12)
        else:
            disp[col] = (cmax - sub[col]) / (cmax - cmin + 1e-12)

    im = a4.imshow(disp.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    a4.set_xticks(range(len(crit)))
    a4.set_xticklabels(crit_labels, rotation=28, ha="right", fontsize=7.6)
    a4.set_yticks(range(len(sub)))
    a4.set_yticklabels(sub.index, fontsize=7.8)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            val_ij = sub.values[i, j]
            txt = f"{val_ij:.1f}" if crit[j] == "P_Bias_abs" else f"{val_ij:.3f}"
            a4.text(j, i, txt, ha="center", va="center", fontsize=6.5, color="black")
    a4.grid(False)
    a4.set_title("(d) Physical Screening Matrix Across 13 CMIP6 Models\n(Shaded by Relative Skill; (+) Benefit, (-) Cost)", fontweight="bold")
    fig.colorbar(im, ax=a4, shrink=0.82, label="Normalized Skill (1 = Ideal Best)")

    fig.savefig(fig_dir / "Fig2_bias_correction_skill.png", dpi=300)
    plt.close(fig)


def plot_figure_3(cfg, tab_dir, idx_dir, fig_dir):
    top5 = cfg["cmip6"]["selected_top5"]
    if not (idx_dir.exists() and any(idx_dir.glob("*_drought_indices.csv"))):
        return

    fig = plt.figure(figsize=(13.2, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.95])
    ax_s1 = fig.add_subplot(gs[0, :])
    ax_s2 = fig.add_subplot(gs[1, :], sharey=ax_s1)
    ax_p1 = fig.add_subplot(gs[2, 0])
    ax_p2 = fig.add_subplot(gs[2, 1])

    ens = {}
    for sc in ["historical", "ssp245", "ssp585"]:
        ser = []
        for g in top5:
            d = pd.read_csv(idx_dir / f"{g}_{sc}_drought_indices.csv", parse_dates=["Date"]).set_index("Date")
            if sc == "historical":
                d = d[(d.index.year >= 1986) & (d.index.year <= 2014)]
            else:
                d = d[(d.index.year >= 2015) & (d.index.year <= 2100)]
            ser.append(d["SPEI_6"].rename(g))
        ens[sc] = pd.concat(ser, axis=1)

    for ax, sc, panel_lbl, f_full, f_eoc in [
        (ax_s1, "ssp245", "(a)", "5.24%", "10.06%"),
        (ax_s2, "ssp585", "(b)", "4.75%", "7.56%"),
    ]:
        h = ens["historical"]
        for frame, colr, lbl in [
            (h, SCEN_COLOR["historical"], "Historical (1986-2014)"),
            (ens[sc], SCEN_COLOR[sc], SCEN_LABEL[sc]),
        ]:
            m = frame.mean(axis=1)
            ax.fill_between(
                frame.index, frame.min(axis=1), frame.max(axis=1),
                color=colr, alpha=0.22, linewidth=0, label=f"{lbl} Top-5 Range"
            )
            ax.plot(frame.index, m, color=colr, lw=0.85, label=f"{lbl} Ensemble Mean")
        ax.axvline(pd.Timestamp("2015-01-01"), color="#333333", ls="-.", lw=1.1)
        ax.axvspan(pd.Timestamp("2071-01-01"), pd.Timestamp("2100-12-31"), color="#ffd92f", alpha=0.14, label="Late-Century Window (2071-2100)")
        ax.axhline(-1.5, color="#d62728", ls="--", lw=1.0, label="Severe Drought Threshold (-1.5)")
        ax.axhline(-1.0, color="#ff7f0e", ls=":", lw=0.9)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(
            f"{panel_lbl} Ensemble SPEI-6 Trajectory: Historical (1986-2014: 7.11% ≤ -1.5) vs. {SCEN_LABEL[sc]} "
            f"(Full 2015-2100: {f_full}; Late-Century 2071-2100: {f_eoc})",
            fontweight="bold", fontsize=9.2,
        )
        ax.set_ylabel("SPEI-6")
        ax.set_ylim(-3.8, 3.2)
        ax.legend(loc="lower left", ncol=4, fontsize=7.0, framealpha=0.92)
    ax_s2.set_xlabel("Year")

    ps = pd.read_csv(tab_dir / "pet_sensitivity_monthly.csv", index_col=0, parse_dates=True)
    d_pet = ps["PET_Hargreaves_minus_Thornthwaite"]
    clim_pet = d_pet.groupby(d_pet.index.month).agg(["mean", "std"])
    months = np.arange(1, 13)
    cols_p = ["#d62728" if v > 0 else "#1f77b4" for v in clim_pet["mean"]]
    ax_p1.bar(months, clim_pet["mean"], yerr=clim_pet["std"], color=cols_p, edgecolor="k", lw=0.4, capsize=2.2)
    ax_p1.axhline(d_pet.mean(), color="k", ls="--", lw=1.0, label=f"Annual Mean = {d_pet.mean():+.2f} mm mo⁻¹")
    mam_m = clim_pet.loc[[3, 4, 5], "mean"].mean()
    ax_p1.axhline(mam_m, color="#e6550d", ls=":", lw=1.3, label=f"Pre-Monsoon (MAM) = {mam_m:+.2f} mm mo⁻¹")
    ax_p1.axhline(0, color="k", lw=0.6)
    ax_p1.set_xticks(months)
    ax_p1.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax_p1.set_ylabel(r"$\mathrm{PET}_{\mathrm{HS}} - \mathrm{PET}_{\mathrm{Th}}$ (mm mo$^{-1}$)")
    ax_p1.set_xlabel("Month")
    ax_p1.set_title("(c) Monthly Evapotranspiration Formulation Sensitivity\n(Hargreaves-Samani vs. Daylength-Adjusted Thornthwaite)", fontweight="bold", fontsize=8.9)
    ax_p1.legend(fontsize=7.2, loc="lower left")

    comp_sum = pd.read_csv(tab_dir / "comprehensive_climate_drought_summary.csv")
    ens_w = comp_sum.groupby(["Scenario", "Window"])[["HotDays35_MAM_yr", "HotDays35_Ann_yr"]].mean()
    ens_w_sd = comp_sum.groupby(["Scenario", "Window"])[["HotDays35_MAM_yr", "HotDays35_Ann_yr"]].std()
    win_order = [
        ("historical", "1986-2014", "Hist\n(1986-2014)"),
        ("ssp245", "2015-2100", "SSP2-4.5\n(2015-2100)"),
        ("ssp245", "2071-2100", "SSP2-4.5\n(2071-2100)"),
        ("ssp585", "2015-2100", "SSP5-8.5\n(2015-2100)"),
        ("ssp585", "2071-2100", "SSP5-8.5\n(2071-2100)"),
    ]
    x_w = np.arange(len(win_order))
    mam_vals = [ens_w.loc[(s, w), "HotDays35_MAM_yr"] for s, w, _ in win_order]
    mam_sds = [ens_w_sd.loc[(s, w), "HotDays35_MAM_yr"] for s, w, _ in win_order]
    ann_vals = [ens_w.loc[(s, w), "HotDays35_Ann_yr"] for s, w, _ in win_order]
    ann_sds = [ens_w_sd.loc[(s, w), "HotDays35_Ann_yr"] for s, w, _ in win_order]

    ax_p2.bar(x_w - 0.18, mam_vals, 0.35, yerr=mam_sds, capsize=2.5, color="#fc8d59", edgecolor="k", lw=0.4, label="Pre-Monsoon Boro Window (Mar-May)")
    ax_p2.bar(x_w + 0.18, ann_vals, 0.35, yerr=ann_sds, capsize=2.5, color="#b30000", edgecolor="k", lw=0.4, label=r"Annual Total ($T_{\max} > 35\,^\circ$C Days)")
    ax_p2.set_xticks(x_w)
    ax_p2.set_xticklabels([lbl for _, _, lbl in win_order], fontsize=7.6)
    ax_p2.set_ylabel(r"Days yr$^{-1}$ with $T_{\max} > 35\,^\circ$C")
    ax_p2.set_title(r"(d) Spikelet-Sterility Heat Threshold Exceedance ($T_{\max} > 35\,^\circ$C)" + "\nDuring Pre-Monsoon Boro Reproductive Window vs. Annual", fontweight="bold", fontsize=8.9)
    ax_p2.legend(fontsize=7.2, loc="upper left")

    fig.savefig(fig_dir / "Fig3_spei6_timeseries.png", dpi=300)
    plt.close(fig)


def plot_figure_4(cfg, tab_dir, fig_dir):
    cop = pd.read_csv(tab_dir / "copula_enhanced_diagnostics.csv")
    sel = pd.read_csv(tab_dir / "copula_family_selection.csv")
    mk_tr = pd.read_csv(tab_dir / "markov_transitions.csv")
    top5 = cfg["cmip6"]["selected_top5"]
    order = top5 + ["ENSEMBLE"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.2, 4.8), gridspec_kw={"width_ratios": [1.3, 0.95, 1.1]})
    w = 0.26
    for i, sc in enumerate(["historical", "ssp245", "ssp585"]):
        s = cop[cop["Scenario"] == sc].set_index("GCM").reindex(order)
        x = np.arange(len(order)) + (i - 1) * w
        lo = np.clip(s["RP_Fixed_yr"] - s["CI_Block_Lo"], 0, None)
        hi = np.clip(s["CI_Block_Hi"] - s["RP_Fixed_yr"], 0, None)
        a1.bar(
            x, s["RP_Fixed_yr"], w, yerr=[lo, hi], capsize=2.0,
            color=SCEN_COLOR[sc], edgecolor="k", lw=0.4,
            label=f"{SCEN_LABEL[sc]} (Empirical Fixed $P_{{90}}$)"
        )
        if "RP_Parametric_Copula_yr" in s.columns:
            a1.scatter(
                x, s["RP_Parametric_Copula_yr"], marker="o", s=22, color="#00ffff",
                edgecolor="k", lw=0.5, zorder=6,
                label="Parametric Copula Eq. (6)" if i == 0 else None
            )
        a1.scatter(
            x, s["RP_CalMonth_yr"], marker="D", s=22, color="#ffd92f",
            edgecolor="k", lw=0.5, zorder=5,
            label="Calendar-Month $P_{90,m}$ Anomaly" if i == 0 else None
        )
        for xi, nev in zip(x, s["N_Events_Fixed"]):
            a1.text(xi, 1.08, f"{int(nev)}", ha="center", va="bottom", fontsize=6.0, color="white", rotation=90, fontweight="bold")

    a1.set_xticks(np.arange(len(order)))
    a1.set_xticklabels(order, rotation=22, ha="right", fontsize=7.8)
    a1.set_yscale("log")
    a1.set_ylim(1.0, 38.0)
    a1.set_ylabel("Compound Hot-Dry Return Period (Years, Log Scale)")
    a1.set_title("(a) Compound Return Periods ($T_{\\max} > P_{90}$ & SPEI-6 $\\leq -1.5$)\n12-Mo Block-Bootstrap 95% CI; Circles = Parametric; Diamonds = $P_{90,m}$", fontweight="bold", fontsize=8.8)
    a1.legend(fontsize=6.8, loc="upper left", framealpha=0.93)

    sel_gcm = sel[sel["GCM"] != "ENSEMBLE"].copy()
    sel_gcm["N"] = sel_gcm["Scenario"].map({"historical": 343, "ssp245": 1027, "ssp585": 1027})
    sel_gcm["AIC_per_N"] = sel_gcm["AIC"] / sel_gcm["N"]
    piv_aic = sel_gcm.pivot_table(index="Family", columns="Scenario", values="AIC_per_N", aggfunc="mean").reindex(cfg["copula"]["families"])
    piv_aic[["historical", "ssp245", "ssp585"]].plot(
        kind="bar", ax=a2, color=[SCEN_COLOR[c] for c in ["historical", "ssp245", "ssp585"]],
        edgecolor="k", lw=0.4, legend=False
    )
    a2.set_ylabel("Sample-Normalized Mean AIC / $N$ (Lower = Better Fit)")
    a2.set_xlabel("Parametric Copula Family")
    a2.tick_params(axis="x", rotation=18)
    a2.set_title("(b) Sample-Normalized Goodness-of-Fit (AIC / $N$)\nConfirming Upper-Tail Gumbel & Joe Dominance", fontweight="bold", fontsize=8.8)

    mk_piv = mk_tr.pivot_table(index=["Scenario", "From_State"], columns="To_State", values="Probability", aggfunc="mean")
    mk_sd = mk_tr.pivot_table(index=["Scenario", "From_State"], columns="To_State", values="Probability", aggfunc="std")
    trans_types = [
        ("Normal", "Severe", "Non-Drought → Severe\n(×10 Scaled)"),
        ("Normal", "ModSev", "Non-Drought → Mod/Sev\n(×5 Scaled)"),
        ("Moderate", "Severe", "Moderate → Severe\n(Escalation)"),
        ("Severe", "Severe", "Severe → Severe\n(Persistence)"),
    ]
    x_t = np.arange(len(trans_types))
    for i, sc in enumerate(["historical", "ssp245", "ssp585"]):
        vals_t, sds_t = [], []
        for fr, to, _ in trans_types:
            if to == "ModSev":
                sub_g = mk_tr[(mk_tr["Scenario"] == sc) & (mk_tr["From_State"] == "Normal") & (mk_tr["To_State"].isin(["Moderate", "Severe"]))].groupby("GCM")["Probability"].sum()
                vals_t.append(sub_g.mean() * 5.0)
                sds_t.append(sub_g.std() * 5.0)
            elif fr == "Normal" and to == "Severe":
                vals_t.append(mk_piv.loc[(sc, fr), to] * 10.0)
                sds_t.append(mk_sd.loc[(sc, fr), to] * 10.0)
            else:
                vals_t.append(mk_piv.loc[(sc, fr), to])
                sds_t.append(mk_sd.loc[(sc, fr), to])
        a3.bar(
            x_t + (i - 1) * w, vals_t, w, yerr=sds_t, capsize=2.0,
            color=SCEN_COLOR[sc], edgecolor="k", lw=0.4, label=SCEN_LABEL[sc]
        )

    a3.set_xticks(x_t)
    a3.set_xticklabels([lbl for _, _, lbl in trans_types], fontsize=7.4)
    a3.set_ylabel("Transition Probability")
    a3.set_title("(c) First-Order Markov SPEI-6 State Transitions\n(Abrupt Onset Escalation vs. Declining Persistence)", fontweight="bold", fontsize=8.8)
    a3.legend(fontsize=7.0, loc="upper left")

    fig.savefig(fig_dir / "Fig4_copula_return_periods.png", dpi=300)
    plt.close(fig)


def plot_figure_5(cfg, tab_dir, fig_dir):
    ml_p = pd.read_csv(tab_dir / "ml_purged_cv_and_persistence.csv")
    sh_p = pd.read_csv(tab_dir / "ml_shap_global_vs_tail.csv")
    top5 = cfg["cmip6"]["selected_top5"]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.4, 4.7), gridspec_kw={"width_ratios": [1.15, 1.0, 1.05]})
    w = 0.25
    for i, sc in enumerate(["historical", "ssp245", "ssp585"]):
        s = ml_p[ml_p["Scenario"] == sc].set_index("GCM").reindex(top5)
        x = np.arange(len(top5)) + (i - 1) * w
        a1.bar(
            x, s["ModelA_R2_cv"], w, yerr=s["ModelA_R2_std"], capsize=2.0,
            color=SCEN_COLOR[sc], edgecolor="k", lw=0.4, label=f"Model A XGBoost: {SCEN_LABEL[sc]}"
        )
        a1.scatter(
            x, s["Persistence_R2"], marker="o", s=22, color="#ffd92f", edgecolor="k", lw=0.5, zorder=5,
            label=r"Naive Persistence Baseline ($\hat{y}_t = \mathrm{SPEI6}_{t-1}$)" if i == 0 else None
        )
        a1.scatter(
            x, s["ModelB_MetOnly_R2_cv"], marker="^", s=22, color="#2ca02c", edgecolor="k", lw=0.5, zorder=5,
            label="Model B (Pure Meteorological Drivers)" if i == 0 else None
        )

    a1.axhline(
        ml_p["ModelA_Train_R2"].mean(), color="k", ls="--", lw=1.0,
        label=f"In-Sample Training $R^2$ ({ml_p['ModelA_Train_R2'].mean():.3f})"
    )
    a1.set_xticks(np.arange(len(top5)))
    a1.set_xticklabels(top5, rotation=22, ha="right", fontsize=7.8)
    a1.set_ylabel("Out-of-Sample $R^2$ (1-Month-Ahead SPEI-6)")
    a1.set_ylim(0.25, 1.25)
    a1.set_title("(a) 5-Fold Blocked Temporal CV (6-Month Purge Buffer)\nModel A vs. Persistence Baseline & Model B (Met-Only)", fontweight="bold", fontsize=8.8)
    a1.legend(fontsize=6.6, loc="upper left", framealpha=0.93)

    piv_ma = sh_p[sh_p["Model_Config"] == "ModelA_Autoreg"].pivot_table(
        index="Variable", columns="Scenario", values="SHAP_Share_Global", aggfunc="mean"
    )[["historical", "ssp245", "ssp585"]].sort_values("historical") * 100.0
    y_ma = np.arange(len(piv_ma))
    for i, sc in enumerate(["historical", "ssp245", "ssp585"]):
        a2.barh(y_ma + (i - 1) * w, piv_ma[sc], w, color=SCEN_COLOR[sc], edgecolor="k", lw=0.4, label=SCEN_LABEL[sc])
    a2.set_yticks(y_ma)
    a2.set_yticklabels(piv_ma.index, fontsize=8.0)
    a2.set_xlabel("Share of Total Out-of-Sample Mean $|$SHAP$|$ (%)")
    a2.set_title("(b) Model A (Autoregressive Forecast) Attribution\nLagged State Memory vs. Meteorological Forcing", fontweight="bold", fontsize=8.8)
    a2.legend(fontsize=6.8, loc="lower right")

    sub_mb = sh_p[sh_p["Model_Config"] == "ModelB_MetOnly"].groupby("Variable")[
        ["SHAP_Share_Global", "SHAP_Share_Tail_le_m1p0", "SHAP_Share_Tail_le_m1p5"]
    ].mean().sort_values("SHAP_Share_Global") * 100.0
    y_mb = np.arange(len(sub_mb))
    a3.barh(y_mb - w, sub_mb["SHAP_Share_Global"], w, color="#3182bd", edgecolor="k", lw=0.4, label="Global (All Out-of-Sample Months)")
    a3.barh(y_mb, sub_mb["SHAP_Share_Tail_le_m1p0"], w, color="#fc8d59", edgecolor="k", lw=0.4, label=r"Moderate Drought Tail ($\mathrm{SPEI6} \leq -1.0$)")
    a3.barh(y_mb + w, sub_mb["SHAP_Share_Tail_le_m1p5"], w, color="#d62728", edgecolor="k", lw=0.4, label=r"Severe Drought Tail ($\mathrm{SPEI6} \leq -1.5$)")
    a3.set_yticks(y_mb)
    a3.set_yticklabels(sub_mb.index, fontsize=8.0)
    a3.set_xlabel("Meteorological Driver Attribution Share (%)")
    a3.set_title(r"(c) Model B (Non-Tautological Met Drivers Only)" + "\n" + r"Global vs. Severe Drought Tail ($\mathrm{SPEI6} \leq -1.5$)", fontweight="bold", fontsize=8.8)
    a3.legend(fontsize=6.8, loc="lower right")

    fig.savefig(fig_dir / "Fig5_ml_shap.png", dpi=300)
    plt.close(fig)


def main():
    config_path = REPO_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = REPO_ROOT / cfg["paths"]["output_dir"]
    tab_dir = out_dir / "tables"
    idx_dir = out_dir / "indices"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_figure_1(cfg, tab_dir, idx_dir, fig_dir)
    plot_figure_2(cfg, tab_dir, fig_dir)
    plot_figure_3(cfg, tab_dir, idx_dir, fig_dir)
    plot_figure_4(cfg, tab_dir, fig_dir)
    plot_figure_5(cfg, tab_dir, fig_dir)
    print(f"Saved manuscript figures to {fig_dir}")


if __name__ == "__main__":
    main()
