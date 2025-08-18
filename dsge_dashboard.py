# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit dashboard that reproduces the R "Best Model" (IS)
# and runs Phillips + Taylor with IRFs.
#
# IS (R Best Model):
#   DlogGDP_t ~ RR_exante_{t-2} + Dlog FD_{t-1} + Dlog REER_{t-1}
#               + Dlog Energy_{t-1} + Dlog NonEnergy_{t-1}
#
# Notes:
# - Real rate is EX-ANTE: i_t - pi_{t-1}, then lagged 2 more quarters.
# - Robust (HC1) SEs in summaries.
# - Plots show Baseline vs Shock paths.
# -----------------------------------------------------------

from typing import Tuple, Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Page setup
# =========================
st.set_page_config(page_title="DSGE IRF Dashboard — R Best IS Spec", layout="wide")
st.title("DSGE IRF Dashboard — Replicating R Best IS Specification")

st.markdown(
    "- **Units**: GDP & CPI in **%** (Dlog × 100); **Nominal rate** in **decimal**.  \n"
    "- **Taylor** uses **inflation gap**: \\(\\pi_t - \\pi^*\\).  \n"
    "- **IS (R Best Model)** uses **ex-ante** real rate at **t−2** and t−1 lags of FD/REER/Energy/NonEnergy."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # If values look like percents (e.g., 3.2), convert to decimal (0.032)
    return s / 100.0 if np.nanmedian(np.abs(s.values)) > 1.0 else s

def fmt_coef(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return f"+{s}" if x >= 0 else s

def row_from_params(params_index: pd.Index, values: Dict[str, float]) -> pd.DataFrame:
    cols = list(params_index)
    row = {}
    for c in cols:
        if c == "const":
            row[c] = 1.0
        else:
            row[c] = float(values.get(c, 0.0))
    return pd.DataFrame([row], columns=cols)

def build_latex_equation(const_val: float, terms: List[tuple], lhs: str, eps_symbol: str) -> str:
    rhs_terms = " ".join([f"{fmt_coef(c)}\\,{sym}" for (c, sym) in terms]) if terms else ""
    eq = rf"""
    \begin{{aligned}}
    {lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol}
    \end{{aligned}}
    """
    return eq

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Data")
    xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"])
    fallback = Path(__file__).parent / "DSGE.xlsx"

    st.header("Spec mode")
    spec_mode = st.radio(
        "Choose IS specification",
        ["R Best Model (fixed)", "Custom (pick variables)"],
        index=0,
        help="R Best Model matches your LaTeX table column (7). Custom lets you experiment."
    )

    st.header("Simulation")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)
    rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

    st.header("Inflation target for Taylor")
    use_sample_mean = st.checkbox("Use sample mean of DlogCPI as π*", value=False)
    if use_sample_mean:
        target_annual_pct = None
        st.caption("π* will be set to sample mean (quarterly) after data loads.")
    else:
        target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)

    st.divider()
    st.header("Shock")
    shock_target = st.selectbox(
        "Apply shock to",
        ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
        index=0
    )
    is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
    pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
    policy_shock_bp_abs = st.number_input("Policy shock size (absolute bp)", value=25, step=5, format="%d")
    shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

    st.divider()
    st.header("Diagnostics")
    robust_se = st.checkbox("Use robust SE (HC1) in OLS summaries", value=True)

    if spec_mode == "Custom (pick variables)":
        st.divider()
        st.header("Custom IS regressors")
        # Provide both the ex-ante L2 real rate and the rest so you can compare
        IS_ALL = [
            "RR_exante_L2",
            "Dlog FD_Lag1", "Dlog_REER_L1", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
            "DlogGDP_L1",  # optional, not in R best model
        ]
        is_selected = st.multiselect("Choose variables (const added automatically):", IS_ALL, default=IS_ALL)

# =========================
# Data prep
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE.xlsx or place it beside this script.")

    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Normalize nominal rate units
    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

    # Lags used elsewhere
    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)

    # --- Real rate definitions ---
    # ex-post: i_t - pi_t (not used in best spec but kept for completeness)
    df["RR_expost"] = df["Nominal Rate"] - df["Dlog_CPI"]
    # ex-ante base: i_t - pi_{t-1}
    df["RR_exante_L1_base"] = df["Nominal Rate"] - df["Dlog_CPI"].shift(1)
    # R Best Model uses ex-ante at t-2:
    df["RR_exante_L2"] = df["RR_exante_L1_base"].shift(2)

    # --- t-1 controls to match R table ---
    df["Dlog FD_Lag1"]        = df["Dlog FD_Lag1"]         # already lagged in your sheet naming
    df["Dlog_REER_L1"]        = df["Dlog_REER"].shift(1)
    df["Dlog_Energy_L1"]      = df["Dlog_Energy"].shift(1)
    df["Dlog_Non_Energy_L1"]  = df["Dlog_Non_Energy"].shift(1)
    df["Dlog_Reer_L2"]        = df["Dlog_REER"].shift(2)   # for Phillips (as in your prior spec)

    # Required columns for clean dropna
    required_cols = [
        "DlogGDP",
        "Dlog_CPI",
        "Nominal Rate",
        "Nominal_Rate_L1",
        # IS (Best Model)
        "RR_exante_L2", "Dlog FD_Lag1", "Dlog_REER_L1", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
        # Phillips
        "Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")
    return df, df_est

# =========================
# Modeling
# =========================
def fit_models(
    df_est: pd.DataFrame,
    pi_star_quarterly: float,
    spec_mode: str,
    is_selected_custom: Optional[List[str]] = None,
    robust: bool = True,
):
    # ---------- IS ----------
    if spec_mode == "R Best Model (fixed)":
        is_cols = ["RR_exante_L2", "Dlog FD_Lag1", "Dlog_REER_L1", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
    else:
        if not is_selected_custom:
            raise ValueError("Pick at least one IS regressor for Custom mode.")
        is_cols = list(is_selected_custom)

    X_is = sm.add_constant(df_est[is_cols], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit(cov_type=("HC1" if robust else "nonrobust"))

    # ---------- Phillips ----------
    X_pc = sm.add_constant(df_est[["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit(cov_type=("HC1" if robust else "nonrobust"))

    # ---------- Taylor (inflation gap) ----------
    infl_gap = df_est["Dlog_CPI"] - pi_star_quarterly
    X_tr = sm.add_constant(pd.DataFrame({
        "Nominal_Rate_L1": df_est["Nominal_Rate_L1"],
        "Inflation_Gap": infl_gap,
        "DlogGDP": df_est["DlogGDP"],
    }), has_constant="add")
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit(cov_type=("HC1" if robust else "nonrobust"))

    # Convert partial-adjustment: i_t = ρ i_{t-1} + (1-ρ)(α* + φπ* gap + φg* g_t)
    b0 = float(model_tr.params.get("const", 0.0))
    rhoh = float(model_tr.params.get("Nominal_Rate_L1", 0.0))
    rhoh = min(max(rhoh, 0.0), 0.99)
    def safe_div(num, den): return num / den if abs(den) > 1e-8 else np.nan
    alpha_star = safe_div(b0, (1 - rhoh))
    bpi = float(model_tr.params.get("Inflation_Gap", 0.0))
    bg  = float(model_tr.params.get("DlogGDP", 0.0))
    phi_pi_star = safe_div(bpi, (1 - rhoh))
    phi_g_star  = safe_div(bg, (1 - rhoh))

    return {
        "model_is": model_is, "is_cols": is_cols,
        "model_pc": model_pc, "model_tr": model_tr,
        "alpha_star": alpha_star, "phi_pi_star": phi_pi_star, "phi_g_star": phi_g_star,
        "rho_hat": rhoh, "pi_star_quarterly": float(pi_star_quarterly),
    }

def build_shocks(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
    is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k-1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k-1]
    elif target == "Taylor (Policy tightening)":
        pol_arr[t0] =  policy_bp_abs / 10000.0
        for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k-1]
    elif target == "Taylor (Policy easing)":
        pol_arr[t0] = -policy_bp_abs / 10000.0
        for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k-1]
    return is_arr, pc_arr, pol_arr

def simulate(
    T: int, rho_sim: float, df_est: pd.DataFrame, models: Dict[str, object],
    means: Dict[str, float], i_mean_dec: float, pi_star_quarterly: float,
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None,
):
    """
    Build X_t each step from the *fitted params list* so the IS uses exactly the chosen columns.
    Real rate inside IS is computed to match RR_exante_L2 timing in the simulation path.
    """
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; is_cols = models["is_cols"]
    model_pc = models["model_pc"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

    for t in range(1, T):
        # --- Build IS row consistent with RR_exante_L2 timing:
        # RR_exante_L2 := (i_{t-2} - pi_{t-3})
        rr_exante_L2 = (i[t-2] - p[t-3]) if t >= 3 else means["RR_exante_L2"]

        vals_is = {
            "RR_exante_L2": rr_exante_L2,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER_L1": means["Dlog_REER_L1"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
            "DlogGDP_L1": g[t-1],  # harmless if not in params
        }
        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        # --- Phillips row (same as your earlier spec) ---
        vals_pc = {
            "Dlog_CPI_L1": p[t-1],
            "DlogGDP_L1": g[t-1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }
        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        # --- Taylor target (inflation gap)
        pi_gap_t = p[t] - pi_star_quarterly
        i_star = alpha_star + phi_pi_star * pi_gap_t + phi_g_star * g[t]
        i[t] = rho_sim * i[t-1] + (1 - rho_sim) * i_star + policy_shock_arr[t]

    return g, p, i

# =========================
# Run
# =========================
try:
    file_source = xlf if xlf is not None else (fallback if 'fallback' in locals() else None)
    df_all, df_est = load_and_prepare(file_source)

    # Determine π* (quarterly decimal)
    if use_sample_mean:
        pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
        st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
    else:
        annual_pct = target_annual_pct if target_annual_pct is not None else 2.0
        pi_star_quarterly = (annual_pct / 100.0) / 4.0
        st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

    # Fit models
    models_o = fit_models(
        df_est, pi_star_quarterly,
        spec_mode=spec_mode,
        is_selected_custom=(is_selected if spec_mode == "Custom (pick variables)" else None),
        robust=robust_se
    )

    # Anchors & means for simulation
    i_mean_dec = float(df_est["Nominal Rate"].mean())
    means_o = {
        "RR_exante_L2": float(df_est["RR_exante_L2"].mean()),
        "Dlog FD_Lag1": float(df_est["Dlog FD_Lag1"].mean()),
        "Dlog_REER_L1": float(df_est["Dlog_REER_L1"].mean()),
        "Dlog_Energy_L1": float(df_est["Dlog_Energy_L1"].mean()),
        "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
        "Dlog_Reer_L2": float(df_est["Dlog_Reer_L2"].mean()),
    }

    # Build shocks & simulate
    is_arr, pc_arr, pol_arr = build_shocks(
        T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
    )
    g0, p0, i0 = simulate(
        T, rho_sim, df_est, models_o, means_o, i_mean_dec, pi_star_quarterly
    )
    gS, pS, iS = simulate(
        T, rho_sim, df_est, models_o, means_o, i_mean_dec, pi_star_quarterly,
        is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr
    )

    # Plot IRFs
    plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    quarters = np.arange(T)
    vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

    axes[0].plot(quarters, g0*100, label="Baseline", linewidth=2)
    axes[0].plot(quarters, gS*100, label="Shock", linewidth=2)
    axes[0].axvline(shock_quarter, **vline_kwargs)
    axes[0].set_title("Real GDP Growth (DlogGDP, %)"); axes[0].set_ylabel("%")
    axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

    axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
    axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
    axes[1].axvline(shock_quarter, **vline_kwargs)
    axes[1].set_title("Inflation (DlogCPI, %)"); axes[1].set_ylabel("%")
    axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

    axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
    axes[2].plot(quarters, iS, label="Shock", linewidth=2)
    axes[2].axvline(shock_quarter, **vline_kwargs)
    axes[2].set_title("Nominal Policy Rate (decimal)")
    axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("decimal")
    axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

    plt.tight_layout(); st.pyplot(fig)

    # ===== Equations =====
    st.subheader("Estimated Equations")
    m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
    alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
    rho_hat = models_o["rho_hat"]

    # IS latex
    pretty_map_is = {
        "RR_exante_L2": r"RR^{ex\text{-}ante}_{t-2}",
        "Dlog FD_Lag1": r"\Delta \log FD_{t-1}",
        "Dlog_REER_L1": r"\Delta \log REER_{t-1}",
        "Dlog_Energy_L1": r"\Delta \log Energy_{t-1}",
        "Dlog_Non_Energy_L1": r"\Delta \log NonEnergy_{t-1}",
        "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
    }
    is_terms = [(float(v), pretty_map_is.get(k, k)) for k, v in m_is.params.items() if k != "const"]
    st.markdown("**IS Curve (R Best Model)** — dependent variable: \\(\\Delta \\log GDP_t\\)")
    st.latex(build_latex_equation(float(m_is.params.get("const", 0.0)), is_terms, r"\Delta \log GDP_t", r"\varepsilon_t"))

    # Phillips latex
    pretty_map_pc = {
        "Dlog_CPI_L1": r"\Delta \log CPI_{t-1}",
        "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
        "Dlog_Reer_L2": r"\Delta \log REER_{t-2}",
        "Dlog_Energy_L1": r"\Delta \log Energy_{t-1}",
        "Dlog_Non_Energy_L1": r"\Delta \log NonEnergy_{t-1}",
    }
    pc_terms = [(float(v), pretty_map_pc.get(k, k)) for k, v in m_pc.params.items() if k != "const"]
    st.markdown("**Phillips Curve** — dependent variable: \\(\\Delta \\log CPI_t\\)")
    st.latex(build_latex_equation(float(m_pc.params.get("const", 0.0)), pc_terms, r"\Delta \log CPI_t", r"u_t"))

    # Taylor latex
    st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
    st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
    st.latex(r"i_t^\* \;=\; \alpha^\* \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,g_t")
    st.latex(
        r",\; ".join([
            rf"\rho = {rho_hat:.3f}",
            rf"\alpha^\* = {alpha_star:.3f}",
            rf"\phi_{{\pi}}^\* = {phi_pi_star:.3f}",
            rf"\phi_{{g}}^\* = {phi_g_star:.3f}",
            rf"\pi^\* = {pi_star_quarterly:.4f}",
        ])
    )

    with st.expander("OLS summaries (robust SE if selected)"):
        st.write("**IS (R Best / Custom)**"); st.text(m_is.summary().as_text())
        st.write("**Phillips**"); st.text(m_pc.summary().as_text())
        st.write("**Taylor**"); st.text(m_tr.summary().as_text())

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()







