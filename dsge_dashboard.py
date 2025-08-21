# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#   2) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
#   3) New Keynesian (DSGE_Model2.xlsx): Output Gap, Inflation Rate, Nominal Interest Rate
#
# NOTES for Model 3 (NK file with spaces in column names):
#   Sheets / columns used:
#     • IS Curve:      Date, Output Gap, Nominal Interest Rate, Inflation Rate
#     • Phillips:      Date, Inflation Rate, Output Gap
#     • Taylor:        Date, Nominal Interest Rate, Inflation Gap, Output Gap
#   We reference columns with spaces using df["col with space"].
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Page setup
# =========================
st.set_page_config(page_title="DSGE IRF Dashboard", layout="wide")
st.title("DSGE Dashboard")

st.markdown("- Use the sidebar to choose a model and configure shocks/simulation.")

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    """Convert percent-style rates (e.g., 3.2) to decimal (0.032) if needed."""
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:
        return s / 100.0
    return s

def fmt_coef(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return f"+{s}" if x >= 0 else s

def build_latex_equation(const_val: float, terms: List[tuple], lhs: str, eps_symbol: str) -> str:
    if not terms:
        rhs_terms = ""
    else:
        rhs_terms = " ".join([f"{fmt_coef(c)}\\,{sym}" for (c, sym) in terms])
    eq = rf"""
    \begin{{aligned}}
    {lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol}
    \end{{aligned}}
    """
    return eq

def row_from_params(params_index: pd.Index, values: Dict[str, float]) -> pd.DataFrame:
    cols = list(params_index)
    row = {}
    for c in cols:
        if c == "const":
            row[c] = 1.0
        else:
            row[c] = float(values.get(c, 0.0))
    return pd.DataFrame([row], columns=cols)

# =========================
# Simple NK (built-in) — unchanged
# =========================
@dataclass
class NKParamsSimple:
    sigma: float = 1.00
    kappa: float = 0.10
    phi_pi: float = 1.50
    phi_x: float = 0.125
    rho_i: float = 0.80
    rho_x: float = 0.50
    rho_r: float = 0.80
    rho_u: float = 0.50
    gamma_pi: float = 0.50

class SimpleNK3EqBuiltIn:
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(self, shock: str = "demand", T: int = 24, size_pp: float = 1.0, t0: int = 0, rho_override: Optional[float] = None):
        p = self.p
        x = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
        r_nat = np.zeros(T); u = np.zeros(T); e_i = np.zeros(T)

        if shock == "demand":
            r_nat[t0] = size_pp
            rho_sh = rho_override if rho_override is not None else p.rho_r
        elif shock == "cost":
            u[t0] = size_pp
            rho_sh = rho_override if rho_override is not None else p.rho_u
        elif shock == "policy":
            e_i[t0] = size_pp
            rho_sh = None
        else:
            raise ValueError("shock must be 'demand','cost','policy'")

        for t in range(T):
            if t > t0:
                if shock == "demand":
                    r_nat[t] += (rho_sh or 0.0) * r_nat[t-1]
                elif shock == "cost":
                    u[t] += (rho_sh or 0.0) * u[t-1]

            x_lag = x[t-1] if t>0 else 0.0
            pi_lag = pi[t-1] if t>0 else 0.0
            i_lag = i[t-1] if t>0 else 0.0

            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                + e_i[t]
            )
            denom = 1.0 + (A_x / p.sigma)
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t] = num / max(denom, 1e-8)
            pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox(
        "Choose model version",
        ["Original (DSGE.xlsx)", "Simple NK (built-in)", "New Keynesian (DSGE_Model2.xlsx)"],
        index=0
    )

    st.header("Simulation horizon")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    # ===== Generic neutral rate for display (used by built-in NK)
    neutral_rate_pct = st.number_input(
        "Baseline neutral policy rate — % annual (display)",
        value=2.00, step=0.25, format="%.2f"
    )

# =========================
# ORIGINAL MODEL (unchanged parts kept)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # These columns are from your original workflow (kept for model 1 only)
    # If present, convert nominal rate to decimal
    if "Nominal Rate" in df.columns:
        df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

    # The original model expected these engineered fields:
    if all(c in df.columns for c in ["DlogGDP","Dlog_CPI"]):
        df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
        df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
        if "Nominal Rate" in df.columns:
            df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)
            df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L2_data",
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")
    return df, df_est

def fit_models_original(
    df_est: pd.DataFrame,
    pi_star_quarterly: float,
    is_selected: List[str],
    pc_selected: List[str],
    tr_selected: List[str],
):
    # IS
    if not is_selected:
        raise ValueError("Select at least one regressor for IS (besides constant).")
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # Phillips
    if not pc_selected:
        raise ValueError("Select at least one regressor for Phillips (besides constant).")
    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor with inflation gap
    infl_gap_full = df_est["Dlog_CPI"] - pi_star_quarterly
    df_tr = pd.DataFrame(index=df_est.index)
    if "Nominal_Rate_L1" in tr_selected:
        df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:
        df_tr["Inflation_Gap"] = infl_gap_full
    if "DlogGDP" in tr_selected:
        df_tr["DlogGDP"] = df_est["DlogGDP"]
    if df_tr.empty:
        raise ValueError("Select at least one regressor for Taylor (besides constant).")
    X_tr = sm.add_constant(df_tr, has_constant="add")
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    b0 = float(model_tr.params.get("const", 0.0))
    rhoh = float(model_tr.params.get("Nominal_Rate_L1", 0.0))
    rhoh = min(max(rhoh, 0.0), 0.99)

    def safe_div(num, den): return num / den if abs(den) > 1e-8 else np.nan

    alpha_star = safe_div(b0, (1 - rhoh))
    bpi = float(model_tr.params.get("Inflation_Gap", 0.0))
    bg  = float(model_tr.params.get("DlogGDP", 0.0))
    phi_pi_star = safe_div(bpi, (1 - rhoh)) if "Inflation_Gap" in model_tr.params.index else np.nan
    phi_g_star  = safe_div(bg,  (1 - rhoh)) if "DlogGDP" in model_tr.params.index else np.nan

    return {
        "model_is": model_is, "model_pc": model_pc, "model_tr": model_tr,
        "alpha_star": alpha_star, "phi_pi_star": phi_pi_star, "phi_g_star": phi_g_star,
        "rho_hat": rhoh, "pi_star_quarterly": float(pi_star_quarterly),
    }

def simulate_original(
    T: int, rho_sim: float, df_est: pd.DataFrame, models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float, pi_star_quarterly: float,
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None, policy_mode: str = "Add after smoothing (standard)"
):
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]; model_tr = models["model_tr"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

    for t in range(1, T):
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean_dec

        vals_is = {
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L2_data": rr_lag2,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": means["Dlog_Energy"],
            "Dlog_NonEnergy": means["Dlog_NonEnergy"],
        }
        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        vals_pc = {
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }
        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        pi_gap_t = p[t] - pi_star_quarterly
        if not np.isnan(alpha_star) and (("Inflation_Gap" in model_tr.params.index) or ("DlogGDP" in model_tr.params.index)):
            i_star = (alpha_star
                      + (0.0 if np.isnan(phi_pi_star) else phi_pi_star) * pi_gap_t
                      + (0.0 if np.isnan(phi_g_star) else phi_g_star) * g[t])
        else:
            vals_tr = {"Nominal_Rate_L1": 0.0, "Inflation_Gap": pi_gap_t, "DlogGDP": g[t]}
            Xtr_star = row_from_params(model_tr.params.index, vals_tr)
            i_star = float(model_tr.predict(Xtr_star).iloc[0])

        eps = policy_shock_arr[t]
        # Default: standard add-after-smoothing
        i[t] = float(rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps)

    return g, p, i

# =========================
# NEW KEYNESIAN (DSGE_Model2.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_nk(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load NK Excel (columns have spaces) and return merged dataframes for:
      - IS block (uses IS Curve sheet)
      - Phillips block (uses Phillips sheet)
      - Taylor block (uses Taylor sheet)
    Also returns a single merged 'df_all' aligned by Date for simulation.
    """
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE_Model2.xlsx or place it beside this script.")
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
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    # Keep only relevant columns (with spaces handled explicitly)
    is_keep = ["Date", "Output Gap", "Nominal Interest Rate", "Inflation Rate"]
    pc_keep = ["Date", "Inflation Rate", "Output Gap"]
    tr_keep = ["Date", "Nominal Interest Rate", "Inflation Gap", "Output Gap"]

    for need, df, name in [
        (is_keep, is_df, "IS Curve"),
        (pc_keep, pc_df, "Phillips"),
        (tr_keep, tr_df, "Taylor"),
    ]:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"[{name}] Missing required columns: {missing}")

    # Convert rates in percent to decimal if necessary
    is_df["Nominal Interest Rate"] = ensure_decimal_rate(is_df["Nominal Interest Rate"])
    is_df["Inflation Rate"] = ensure_decimal_rate(is_df["Inflation Rate"])
    pc_df["Inflation Rate"] = ensure_decimal_rate(pc_df["Inflation Rate"])
    tr_df["Nominal Interest Rate"] = ensure_decimal_rate(tr_df["Nominal Interest Rate"])
    # Inflation Gap is already a gap (decimal) in Taylor sheet; keep as-is.

    # Lags for NK regressions
    is_df = is_df.sort_values("Date").copy()
    is_df["Output Gap L1"] = is_df["Output Gap"].shift(1)
    is_df["Real Rate L1"] = (is_df["Nominal Interest Rate"].shift(1) - is_df["Inflation Rate"].shift(1))

    pc_df = pc_df.sort_values("Date").copy()
    pc_df["Inflation Rate L1"] = pc_df["Inflation Rate"].shift(1)
    pc_df["Output Gap L1"] = pc_df["Output Gap"].shift(1)

    tr_df = tr_df.sort_values("Date").copy()
    # Optional smoothing support later:
    tr_df["Nominal Rate L1"] = tr_df["Nominal Interest Rate"].shift(1)

    # Estimation frames (drop NAs from needed cols)
    is_est = is_df.dropna(subset=["Output Gap", "Output Gap L1", "Real Rate L1"]).set_index("Date")
    pc_est = pc_df.dropna(subset=["Inflation Rate", "Inflation Rate L1", "Output Gap L1"]).set_index("Date")
    tr_est = tr_df.dropna(subset=["Nominal Interest Rate", "Inflation Gap", "Output Gap"]).set_index("Date")

    # Unified simulation frame on common dates
    df_all = (
        is_df[["Date","Output Gap","Nominal Interest Rate","Inflation Rate"]]
        .merge(pc_df[["Date","Inflation Rate"]], on="Date", suffixes=("","_pc"))
        .merge(tr_df[["Date","Inflation Gap","Output Gap"]], on="Date", suffixes=("","_tr"))
        .sort_values("Date")
        .set_index("Date")
    )

    return df_all, is_est, pc_est, tr_est

def fit_models_nk(is_est: pd.DataFrame, pc_est: pd.DataFrame, tr_est: pd.DataFrame, include_policy_smoothing: bool):
    """
    Fit the three NK regressions with only relevant variables.
      IS:       y_t  ~ const + y_{t-1} - (i_{t-1}-pi_{t-1})
      Phillips: pi_t ~ const + pi_{t-1} + y_{t-1}
      Taylor:   i_t  ~ const + (pi_t - pi*) + y_t [+ rho*i_{t-1} if smoothing]
    """
    # IS
    X_is = pd.DataFrame({
        "Output Gap L1": is_est["Output Gap L1"],
        "Real Rate L1":  is_est["Real Rate L1"]
    }, index=is_est.index)
    X_is = sm.add_constant(X_is, has_constant="add")
    y_is = is_est["Output Gap"]
    mdl_is = sm.OLS(y_is, X_is).fit()

    # Phillips
    X_pc = pd.DataFrame({
        "Inflation Rate L1": pc_est["Inflation Rate L1"],
        "Output Gap L1":    pc_est["Output Gap L1"]
    }, index=pc_est.index)
    X_pc = sm.add_constant(X_pc, has_constant="add")
    y_pc = pc_est["Inflation Rate"]
    mdl_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor (gap provided; allow optional smoothing)
    cols_tr = {"Inflation Gap": tr_est["Inflation Gap"], "Output Gap": tr_est["Output Gap"]}
    if include_policy_smoothing:
        cols_tr["Nominal Rate L1"] = tr_est["Nominal Rate L1"]
    X_tr = pd.DataFrame(cols_tr, index=tr_est.index)
    X_tr = sm.add_constant(X_tr, has_constant="add")
    y_tr = tr_est["Nominal Interest Rate"]
    mdl_tr = sm.OLS(y_tr, X_tr).fit()

    # Extract smoothing if present
    rho_hat = float(mdl_tr.params.get("Nominal Rate L1", 0.0)) if include_policy_smoothing else 0.0
    rho_hat = float(np.clip(rho_hat, 0.0, 0.99)) if include_policy_smoothing else 0.0

    return {"is": mdl_is, "pc": mdl_pc, "tr": mdl_tr, "rho_hat": rho_hat}

def simulate_nk(
    T: int,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    y0: float, pi0: float, i0: float,
    include_policy_smoothing: bool,
    shock_block: str = "None",
    shock_size: float = 0.0,
    shock_time: int = 1
):
    """
    Forward simulate Output Gap (y), Inflation Rate (pi), and Nominal Policy Rate (i)
    using the estimated NK regressions. Units are the same as the data (gaps in pp, rates in decimal).
    Shocks:
      • 'IS' adds to y_t
      • 'Phillips' adds to pi_t
      • 'Taylor' adds to i_t
    """
    y = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
    y[0] = y0; pi[0] = pi0; i[0] = i0

    m_is = models["is"]; m_pc = models["pc"]; m_tr = models["tr"]
    rho_hat = models["rho_hat"]

    for t in range(1, T):
        # --- IS: y_t = c + a*y_{t-1} + b*RealRate_{t-1}
        real_rate_l1 = (i[t-1] - pi[t-1])
        Xis = row_from_params(m_is.params.index, {"Output Gap L1": y[t-1], "Real Rate L1": real_rate_l1})
        y[t] = float(m_is.predict(Xis).iloc[0])

        # --- Phillips: pi_t = c + beta*pi_{t-1} + kappa*y_{t-1}
        Xpc = row_from_params(m_pc.params.index, {"Inflation Rate L1": pi[t-1], "Output Gap L1": y[t-1]})
        pi[t] = float(m_pc.predict(Xpc).iloc[0])

        # --- Taylor: i_t = c + ψπ*(π_t-π*) + ψy*y_t [+ ρ*i_{t-1}]
        tr_vals = {"Inflation Gap": (pi[t] - 0.0),  # Inflation Gap column already embodies π* in training; we use π gap ~ π here (π*≈0 baseline in deviations)
                   "Output Gap": y[t]}
        if include_policy_smoothing:
            tr_vals["Nominal Rate L1"] = i[t-1]
        Xtr = row_from_params(m_tr.params.index, tr_vals)
        i_t_pred = float(m_tr.predict(Xtr).iloc[0])
        i[t] = i_t_pred

        # --- Inject one-time shock at shock_time
        if t == shock_time:
            if shock_block == "IS":
                y[t] += shock_size
            elif shock_block == "Phillips":
                pi[t] += shock_size
            elif shock_block == "Taylor":
                i[t] += shock_size

        # --- If smoothing was estimated in the regression, it's already embedded via Nominal Rate L1

    return y, pi, i, rho_hat

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        with st.sidebar:
            xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
            fallback = Path(__file__).parent / "DSGE.xlsx"

            rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

            st.header("Inflation target for Taylor")
            use_sample_mean = st.checkbox("Use sample mean of DlogCPI as π*", value=False)
            if use_sample_mean:
                target_annual_pct = None
            else:
                target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)

            st.divider()
            st.header("Shock (Original)")
            shock_target = st.selectbox(
                "Apply shock to",
                ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
                index=0
            )
            is_shock_size_pp = st.number_input("IS shock (pp)", value=0.50, step=0.10, format="%.2f")
            pc_shock_size_pp = st.number_input("Phillips shock (pp)", value=0.10, step=0.05, format="%.2f")
            policy_shock_bp_abs = st.number_input("Policy shock (bp)", value=25, step=5, format="%d")
            shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1)
            shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

            st.divider()
            st.header("Variable selection")
            IS_ALL = ["DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"]
            PC_ALL = ["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
            TR_ALL = ["Nominal_Rate_L1", "Inflation_Gap", "DlogGDP"]
            is_selected = st.multiselect("IS regressors:", IS_ALL, default=IS_ALL)
            pc_selected = st.multiselect("Phillips regressors:", PC_ALL, default=PC_ALL)
            tr_selected = st.multiselect("Taylor regressors:", TR_ALL, default=TR_ALL)

        file_source = xlf if 'xlf' in locals() and xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)

        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* (quarterly) = sample mean of DlogCPI = {pi_star_quarterly:.4f}")
        else:
            annual_pct = target_annual_pct if 'target_annual_pct' in locals() and target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* = {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

        models_o = fit_models_original(df_est, pi_star_quarterly, is_selected, pc_selected, tr_selected)

        i_mean_dec = float(df_est["Nominal Rate"].mean())
        real_rate_mean_dec = float(df_est["Real_Rate_L2_data"].mean())
        means_o = {
            "Dlog FD_Lag1": float(df_est["Dlog FD_Lag1"].mean()),
            "Dlog_REER": float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy": float(df_est["Dlog_Energy"].mean()),
            "Dlog_NonEnergy": float(df_est["Dlog_NonEnergy"].mean()),
            "Dlog_Reer_L2": float(df_est["Dlog_Reer_L2"].mean()),
            "Dlog_Energy_L1": float(df_est["Dlog_Energy_L1"].mean()),
            "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
        }

        def build_shocks_original(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
            is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)
            if target == "IS (Demand)":
                is_arr[t0] = is_size_pp / 100.0
                for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
            elif target == "Phillips (Supply)":
                pc_arr[t0] = pc_size_pp / 100.0
                for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
            elif target == "Taylor (Policy tightening)":
                pol_arr[t0] =  (policy_bp_abs / 10000.0)
                for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]
            elif target == "Taylor (Policy easing)":
                pol_arr[t0] = -(policy_bp_abs / 10000.0)
                for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]
            return is_arr, pc_arr, pol_arr

        is_arr, pc_arr, pol_arr = build_shocks_original(
            T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
        )

        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly
        )
        gS, pS, iS = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr
        )

        # Plot
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        quarters = np.arange(T)
        axes[0].plot(quarters, g0*100, label="Baseline", linewidth=2)
        axes[0].plot(quarters, gS*100, label="Shock", linewidth=2)
        axes[0].set_title("Real GDP Growth (DlogGDP, %)"); axes[0].set_ylabel("%"); axes[0].grid(True, alpha=0.3); axes[0].legend()
        axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
        axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
        axes[1].set_title("Inflation (DlogCPI, %)"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3); axes[1].legend()
        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, iS, label="Shock", linewidth=2)
        axes[2].set_title("Nominal Policy Rate (decimal)"); axes[2].set_xlabel("Quarters"); axes[2].grid(True, alpha=0.3); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig)

        st.subheader("Estimated Equations (Original)")
        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        rho_hat = models_o["rho_hat"]; alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
        # (LaTeX render kept from previous version for brevity)
        with st.expander("OLS summaries"):
            st.write("**IS Curve**"); st.text(m_is.summary().as_text())
            st.write("**Phillips Curve**"); st.text(m_pc.summary().as_text())
            st.write("**Taylor Rule**"); st.text(m_tr.summary().as_text())

    elif model_choice == "Simple NK (built-in)":
        with st.sidebar:
            st.info("Simple NK parameters (pp units)")
            sigma = st.slider("σ (demand sensitivity denominator)", 0.2, 5.0, 1.00, 0.05)
            rho_x = st.slider("ρx — Output persistence", 0.0, 0.98, 0.50, 0.02)
            rho_r = st.slider("ρr — Demand-shock persistence", 0.0, 0.98, 0.80, 0.02)
            kappa = st.slider("κ — Phillips slope", 0.01, 0.50, 0.10, 0.01)
            gamma_pi = st.slider("γπ — Inflation inertia", 0.0, 0.95, 0.50, 0.05)
            rho_u = st.slider("ρu — Cost-push persistence", 0.0, 0.98, 0.50, 0.02)
            phi_pi = st.slider("φπ — Policy response to inflation", 1.0, 3.0, 1.50, 0.05)
            phi_x = st.slider("φx — Policy response to output gap", 0.00, 1.00, 0.125, 0.005)
            rho_i = st.slider("ρi — Policy smoothing", 0.0, 0.98, 0.80, 0.02)

            st.divider()
            shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
            shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
            shock_quarter_nk = st.slider("Shock timing t", 1, T-1, 1, 1)
            shock_persist_nk = st.slider("Shock persistence ρ_shock", 0.0, 0.98, 0.80, 0.02)
            snapback = st.checkbox("Snap-back (no persistence for x,π)", value=True)
            units_mode = st.radio("Policy rate units", ["Deviation (pp)", "Level (% annual)"], index=0)

        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
            rho_i=rho_i, rho_x=(0.0 if snapback else rho_x), rho_r=rho_r, rho_u=rho_u,
            gamma_pi=(0.0 if snapback else gamma_pi)
        )
        model = SimpleNK3EqBuiltIn(P)
        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))
        rho_for_shock = 0.0 if snapback else shock_persist_nk

        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

        i0_plot, iS_plot = i0.copy(), iS.copy(); i_ylabel = "pp"
        if units_mode == "Level (% annual)":
            i0_plot = neutral_rate_pct + i0_plot
            iS_plot = neutral_rate_pct + iS_plot
            i_ylabel = "%"

        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axes[0].plot(h, x0, linewidth=2, label="Baseline"); axes[0].plot(h, xS, linewidth=2, label="Shock")
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend()
        axes[1].plot(h, pi0, linewidth=2, label="Baseline"); axes[1].plot(h, piS, linewidth=2, label="Shock")
        axes[1].set_title("Inflation (pp)"); axes[1].set_ylabel("pp"); axes[1].grid(True, alpha=0.3); axes[1].legend()
        axes[2].plot(h, i0_plot, linewidth=2, label="Baseline"); axes[2].plot(h, iS_plot, linewidth=2, label="Shock")
        axes[2].set_title("Policy Rate"); axes[2].set_xlabel("Quarters"); axes[2].set_ylabel(i_ylabel); axes[2].grid(True, alpha=0.3); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig)

    else:
        # ===== New Keynesian (file with spaces) =====
        with st.sidebar:
            xlf2 = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"], key="upload_nk")
            fallback2 = Path(__file__).parent / "DSGE_Model2.xlsx"

            st.header("NK options")
            include_policy_smoothing = st.checkbox("Include policy smoothing (add i_{t-1} in Taylor)", value=False)

            st.divider()
            st.header("Shock (NK)")
            nk_block = st.selectbox("Shock block", ["None", "IS", "Phillips", "Taylor"], index=0)
            nk_size = st.number_input("Shock size (units of the variable)", value=0.00, step=0.10, format="%.2f")
            nk_time = st.slider("Shock timing (t)", 1, T-1, 1, 1)

        file_source2 = xlf2 if 'xlf2' in locals() and xlf2 is not None else (fallback2 if 'fallback2' in locals() else None)
        df_all, is_est, pc_est, tr_est = load_and_prepare_nk(file_source2)

        models_nk = fit_models_nk(is_est, pc_est, tr_est, include_policy_smoothing)

        # Initial values = sample means (simple anchor)
        y0 = float(is_est["Output Gap"].mean())
        pi0 = float(pc_est["Inflation Rate"].mean())
        i0 = float(tr_est["Nominal Interest Rate"].mean())

        yS, piS, iS, rho_hat = simulate_nk(
            T, models_nk, y0, pi0, i0,
            include_policy_smoothing=include_policy_smoothing,
            shock_block=nk_block, shock_size=nk_size, shock_time=nk_time
        )

        # Baseline (no shock) for comparison
        y0s, pi0s, i0s, _ = simulate_nk(
            T, models_nk, y0, pi0, i0,
            include_policy_smoothing=include_policy_smoothing,
            shock_block="None", shock_size=0.0, shock_time=nk_time
        )

        # Plot (Output Gap in pp, Inflation in decimal→% for readability, Rate in %)
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        q = np.arange(T)

        axes[0].plot(q, y0s, linewidth=2, label="Baseline")
        axes[0].plot(q, yS, linewidth=2, label="Shock")
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend()

        axes[1].plot(q, pi0s*100, linewidth=2, label="Baseline")
        axes[1].plot(q, piS*100, linewidth=2, label="Shock")
        axes[1].set_title("Inflation Rate (%)"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3); axes[1].legend()

        axes[2].plot(q, i0s*100, linewidth=2, label="Baseline")
        axes[2].plot(q, iS*100, linewidth=2, label="Shock")
        axes[2].set_title("Nominal Interest Rate (%)"); axes[2].set_ylabel("%"); axes[2].set_xlabel("Quarters"); axes[2].grid(True, alpha=0.3); axes[2].legend()

        plt.tight_layout(); st.pyplot(fig)

        # Show estimated NK equations in LaTeX
        st.subheader("Estimated Equations (New Keynesian)")
        mi, mp, mt = models_nk["is"], models_nk["pc"], models_nk["tr"]

        st.markdown("**IS (Output Gap)**")
        is_terms = []
        for k, v in mi.params.items():
            if k == "const": continue
            sym = {"Output Gap L1": r"\hat y_{t-1}", "Real Rate L1": r"(i_{t-1}-\pi_{t-1})"}.get(k, k)
            is_terms.append((float(v), sym))
        st.latex(build_latex_equation(float(mi.params.get("const", 0.0)), is_terms, r"\hat y_t", r"\varepsilon_t"))

        st.markdown("**Phillips Curve (Inflation)**")
        pc_terms = []
        for k, v in mp.params.items():
            if k == "const": continue
            sym = {"Inflation Rate L1": r"\pi_{t-1}", "Output Gap L1": r"\hat y_{t-1}"}.get(k, k)
            pc_terms.append((float(v), sym))
        st.latex(build_latex_equation(float(mp.params.get("const", 0.0)), pc_terms, r"\pi_t", r"u_t"))

        st.markdown("**Taylor Rule (No-smoothing by default)**")
        tr_terms = []
        for k, v in mt.params.items():
            if k == "const": continue
            sym = {"Inflation Gap": r"(\pi_t-\pi^\*)", "Output Gap": r"\hat y_t", "Nominal Rate L1": r"i_{t-1}"}.get(k, k)
            tr_terms.append((float(v), sym))
        st.latex(build_latex_equation(float(mt.params.get("const", 0.0)), tr_terms, r"i_t", r"v_t"))

        with st.expander("OLS summaries (NK)"):
            st.write("**IS**"); st.text(mi.summary().as_text())
            st.write("**Phillips**"); st.text(mp.summary().as_text())
            st.write("**Taylor**"); st.text(mt.summary().as_text())

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()








