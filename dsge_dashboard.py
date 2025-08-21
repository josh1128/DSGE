# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#   2) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
#
# Edits in this version (Simple NK):
#   • Enforce mean-reverting dynamics back to steady state:
#       x* = 0 (pp), π* = target% annual, i* = neutral rate (decimal)
#     – Use parameters |ρx|<1, 0≤γπ<1, ρi<1 and shock persistence <1
#     – Optional "Force snap-back" makes x and π one-period (ρx=γπ=0, shock AR=0)
#   • Taylor rule now computed and shown in DECIMALS (levels), not deviations.
#     i_t(level) = ρ_i i_{t-1}(level) + (1-ρ_i)[ i* + φπ(π_t-π*) + φx x_t ] + ε^i_t(level)
#   • Inflation shown in % annual LEVEL (π* + deviations), rate shown in DECIMAL LEVEL.
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
st.set_page_config(page_title="DSGE IRF & Forecast Dashboard", layout="wide")
st.title("DSGE IRF & Forecast Dashboard — Original vs Simple NK")

st.markdown(
    "- **Original**: GDP & CPI in **%** (Dlog × 100); **Nominal rate** in **decimal**.\n"
    "- **Simple NK**: Output gap in **pp** (percentage points), inflation shown in **% annual (level)**, policy rate shown in **decimal (level)**.\n"
    "- **Steady state** (Simple NK): x*=0, π*=target%, i*=neutral rate."
)

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
        row[c] = 1.0 if c == "const" else float(values.get(c, 0.0))
    return pd.DataFrame([row], columns=cols)

# =========================
# Simple NK (built-in)
# =========================
@dataclass
class NKParamsSimple:
    # Dynamics
    sigma: float = 1.00       # IS sensitivity denominator
    kappa: float = 0.10       # Phillips slope
    gamma_pi: float = 0.50    # Inflation inertia (0..1)
    rho_x: float = 0.50       # Output persistence (0..1)
    # Policy
    phi_pi: float = 1.50      # Taylor response to inflation gap
    phi_x: float = 0.125      # Taylor response to output gap
    rho_i: float = 0.80       # Policy rate smoothing (0..1)
    # Shock persistence (AR parts)
    rho_r: float = 0.80       # Demand shock
    rho_u: float = 0.50       # Cost-push shock

class SimpleNK3EqBuiltIn:
    """
    3-eq NK (reduced-form, contemporaneous) with policy in *levels (decimal)* and
    inflation in *levels (% annual)* for display; core state vars use deviations.
    """
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(
        self,
        shock: str = "demand",
        T: int = 24,
        size_pp: float = 1.0,
        t0: int = 0,
        rho_override: Optional[float] = None,
        pi_star_annual_pct: float = 2.0,   # π* (% annual)
        i_star_neutral_pct: float = 2.0    # i* (% annual)
    ):
        """
        Returns (quarters, x(pp), pi_level(% annual), i_level(decimal)).
        Internally:
          • x_t  in pp
          • tilde_pi_t = (π_t - π*) in decimal (annual)
          • i_t(level) in decimal (annual)
        """
        p = self.p

        # Convert steady-state targets to appropriate units
        pi_star_dec = pi_star_annual_pct / 100.0     # decimal annual
        i_star_dec  = i_star_neutral_pct / 100.0     # decimal annual

        # State arrays
        x = np.zeros(T)                         # pp
        tilde_pi = np.zeros(T)                  # decimal
        i_level = np.zeros(T)                   # decimal (level)

        # Shock processes (units: pp for demand/cost; decimal for policy level shock)
        r_nat = np.zeros(T)                     # demand shock in pp
        u = np.zeros(T)                         # cost-push in pp -> will convert to decimal
        e_i_level = np.zeros(T)                 # policy rate level shock in DECIMAL

        # Initialize shock at t0
        if shock == "demand":
            r_nat[t0] = size_pp
            rho_sh = p.rho_r if rho_override is None else rho_override
        elif shock == "cost":
            u[t0] = size_pp
            rho_sh = p.rho_u if rho_override is None else rho_override
        elif shock == "policy":
            # interpret size_pp as *percentage points* for convenience, convert to decimal
            e_i_level[t0] = size_pp / 100.0
            rho_sh = None
        else:
            raise ValueError("shock must be 'demand','cost','policy'")

        # Simulate
        for t in range(T):
            # Persist shocks (AR(1))
            if t > t0:
                if shock == "demand":
                    r_nat[t] += (rho_sh or 0.0) * r_nat[t-1]
                elif shock == "cost":
                    u[t] += (rho_sh or 0.0) * u[t-1]
                elif shock == "policy":
                    e_i_level[t] += 0.0  # keep one-off unless you add a rho here

            # Lags
            x_lag = x[t-1] if t > 0 else 0.0                  # pp
            pi_lag = tilde_pi[t-1] if t > 0 else 0.0          # decimal
            i_lag = i_level[t-1] if t > 0 else i_star_dec     # decimal

            # ---- Algebra to solve for x_t with contemporaneous i_t and π_t
            # Use reduced-form structure from prior app; convert units carefully.
            # Map u(pp) -> decimal for Phillips add-on:
            u_dec = u[t] / 100.0

            # A_x and B_const (carry unit consistency). Here, p.kappa*x enters π (pp vs decimal):
            # Convert κ*x(pp) from pp into decimal: (κ*x)/100
            A_x = (1 - p.rho_i) * (p.phi_pi * (p.kappa/100.0) + p.phi_x) - (p.kappa/100.0)
            B_const = (
                p.rho_i * i_lag
                + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                + ((1 - p.rho_i) * p.phi_pi - 1.0) * u_dec
                + e_i_level[t]
                + (1 - p.rho_i) * i_star_dec   # target i* term leaks into algebra
            )
            denom = 1.0 + (A_x / p.sigma)
            # Natural rate r_nat is in pp -> convert to decimal effect via /100 and 1/σ scaling:
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + ((r_nat[t] / 100.0) / p.sigma)
            x[t] = num / max(denom, 1e-10)

            # Phillips (tilde_pi in decimal):
            tilde_pi[t] = p.gamma_pi * pi_lag + (p.kappa/100.0) * x[t] + u_dec

            # Taylor rule in LEVELS (DECIMAL):
            # i_t = ρ_i i_{t-1} + (1-ρ_i)[ i* + φπ * tilde_pi_t + φx * x_t ] + ε^i_t(level)
            i_target = i_star_dec + p.phi_pi * tilde_pi[t] + p.phi_x * (x[t] / 100.0)  # x: pp -> decimal contribution
            i_level[t] = p.rho_i * i_lag + (1 - p.rho_i) * i_target + e_i_level[t]

        # Convert inflation to LEVEL % annual for display:
        pi_level_pct = (tilde_pi + pi_star_dec) * 100.0  # %
        return np.arange(T), x, pi_level_pct, i_level, pi_star_dec, i_star_dec

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=1)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    # Global neutral rate used by Simple NK (level, % annual)
    neutral_rate_pct = st.number_input(
        "Baseline neutral nominal policy rate — % annual",
        value=2.00, step=0.25, format="%.2f",
        help="Neutral (steady-state) policy rate i* in percent (annual)."
    )

    if model_choice == "Original (DSGE.xlsx)":
        # (Original model controls kept as in your version)
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original",
                               help="If omitted, the app looks for 'DSGE.xlsx' next to this script.")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

        st.header("Inflation target for Taylor (Original)")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False)
        if use_sample_mean:
            target_annual_pct = None
            st.caption("π* will be set to sample mean (quarterly) after data loads.")
        else:
            target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)

        st.divider()
        st.header("Shock (Original)")
        shock_target = st.selectbox(
            "Apply shock to",
            ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
            index=0,
        )
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
        policy_shock_bp_abs = st.number_input("Policy shock size (bp)", value=25, step=5, format="%d")
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

    else:
        st.header("Simple NK parameters")
        with st.expander("IS / Phillips dynamics"):
            sigma = st.slider("σ — Demand sensitivity denominator", 0.2, 5.0, 1.00, 0.05)
            rho_x = st.slider("ρx — Output persistence", 0.0, 0.98, 0.50, 0.02)
            kappa = st.slider("κ — Phillips slope", 0.01, 0.50, 0.10, 0.01)
            gamma_pi = st.slider("γπ — Inflation inertia", 0.0, 0.95, 0.50, 0.05)
        with st.expander("Policy rule"):
            phi_pi = st.slider("φπ — Response to inflation gap", 1.0, 3.0, 1.50, 0.05)
            phi_x = st.slider("φx — Response to output gap", 0.00, 1.00, 0.125, 0.005)
            rho_i = st.slider("ρi — Policy smoothing", 0.0, 0.98, 0.80, 0.02)

        st.divider()
        st.header("Steady-state targets")
        pi_star_pct = st.number_input("Inflation target π* — % annual", value=2.00, step=0.10, format="%.2f")

        st.divider()
        st.header("Shock (Simple NK)")
        shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
        shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
        shock_quarter_nk = st.slider("Shock timing t", 1, T-1, 1, 1)
        shock_persist_nk = st.slider("Shock persistence ρ_shock", 0.0, 0.98, 0.80, 0.02)

        snapback_force = st.checkbox(
            "Force snap-back (one-period x & π, policy still smoothed)",
            value=False,
            help="Sets ρx=0, γπ=0 and ρ_shock=0 to guarantee immediate mean reversion. "
                 "Policy rate still decays with ρi."
        )

# =========================
# ORIGINAL MODEL (unchanged core logic)
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
             .sort_values("Date").set_index("Date")
    )

    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])
    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP","DlogGDP_L1","Dlog_CPI","Dlog_CPI_L1","Nominal Rate","Nominal_Rate_L1","Real_Rate_L2_data",
        "Dlog FD_Lag1","Dlog_REER","Dlog_Energy","Dlog_NonEnergy","Dlog_Reer_L2","Dlog_Energy_L1","Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns.")
    return df, df_est

def fit_models_original(
    df_est: pd.DataFrame,
    pi_star_quarterly: float,
    is_selected: List[str],
    pc_selected: List[str],
    tr_selected: List[str],
):
    if not is_selected:
        raise ValueError("Select at least one regressor for IS.")
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    if not pc_selected:
        raise ValueError("Select at least one regressor for Phillips.")
    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    infl_gap_full = df_est["Dlog_CPI"] - pi_star_quarterly
    df_tr = pd.DataFrame(index=df_est.index)
    if "Nominal_Rate_L1" in tr_selected: df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:   df_tr["Inflation_Gap"] = infl_gap_full
    if "DlogGDP" in tr_selected:         df_tr["DlogGDP"] = df_est["DlogGDP"]
    if df_tr.empty: raise ValueError("Select at least one regressor for Taylor.")
    X_tr = sm.add_constant(df_tr, has_constant="add")
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    b0 = float(model_tr.params.get("const", 0.0))
    rhoh = float(model_tr.params.get("Nominal_Rate_L1", 0.0))
    rhoh = min(max(rhoh, 0.0), 0.99)
    def safe_div(num, den): return num/den if abs(den)>1e-8 else np.nan
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

def build_shocks_original(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
    is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0+1, T): is_arr[k] = rho * is_arr[k-1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0+1, T): pc_arr[k] = rho * pc_arr[k-1]
    elif target == "Taylor (Policy tightening)":
        pol_arr[t0] = (policy_bp_abs / 10000.0)
        for k in range(t0+1, T): pol_arr[k] = rho * pol_arr[k-1]
    elif target == "Taylor (Policy easing)":
        pol_arr[t0] = -(policy_bp_abs / 10000.0)
        for k in range(t0+1, T): pol_arr[k] = rho * pol_arr[k-1]
    return is_arr, pc_arr, pol_arr

def simulate_original(
    T: int, rho_sim: float, df_est: pd.DataFrame, models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float, pi_star_quarterly: float,
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None, policy_mode: str = "Add after smoothing (standard)"
):
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    g[0] = float(df_est["DlogGDP"].mean()); p[0] = float(df_est["Dlog_CPI"].mean()); i[0] = i_mean_dec
    model_is = models["model_is"]; model_pc = models["model_pc"]; model_tr = models["model_tr"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]
    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

    for t in range(1, T):
        rr_lag2 = (i[t-2] - p[t-2]) if t >= 2 else real_rate_mean_dec
        vals_is = {"DlogGDP_L1": g[t-1], "Real_Rate_L2_data": rr_lag2,
                   "Dlog FD_Lag1": means["Dlog FD_Lag1"], "Dlog_REER": means["Dlog_REER"],
                   "Dlog_Energy": means["Dlog_Energy"], "Dlog_NonEnergy": means["Dlog_NonEnergy"]}
        g[t] = float(model_is.predict(row_from_params(model_is.params.index, vals_is)).iloc[0]) + is_shock_arr[t]

        vals_pc = {"Dlog_CPI_L1": p[t-1], "DlogGDP_L1": g[t-1],
                   "Dlog_Reer_L2": means["Dlog_Reer_L2"], "Dlog_Energy_L1": means["Dlog_Energy_L1"],
                   "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"]}
        p[t] = float(model_pc.predict(row_from_params(model_pc.params.index, vals_pc)).iloc[0]) + pc_shock_arr[t]

        pi_gap_t = p[t] - pi_star_quarterly
        if not np.isnan(alpha_star) and (("Inflation_Gap" in model_tr.params.index) or ("DlogGDP" in model_tr.params.index)):
            i_star = alpha_star + (0.0 if np.isnan(phi_pi_star) else phi_pi_star) * pi_gap_t + (0.0 if np.isnan(phi_g_star) else phi_g_star) * g[t]
        else:
            vals_tr = {"Nominal_Rate_L1": 0.0, "Inflation_Gap": pi_gap_t, "DlogGDP": g[t]}
            i_star = float(model_tr.predict(row_from_params(model_tr.params.index, vals_tr)).iloc[0])

        eps = policy_shock_arr[t]
        if policy_mode.startswith("Add after"):
            i[t] = float(rho_sim * i[t-1] + (1 - rho_sim) * i_star + eps)
        elif policy_mode.startswith("Add to target"):
            i[t] = float(rho_sim * i[t-1] + (1 - rho_sim) * (i_star + eps))
        else:
            raw = rho_sim * i[t-1] + (1 - rho_sim) * i_star + eps
            if eps > 0: raw = max(raw, i[t-1] + abs(eps))
            elif eps < 0: raw = min(raw, i[t-1] - abs(eps))
            i[t] = float(raw)
    return g, p, i

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        file_source = xlf if 'xlf' in locals() and xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)
        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
        else:
            annual_pct = target_annual_pct if 'target_annual_pct' in locals() and target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

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

        # Plots
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        q = np.arange(T); vline = dict(color="black", linestyle=":", linewidth=1)
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axes[0].plot(q, g0*100, label="Baseline", linewidth=2); axes[0].plot(q, gS*100, label="Shock", linewidth=2)
        axes[0].axvline(shock_quarter, **vline); axes[0].set_title("Real GDP Growth (DlogGDP, %)"); axes[0].set_ylabel("%"); axes[0].grid(True, alpha=0.3); axes[0].legend()
        axes[1].plot(q, p0*100, label="Baseline", linewidth=2); axes[1].plot(q, pS*100, label="Shock", linewidth=2)
        axes[1].axvline(shock_quarter, **vline); axes[1].set_title("Inflation (DlogCPI, %)"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3); axes[1].legend()
        axes[2].plot(q, i0, label="Baseline", linewidth=2); axes[2].plot(q, iS, label="Shock", linewidth=2)
        axes[2].axvline(shock_quarter, **vline); axes[2].set_title("Nominal Policy Rate (decimal)"); axes[2].set_xlabel("Quarters"); axes[2].set_ylabel("decimal"); axes[2].grid(True, alpha=0.3); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig)

        with st.expander("OLS summaries (Original)"):
            st.write("**IS**"); st.text(models_o["model_is"].summary().as_text())
            st.write("**Phillips**"); st.text(models_o["model_pc"].summary().as_text())
            st.write("**Taylor**"); st.text(models_o["model_tr"].summary().as_text())

    else:
        # =========================
        # Simple NK (edited)
        # =========================
        # Apply optional snap-back
        rho_x_eff = 0.0 if snapback_force else rho_x
        gamma_pi_eff = 0.0 if snapback_force else gamma_pi
        rho_shock_eff = 0.0 if snapback_force else shock_persist_nk

        params = NKParamsSimple(
            sigma=sigma, kappa=kappa, gamma_pi=gamma_pi_eff, rho_x=rho_x_eff,
            phi_pi=phi_pi, phi_x=phi_x, rho_i=rho_i, rho_r=0.80, rho_u=0.50
        )
        nk = SimpleNK3EqBuiltIn(params)
        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))

        # Baseline (no shock)
        h, x0, pi0_pct, i0_dec, pi_star_dec, i_star_dec = nk.irf(
            code, T, 0.0, t0, rho_override=rho_shock_eff,
            pi_star_annual_pct=pi_star_pct, i_star_neutral_pct=neutral_rate_pct
        )
        # Shock path
        h, xS, piS_pct, iS_dec, pi_star_dec, i_star_dec = nk.irf(
            code, T, shock_size_pp_nk, t0, rho_override=rho_shock_eff,
            pi_star_annual_pct=pi_star_pct, i_star_neutral_pct=neutral_rate_pct
        )

        # Plot
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        vline = dict(color="black", linestyle=":", linewidth=1)
        # Output gap (pp)
        axes[0].plot(h, x0, linewidth=2, label="Baseline")
        axes[0].plot(h, xS, linewidth=2, label="Shock")
        axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1)
        axes[0].axvline(t0, **vline)
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend()
        # Inflation level (% annual)
        axes[1].plot(h, pi0_pct, linewidth=2, label="Baseline")
        axes[1].plot(h, piS_pct, linewidth=2, label="Shock")
        axes[1].axhline(pi_star_pct, color="gray", linestyle="--", linewidth=1)
        axes[1].axvline(t0, **vline)
        axes[1].set_title("Inflation (level, % annual)"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3); axes[1].legend()
        # Policy rate level (decimal)
        axes[2].plot(h, i0_dec, linewidth=2, label="Baseline")
        axes[2].plot(h, iS_dec, linewidth=2, label="Shock")
        axes[2].axhline(i_star_dec, color="gray", linestyle="--", linewidth=1)
        axes[2].axvline(t0, **vline)
        axes[2].set_title("Policy Rate (level, decimal)"); axes[2].set_xlabel("Quarters"); axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3); axes[2].legend()
        plt.tight_layout(); st.pyplot(fig)

        # Readout
        st.info(
            f"Steady state: x*=0 pp, π*={pi_star_pct:.2f}% annual, i*={i_star_dec:.3f} (decimal).  "
            f"Snap-back={'ON' if snapback_force else 'OFF'} | ρx={rho_x_eff:.2f}, γπ={gamma_pi_eff:.2f}, ρi={rho_i:.2f}."
        )

        with st.expander("Simple NK equations (this app)"):
            st.latex(r"x_t = \rho_x x_{t-1} \;-\; \frac{1}{\sigma}\Big(i_t^{\text{level}} - \tilde{\pi}_t\Big) \;+\; r_t^n")
            st.latex(r"\tilde{\pi}_t = \gamma_\pi \tilde{\pi}_{t-1} \;+\; \frac{\kappa}{100}\,x_t \;+\; u_t")
            st.latex(r"i_t^{\text{level}} = \rho_i i_{t-1}^{\text{level}} \;+\; (1-\rho_i)\Big(i^\* + \phi_\pi \tilde{\pi}_t + \phi_x \frac{x_t}{100}\Big) \;+\; \varepsilon_t^{i}")
            st.markdown(
                "- \( \tilde{\pi}_t = \pi_t - \pi^\* \) is **decimal** (annual).  \n"
                "- \( x_t \) is **pp**.  \n"
                "- \( i_t^{\\text{level}} \) is **decimal** (annual).  \n"
                "- Shocks: demand \(r_t^n\) (pp), cost-push \(u_t\) (pp → decimal inside), policy \(\\varepsilon_t^i\) (decimal)."
            )

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()







