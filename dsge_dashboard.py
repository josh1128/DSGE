# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Sidebar toggles to include/exclude regressors in each curve
#      - NEW: Per-regressor lag sliders (Lag k shifts series down by k => uses t-k)
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, and Taylor (tightening/easing)
#      - Policy shock behavior selector:
#          • Add after smoothing (default)
#          • Add to target (inside 1−ρ)
#          • Force local jump (override)
#      - LaTeX equations shown below charts (auto-updates to reflect selected vars)
#   2) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
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
st.title("DSGE IRF Dashboard — IS, Phillips, Taylor")
st.markdown(
    "- **Original**: GDP & CPI in **%** (Dlog × 100); **Nominal rate** in **decimal**.\n"
    "- **Taylor** uses **inflation gap**: \\(\\pi_t - \\pi^*\\).\n"
    "- Use the sidebar to **toggle variables** and set **lags** (Lag k ⇒ use value at t−k)."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:  # e.g., 3.2 means 3.2%
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

def make_real_rate(nom: pd.Series, infl: pd.Series) -> pd.Series:
    return nom - infl

def shift_safe(s: pd.Series, k: int) -> pd.Series:
    k = int(max(0, k))
    return s.shift(k) if k else s

# =========================
# Simple NK (built-in)
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
                    r_nat[t] += rho_sh * r_nat[t-1]
                elif shock == "cost":
                    u[t] += rho_sh * u[t-1]
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
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=0)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

        st.header("Inflation target for Taylor")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False)
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

        st.header("Policy shock behavior")
        policy_mode = st.radio(
            "Choose how the policy shock is applied",
            ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
            index=0,
            help=("• Add after smoothing: i_t = ρ i_{t-1} + (1−ρ) i*_t + ε_t^{pol}  "
                  "• Add to target: i_t = ρ i_{t-1} + (1−ρ)(i*_t + ε_t^{pol})  "
                  "• Force local jump: ensures tightening raises i_t vs i_{t-1} by at least the shock size.")
        )

        # =========================
        # NEW: Variable Toggles + Lags
        # =========================
        st.divider()
        st.header("Variable selection & lags (Lag k ⇒ use t−k)")

        # IS curve controls
        with st.expander("IS Curve regressors", expanded=True):
            use_is_gdp_lag = st.checkbox("Include DlogGDP lag", value=True)
            lag_is_gdp = st.number_input("Lag for DlogGDP (IS)", min_value=0, max_value=12, value=1, step=1)

            use_is_rr = st.checkbox("Include Real Interest Rate", value=True, help="Nominal - DlogCPI")
            lag_is_rr = st.number_input("Lag for Real Rate (IS)", min_value=0, max_value=12, value=2, step=1)

            use_is_fd = st.checkbox("Include Dlog FD", value=True)
            lag_is_fd = st.number_input("Lag for Dlog FD", min_value=0, max_value=12, value=1, step=1)

            use_is_reer = st.checkbox("Include Dlog REER", value=True)
            lag_is_reer = st.number_input("Lag for Dlog REER", min_value=0, max_value=12, value=0, step=1)

            use_is_energy = st.checkbox("Include Dlog Energy", value=True)
            lag_is_energy = st.number_input("Lag for Dlog Energy", min_value=0, max_value=12, value=0, step=1)

            use_is_nonenergy = st.checkbox("Include Dlog Non-Energy", value=True)
            lag_is_nonenergy = st.number_input("Lag for Dlog Non-Energy", min_value=0, max_value=12, value=0, step=1)

        # Phillips curve controls
        with st.expander("Phillips Curve regressors", expanded=True):
            use_pc_cpi_lag = st.checkbox("Include DlogCPI lag", value=True)
            lag_pc_cpi = st.number_input("Lag for DlogCPI (Phillips)", min_value=0, max_value=12, value=1, step=1)

            use_pc_gdp_lag = st.checkbox("Include DlogGDP lag", value=True)
            lag_pc_gdp = st.number_input("Lag for DlogGDP (Phillips)", min_value=0, max_value=12, value=1, step=1)

            use_pc_reer = st.checkbox("Include Dlog REER", value=True)
            lag_pc_reer = st.number_input("Lag for Dlog REER (Phillips)", min_value=0, max_value=12, value=2, step=1)

            use_pc_energy = st.checkbox("Include Dlog Energy", value=True)
            lag_pc_energy = st.number_input("Lag for Dlog Energy (Phillips)", min_value=0, max_value=12, value=1, step=1)

            use_pc_nonenergy = st.checkbox("Include Dlog Non-Energy", value=True)
            lag_pc_nonenergy = st.number_input("Lag for Dlog Non-Energy (Phillips)", min_value=0, max_value=12, value=1, step=1)

        # Taylor rule controls
        with st.expander("Taylor Rule regressors", expanded=True):
            use_tr_nr_lag = st.checkbox("Include Nominal Rate lag (partial adjustment)", value=True)
            lag_tr_nr = st.number_input("Lag for Nominal Rate", min_value=1, max_value=12, value=1, step=1)

            use_tr_gap = st.checkbox("Include Inflation Gap (π − π*)", value=True)
            lag_tr_gap = st.number_input("Lag for Inflation Gap", min_value=0, max_value=12, value=0, step=1)

            use_tr_gdp = st.checkbox("Include DlogGDP", value=True)
            lag_tr_gdp = st.number_input("Lag for DlogGDP (Taylor)", min_value=0, max_value=12, value=0, step=1)

    else:
        st.info("**Which parameters affect which curve?**  \n"
                "• **IS (Demand)**: σ, ρx, ρr  \n"
                "• **Phillips (Supply)**: κ, γπ, ρu  \n"
                "• **Taylor Rule (Policy)**: φπ, φx, ρi")

        st.header("Simple NK parameters (pp units)")
        st.subheader("IS Curve (Demand)")
        sigma = st.slider("σ — Demand sensitivity denominator", 0.2, 5.0, 1.00, 0.05)
        rho_x = st.slider("ρx — Output persistence", 0.0, 0.98, 0.50, 0.02)
        rho_r = st.slider("ρr — Demand-shock persistence (r^n_t)", 0.0, 0.98, 0.80, 0.02)
        st.subheader("Phillips Curve (Supply)")
        kappa = st.slider("κ — Phillips slope", 0.01, 0.50, 0.10, 0.01)
        gamma_pi = st.slider("γπ — Inflation inertia", 0.0, 0.95, 0.50, 0.05)
        rho_u = st.slider("ρu — Cost-push shock persistence (u_t)", 0.0, 0.98, 0.50, 0.02)
        st.subheader("Taylor Rule (Policy)")
        phi_pi = st.slider("φπ — Response to inflation", 1.0, 3.0, 1.50, 0.05)
        phi_x = st.slider("φx — Response to output gap", 0.00, 1.00, 0.125, 0.005)
        rho_i = st.slider("ρi — Policy rate smoothing", 0.0, 0.98, 0.80, 0.02)

        st.divider()
        st.header("Shock")
        shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
        shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
        shock_quarter_nk = st.slider("Shock timing t", 1, T-1, 1, 1)
        shock_persist_nk = st.slider("Shock persistence ρ_shock", 0.0, 0.98, 0.80, 0.02)

# =========================
# ORIGINAL MODEL (DSGE.xlsx)
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

    # Normalize rate units
    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

    # Convenience bases (unlagged)
    df["Real_Rate_base"] = make_real_rate(df["Nominal Rate"], df["Dlog_CPI"])  # decimal
    # Note: Your file may already contain lagged variants like "Dlog FD_Lag1"; we still allow extra lagging.

    # We don't create lag columns here; we will shift on the fly based on user lag settings.
    # Build "estimation-ready" df by dropping rows that would become NA after maximum lag is applied later.
    return df, df.copy()

def build_design_matrix(df: pd.DataFrame, spec: Dict[str, Dict[str, int]]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Build X, y for a given curve based on 'spec':
      spec = {
        'y': {'name': 'DlogGDP'},      # dependent variable
        'vars': { 'key': {'base': 'colname-or-virtual', 'lag': k, 'label': 'Latex label', 'name': 'Xname'} , ... }
      }
    Special 'base' values supported:
      - 'Real_Rate_base' (computed as Nominal - Dlog_CPI)
      - 'Inflation_Gap' (computed as Dlog_CPI - pi_star_quarterly) -> handled outside here for Taylor
    """
    y_name = spec['y']['name']
    y = df[y_name]

    Xparts = {}
    labels = []
    for key, meta in spec['vars'].items():
        base = meta['base']; lag = int(meta.get('lag', 0)); name = meta.get('name', f"{base}_L{lag}")
        if base == "Real_Rate_base":
            series = df["Real_Rate_base"]
        else:
            series = df[base]
        Xparts[name] = shift_safe(series, lag)
        labels.append(meta.get('label', name))
    X = pd.DataFrame(Xparts, index=df.index)
    X = sm.add_constant(X, has_constant="add")
    return X, y, labels

def fit_models_original(
    df_est: pd.DataFrame,
    pi_star_quarterly: float,
    is_cfg: Dict[str, Dict[str, int]],
    pc_cfg: Dict[str, Dict[str, int]],
    tr_cfg: Dict[str, Dict[str, int]],
):
    # IS
    X_is, y_is, _ = build_design_matrix(df_est, is_cfg)
    df_is = pd.concat([y_is, X_is], axis=1).dropna()
    model_is = sm.OLS(df_is[is_cfg['y']['name']], df_is[X_is.columns]).fit()

    # Phillips
    X_pc, y_pc, _ = build_design_matrix(df_est, pc_cfg)
    df_pc = pd.concat([y_pc, X_pc], axis=1).dropna()
    model_pc = sm.OLS(df_pc[pc_cfg['y']['name']], df_pc[X_pc.columns]).fit()

    # Taylor: Inflation gap possibly lagged
    # Build a working frame with gap and its lag
    gap = df_est["Dlog_CPI"] - pi_star_quarterly
    gap_name = f"Inflation_Gap_L{tr_cfg['vars'].get('Inflation_Gap',{}).get('lag',0)}"
    df_tr_work = df_est.copy()
    df_tr_work[gap_name] = shift_safe(gap, tr_cfg['vars'].get('Inflation_Gap',{}).get('lag',0))

    # Build TR X via same utility, but substituting gap column name if used
    tr_vars_adj = {}
    for key, meta in tr_cfg['vars'].items():
        if key == "Inflation_Gap":
            tr_vars_adj[key] = {'base': gap_name, 'lag': 0, 'label': meta['label'], 'name': meta.get('name', gap_name)}
        else:
            tr_vars_adj[key] = meta
    tr_spec = {'y': tr_cfg['y'], 'vars': tr_vars_adj}
    X_tr, y_tr, _ = build_design_matrix(df_tr_work, tr_spec)
    df_tr = pd.concat([y_tr, X_tr], axis=1).dropna()
    model_tr = sm.OLS(df_tr[tr_cfg['y']['name']], df_tr[X_tr.columns]).fit()

    # Convert partial-adjustment to star-form if smoothing term is included
    b0 = float(model_tr.params.get("const", 0.0))
    # Find any column that corresponds to "Nominal Rate" lag regressor (we named it "NR_L{lag}")
    nr_term = [c for c in model_tr.params.index if c.startswith("NR_L")]
    rhoh = float(model_tr.params.get(nr_term[0], 0.0)) if nr_term else 0.0
    rhoh = min(max(rhoh, 0.0), 0.99)

    def safe_div(num, den):
        return num / den if abs(den) > 1e-8 else np.nan

    alpha_star = safe_div(b0, (1 - rhoh))
    # Extract the phi's based on which fields are present
    # Inflation gap term name:
    gap_term = [c for c in model_tr.params.index if c.startswith("Inflation_Gap")]
    bpi = float(model_tr.params.get(gap_term[0], 0.0)) if gap_term else 0.0
    gdp_term = [c for c in model_tr.params.index if c.startswith("DlogGDP")]
    bg = float(model_tr.params.get(gdp_term[0], 0.0)) if gdp_term else 0.0
    phi_pi_star = safe_div(bpi, (1 - rhoh)) if gap_term else np.nan
    phi_g_star = safe_div(bg, (1 - rhoh)) if gdp_term else np.nan

    return {
        "model_is": model_is, "model_pc": model_pc, "model_tr": model_tr,
        "alpha_star": alpha_star, "phi_pi_star": phi_pi_star, "phi_g_star": phi_g_star,
        "rho_hat": rhoh, "pi_star_quarterly": float(pi_star_quarterly),
        "gap_colname": gap_name,  # for display
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

def simulate_original(
    T: int, rho_sim: float, df_est: pd.DataFrame,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float, pi_star_quarterly: float,
    is_cfg: Dict[str, Dict[str, int]],
    pc_cfg: Dict[str, Dict[str, int]],
    tr_cfg: Dict[str, Dict[str, int]],
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None, policy_mode: str = "Add after smoothing (standard)"
):
    """
    Builds X_t each step using ONLY the regressors that were selected, with their chosen lags.
    Lag k means use value at t-k; for early t<k we fall back to sample means (or initial anchors).
    """
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]; model_tr = models["model_tr"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

    # Helper to fetch lagged simulated value with fallback
    def lag_fetch(arr, t, k, fallback):
        return float(arr[t-k]) if t - k >= 0 else float(fallback)

    for t in range(1, T):
        # ---------- IS ----------
        vals_is = {}
        for key, meta in is_cfg['vars'].items():
            lag = int(meta.get('lag', 0))
            colname = model_is.params.index
            name = meta.get('name', f"{meta['base']}_L{lag}")

            if key == "DlogGDP_lag":
                vals_is[name] = lag_fetch(g, t, lag, df_est["DlogGDP"].mean())
            elif key == "Real_Rate":
                # real rate = i - p
                rr_val = lag_fetch(i, t, lag, i_mean_dec) - lag_fetch(p, t, lag, df_est["Dlog_CPI"].mean())
                vals_is[name] = rr_val
            elif key == "Dlog_FD":
                vals_is[name] = means["Dlog FD_Lag1"]  # mean used; lag doesn't change mean
            elif key == "Dlog_REER":
                vals_is[name] = means["Dlog_REER"]
            elif key == "Dlog_Energy":
                vals_is[name] = means["Dlog_Energy"]
            elif key == "Dlog_NonEnergy":
                vals_is[name] = means["Dlog_NonEnergy"]

        Xis = row_from_params(model_is.params.index, vals_is)
        g[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        # ---------- Phillips ----------
        vals_pc = {}
        for key, meta in pc_cfg['vars'].items():
            lag = int(meta.get('lag', 0))
            name = meta.get('name', f"{meta['base']}_L{lag}")
            if key == "DlogCPI_lag":
                vals_pc[name] = lag_fetch(p, t, lag, df_est["Dlog_CPI"].mean())
            elif key == "DlogGDP_lag":
                vals_pc[name] = lag_fetch(g, t, lag, df_est["DlogGDP"].mean())
            elif key == "Dlog_REER":
                vals_pc[name] = means["Dlog_Reer_L2"]  # mean; lag not material here
            elif key == "Dlog_Energy":
                vals_pc[name] = means["Dlog_Energy_L1"]
            elif key == "Dlog_NonEnergy":
                vals_pc[name] = means["Dlog_Non_Energy_L1"]

        Xpc = row_from_params(model_pc.params.index, vals_pc)
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        # ---------- Taylor (target) ----------
        # Build i* either from star-form (if available) or by zeroing smoothing term in predict
        # First compute gap (possibly lagged per config)
        gap_lag = int(tr_cfg['vars'].get("Inflation_Gap", {}).get('lag', 0))
        pi_gap_t = (lag_fetch(p, t, gap_lag, df_est["Dlog_CPI"].mean()) - pi_star_quarterly)

        if not np.isnan(alpha_star) and (("Inflation_Gap" in model_tr.params.index) or any(k.startswith("Inflation_Gap") for k in model_tr.params.index) or any(k.startswith("DlogGDP") for k in model_tr.params.index)):
            i_star = (alpha_star
                      + (0.0 if np.isnan(phi_pi_star) else phi_pi_star) * pi_gap_t
                      + (0.0 if np.isnan(phi_g_star) else phi_g_star) * (lag_fetch(g, t, int(tr_cfg['vars'].get("DlogGDP",{}).get('lag',0)), df_est["DlogGDP"].mean()) if "DlogGDP" in tr_cfg['vars'] else 0.0))
        else:
            # Fallback: direct predict with i_{t-1...} set to 0 to emulate target
            vals_tr = {}
            # NR lag term(s): set to 0 in target calc
            for c in model_tr.params.index:
                if c == "const":
                    continue
                if c.startswith("NR_L"):
                    vals_tr[c] = 0.0
                elif c.startswith("Inflation_Gap"):
                    vals_tr[c] = pi_gap_t
                elif c.startswith("DlogGDP"):
                    vals_tr[c] = lag_fetch(g, t, int(tr_cfg['vars'].get("DlogGDP",{}).get('lag',0)), df_est["DlogGDP"].mean())
                else:
                    vals_tr[c] = 0.0
            Xtr_star = row_from_params(model_tr.params.index, vals_tr)
            i_star = float(model_tr.predict(Xtr_star).iloc[0])

        # Apply policy shock with smoothing
        eps = policy_shock_arr[t]
        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:  # Force local jump
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0:
                i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0:
                i_raw = min(i_raw, i[t - 1] - abs(eps))
        i[t] = float(i_raw)

    return g, p, i

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        file_source = xlf if 'xlf' in locals() and xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)

        # Determine π* (quarterly decimal)
        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
        else:
            annual_pct = target_annual_pct if 'target_annual_pct' in locals() and target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

        # Pack regressor inclusion + lag config for each equation
        # IS spec
        is_vars = {}
        if use_is_gdp_lag:   is_vars["DlogGDP_lag"] = {'base': 'DlogGDP', 'lag': lag_is_gdp, 'label': r"\Delta\log GDP_{t-%d}"%lag_is_gdp, 'name': f"DlogGDP_L{lag_is_gdp}"}
        if use_is_rr:        is_vars["Real_Rate"]   = {'base': 'Real_Rate_base', 'lag': lag_is_rr, 'label': r"RR_{t-%d}"%lag_is_rr, 'name': f"RR_L{lag_is_rr}"}
        if use_is_fd:        is_vars["Dlog_FD"]     = {'base': 'Dlog FD_Lag1', 'lag': lag_is_fd, 'label': r"\Delta\log FD_{t-%d}"%lag_is_fd, 'name': f"DlogFD_L{lag_is_fd}"}
        if use_is_reer:      is_vars["Dlog_REER"]   = {'base': 'Dlog_REER', 'lag': lag_is_reer, 'label': r"\Delta\log REER_{t-%d}"%lag_is_reer, 'name': f"DlogREER_L{lag_is_reer}"}
        if use_is_energy:    is_vars["Dlog_Energy"] = {'base': 'Dlog_Energy', 'lag': lag_is_energy, 'label': r"\Delta\log Energy_{t-%d}"%lag_is_energy, 'name': f"DlogEnergy_L{lag_is_energy}"}
        if use_is_nonenergy: is_vars["Dlog_NonEnergy"] = {'base': 'Dlog_NonEnergy', 'lag': lag_is_nonenergy, 'label': r"\Delta\log NonEnergy_{t-%d}"%lag_is_nonenergy, 'name': f"DlogNonEnergy_L{lag_is_nonenergy}"}

        is_cfg = {'y': {'name': 'DlogGDP'}, 'vars': is_vars}
        if not is_vars:
            raise ValueError("Select at least one regressor for IS.")

        # Phillips spec
        pc_vars = {}
        if use_pc_cpi_lag:     pc_vars["DlogCPI_lag"]   = {'base': 'Dlog_CPI', 'lag': lag_pc_cpi, 'label': r"\Delta\log CPI_{t-%d}"%lag_pc_cpi, 'name': f"DlogCPI_L{lag_pc_cpi}"}
        if use_pc_gdp_lag:     pc_vars["DlogGDP_lag"]   = {'base': 'DlogGDP', 'lag': lag_pc_gdp, 'label': r"\Delta\log GDP_{t-%d}"%lag_pc_gdp, 'name': f"DlogGDP_L{lag_pc_gdp}"}
        if use_pc_reer:        pc_vars["Dlog_REER"]     = {'base': 'Dlog_REER', 'lag': lag_pc_reer, 'label': r"\Delta\log REER_{t-%d}"%lag_pc_reer, 'name': f"DlogREER_L{lag_pc_reer}"}
        if use_pc_energy:      pc_vars["Dlog_Energy"]   = {'base': 'Dlog_Energy', 'lag': lag_pc_energy, 'label': r"\Delta\log Energy_{t-%d}"%lag_pc_energy, 'name': f"DlogEnergy_L{lag_pc_energy}"}
        if use_pc_nonenergy:   pc_vars["Dlog_NonEnergy"]= {'base': 'Dlog_NonEnergy', 'lag': lag_pc_nonenergy, 'label': r"\Delta\log NonEnergy_{t-%d}"%lag_pc_nonenergy, 'name': f"DlogNonEnergy_L{lag_pc_nonenergy}"}

        pc_cfg = {'y': {'name': 'Dlog_CPI'}, 'vars': pc_vars}
        if not pc_vars:
            raise ValueError("Select at least one regressor for Phillips.")

        # Taylor spec
        tr_vars = {}
        if use_tr_nr_lag: tr_vars["Nominal_Rate_Lag"] = {'base': 'Nominal Rate', 'lag': lag_tr_nr, 'label': r"i_{t-%d}"%lag_tr_nr, 'name': f"NR_L{lag_tr_nr}"}
        if use_tr_gap:    tr_vars["Inflation_Gap"]    = {'base': 'Inflation_Gap', 'lag': lag_tr_gap, 'label': r"(\pi-\pi^\*)_{t-%d}"%lag_tr_gap, 'name': f"Inflation_Gap_L{lag_tr_gap}"}
        if use_tr_gdp:    tr_vars["DlogGDP"]          = {'base': 'DlogGDP', 'lag': lag_tr_gdp, 'label': r"g_{t-%d}"%lag_tr_gdp, 'name': f"DlogGDP_L{lag_tr_gdp}"}

        tr_cfg = {'y': {'name': 'Nominal Rate'}, 'vars': tr_vars}
        if not tr_vars:
            raise ValueError("Select at least one regressor for Taylor.")

        # Fit with selected regressors and chosen lags
        models_o = fit_models_original(df_est, pi_star_quarterly, is_cfg, pc_cfg, tr_cfg)

        # Anchors & means
        i_mean_dec = float(df_est["Nominal Rate"].mean())
        real_rate_mean_dec = float(df_est["Real_Rate_base"].mean())
        means_o = {
            "Dlog FD_Lag1": float(df_est["Dlog FD_Lag1"].mean()),
            "Dlog_REER": float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy": float(df_est["Dlog_Energy"].mean()),
            "Dlog_NonEnergy": float(df_est["Dlog_NonEnergy"].mean()),
            "Dlog_Reer_L2": float(df_est["Dlog_Reer_L2"].mean()) if "Dlog_Reer_L2" in df_est.columns else float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy_L1": float(df_est["Dlog_Energy_L1"].mean()) if "Dlog_Energy_L1" in df_est.columns else float(df_est["Dlog_Energy"].mean()),
            "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()) if "Dlog_Non_Energy_L1" in df_est.columns else float(df_est["Dlog_NonEnergy"].mean()),
        }

        # Build shocks & simulate
        is_arr, pc_arr, pol_arr = build_shocks_original(
            T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
        )
        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_cfg=is_cfg, pc_cfg=pc_cfg, tr_cfg=tr_cfg, policy_mode=policy_mode
        )
        gS, pS, iS = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_cfg=is_cfg, pc_cfg=pc_cfg, tr_cfg=tr_cfg,
            is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr, policy_mode=policy_mode
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

        # Readout at the shock quarter
        if shock_target.startswith("Taylor"):
            delta_i_bp = (iS - i0)[shock_quarter] * 10000.0
            st.info(f"Δ policy rate at t={shock_quarter}: {delta_i_bp:.1f} bp  |  mode: {policy_mode}  |  ρ={rho_sim:.2f}")

        # ===== LaTeX equations (reflect chosen variables & lags) =====
        st.subheader("Estimated Equations (Original model)")

        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
        rho_hat = models_o["rho_hat"]

        # IS equation
        is_terms = []
        for k, v in m_is.params.items():
            if k == "const": continue
            # Build readable symbol from name suffix
            if k.startswith("DlogGDP_L"): sym = r"\Delta \log GDP_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("RR_L"):     sym = r"RR_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogFD_L"): sym = r"\Delta \log FD_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogREER_L"): sym = r"\Delta \log REER_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogEnergy_L"): sym = r"\Delta \log Energy_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogNonEnergy_L"): sym = r"\Delta \log NonEnergy_{t-" + k.split('_L')[-1] + "}"
            else: sym = k
            is_terms.append((float(v), sym))
        st.markdown("**IS Curve (\\(\\Delta \\log GDP_t\\))**")
        st.latex(build_latex_equation(float(m_is.params.get("const", 0.0)), is_terms, r"\Delta \log GDP_t", r"\varepsilon_t"))

        # Phillips equation
        pc_terms = []
        for k, v in m_pc.params.items():
            if k == "const": continue
            if k.startswith("DlogCPI_L"):  sym = r"\Delta \log CPI_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogGDP_L"): sym = r"\Delta \log GDP_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogREER_L"): sym = r"\Delta \log REER_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogEnergy_L"): sym = r"\Delta \log Energy_{t-" + k.split('_L')[-1] + "}"
            elif k.startswith("DlogNonEnergy_L"): sym = r"\Delta \log NonEnergy_{t-" + k.split('_L')[-1] + "}"
            else: sym = k
            pc_terms.append((float(v), sym))
        st.markdown("**Phillips Curve (\\(\\Delta \\log CPI_t\\))**")
        st.latex(build_latex_equation(float(m_pc.params.get("const", 0.0)), pc_terms, r"\Delta \log CPI_t", r"u_t"))

        # Taylor rule (display)
        st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
        if policy_mode.startswith("Add after"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
        elif policy_mode.startswith("Add to target"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\,\big(i_t^\* + \varepsilon^{\text{pol}}_t\big)")
        else:
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t \quad (\text{with local-jump override})")

        parts = [rf"\rho = {rho_hat:.3f}"]
        if not np.isnan(alpha_star): parts.append(rf"\alpha^\* = {alpha_star:.3f}")
        if not np.isnan(phi_pi_star): parts.append(rf"\phi_{{\pi}}^\* = {phi_pi_star:.3f}")
        if not np.isnan(phi_g_star): parts.append(rf"\phi_{{g}}^\* = {phi_g_star:.3f}")
        parts.append(rf"\pi^\* = {pi_star_quarterly:.4f}")
        st.latex(r"i_t^\* \;=\; \alpha^\* \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,g_t")
        st.latex(r",\; ".join(parts))

        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(models_o["model_is"].summary().as_text())
            st.write("**Phillips Curve**"); st.text(models_o["model_pc"].summary().as_text())
            st.write("**Taylor Rule**"); st.text(models_o["model_tr"].summary().as_text())

    else:
        # =========================
        # Simple NK (built-in)
        # =========================
        P = NKParamsSimple(sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
                           rho_i=rho_i, rho_x=rho_x, rho_r=rho_r, rho_u=rho_u, gamma_pi=gamma_pi)
        model = SimpleNK3EqBuiltIn(P)
        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))

        st.info("**Model key (Simple NK):**  "
                r"$x_t$ = output gap (pp),  "
                r"$\pi_t$ = inflation (pp),  "
                r"$i_t$ = nominal policy rate (pp).  "
                r"$r_t^n$ = demand/natural-rate shock (pp),  "
                r"$u_t$ = cost-push shock (pp).")

        # Baseline vs Shock
        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, shock_persist_nk)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, shock_persist_nk)

        # Plot IRFs (pp)
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(h, x0, linewidth=2, label="Baseline")
        axes[0].plot(h, xS, linewidth=2, label="Shock")
        axes[0].axvline(t0, **vline_kwargs); axes[0].set_title("Output Gap (x_t, pp)"); axes[0].set_ylabel("pp")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(h, pi0, linewidth=2, label="Baseline")
        axes[1].plot(h, piS, linewidth=2, label="Shock")
        axes[1].axvline(t0, **vline_kwargs); axes[1].set_title("Inflation (π_t, pp)"); axes[1].set_ylabel("pp")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(h, i0, linewidth=2, label="Baseline")
        axes[2].plot(h, iS, linewidth=2, label="Shock")
        axes[2].axvline(t0, **vline_kwargs); axes[2].set_title("Nominal Policy Rate (i_t, pp)")
        axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("pp")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        with st.expander("Simple NK equations"):
            st.latex(r"x_t = \rho_x x_{t-1} \;-\; \frac{1}{\sigma}\big( i_t - \pi_{t+1} - r^n_t \big)")
            st.latex(r"\pi_t = \gamma_\pi \pi_{t-1} \;+\; \kappa x_t \;+\; u_t")
            st.latex(r"i_t = \rho_i i_{t-1} \;+\; (1-\rho_i)(\phi_\pi \pi_t + \phi_x x_t) \;+\; \varepsilon^i_t")

        with st.expander("Symbol glossary (Simple NK)"):
            st.markdown(
                r"""
- **$x_t$** — Output gap (percentage points, pp)  
- **$\pi_t$** — Inflation (pp)  
- **$i_t$** — Nominal policy rate (pp)  
- **$r_t^n$** — Demand / natural-rate shock (pp)  
- **$u_t$** — Cost-push shock (pp)  
- **$\sigma$,\,$\rho_x$,\,$\rho_r$** — IS dynamics  
- **$\kappa$,\,$\gamma_\pi$,\,$\rho_u$** — Phillips dynamics  
- **$\phi_\pi$,\,$\phi_x$,\,$\rho_i$** — Taylor dynamics  
                """
            )

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()







