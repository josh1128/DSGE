# dsge_dashboard_enhanced.py
# -----------------------------------------------------------
# Enhanced Streamlit DSGE Dashboard with:
#   - Equations display for all models
#   - Parameter explanations for Simple NK
#   - Improved NK dynamics with shock decay
#   - Better user functionality and code organization
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Page setup
# =========================
st.set_page_config(
    page_title="DSGE IRF Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏦 DSGE Model Dashboard")
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {gap: 24px;}
    .stTabs [data-baseweb="tab"] {height: 50px; padding-left: 20px; padding-right: 20px;}
    div[data-testid="metric-container"] {background-color: #f0f2f6; border-radius: 5px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# =========================
# Enhanced Helpers
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

def create_interactive_plots(quarters, baseline_data, shock_data, titles, ylabels, colors=['#1f77b4', '#ff7f0e']):
    """Create interactive Plotly plots instead of matplotlib"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=titles,
        vertical_spacing=0.12,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    for i, (b_data, s_data, ylabel) in enumerate(zip(baseline_data, shock_data, ylabels), 1):
        fig.add_trace(
            go.Scatter(x=quarters, y=b_data, mode='lines', name='Baseline',
                      line=dict(color=colors[0], width=2.5),
                      showlegend=(i==1)),
            row=i, col=1
        )
        fig.add_trace(
            go.Scatter(x=quarters, y=s_data, mode='lines', name='Shock',
                      line=dict(color=colors[1], width=2.5),
                      showlegend=(i==1)),
            row=i, col=1
        )
        fig.update_yaxes(title_text=ylabel, row=i, col=1, gridcolor='rgba(128,128,128,0.2)')
    
    fig.update_xaxes(title_text="Quarters", row=3, col=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_layout(
        height=900,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    return fig

# =========================
# Simple NK (built-in) with parameter explanations
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
# ORIGINAL MODEL
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

    if "Nominal Rate" in df.columns:
        df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

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
    if "Nominal_Rate_L1" in tr_selected:
        df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:
        df_tr["Inflation_Gap"] = infl_gap_full
    if "DlogGDP" in tr_selected:
        df_tr["DlogGDP"] = df_est["DlogGDP"]
    if df_tr.empty:
        raise ValueError("Select at least one regressor for Taylor.")
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
    T: int, rho_sim: float, df_est: pd.DataFrame, models: Dict,
    means: Dict[str, float], i_mean_dec: float, real_rate_mean_dec: float, pi_star_quarterly: float,
    is_shock_arr=None, pc_shock_arr=None, policy_shock_arr=None
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
        i[t] = float(rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps)

    return g, p, i

# =========================
# NEW KEYNESIAN MODEL with improved dynamics
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_nk(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    is_df["Nominal Interest Rate"] = ensure_decimal_rate(is_df["Nominal Interest Rate"])
    is_df["Inflation Rate"] = ensure_decimal_rate(is_df["Inflation Rate"])
    pc_df["Inflation Rate"] = ensure_decimal_rate(pc_df["Inflation Rate"])
    tr_df["Nominal Interest Rate"] = ensure_decimal_rate(tr_df["Nominal Interest Rate"])

    is_df = is_df.sort_values("Date").copy()
    is_df["Output Gap L1"] = is_df["Output Gap"].shift(1)
    is_df["Real Rate L1"] = (is_df["Nominal Interest Rate"].shift(1) - is_df["Inflation Rate"].shift(1))

    pc_df = pc_df.sort_values("Date").copy()
    pc_df["Inflation Rate L1"] = pc_df["Inflation Rate"].shift(1)
    pc_df["Output Gap L1"] = pc_df["Output Gap"].shift(1)

    tr_df = tr_df.sort_values("Date").copy()
    tr_df["Nominal Rate L1"] = tr_df["Nominal Interest Rate"].shift(1)

    is_est = is_df.dropna(subset=["Output Gap", "Output Gap L1", "Real Rate L1"]).set_index("Date")
    pc_est = pc_df.dropna(subset=["Inflation Rate", "Inflation Rate L1", "Output Gap L1"]).set_index("Date")
    tr_est = tr_df.dropna(subset=["Nominal Interest Rate", "Inflation Gap", "Output Gap"]).set_index("Date")

    df_all = (
        is_df[["Date","Output Gap","Nominal Interest Rate","Inflation Rate"]]
        .merge(pc_df[["Date","Inflation Rate"]], on="Date", suffixes=("","_pc"))
        .merge(tr_df[["Date","Inflation Gap","Output Gap"]], on="Date", suffixes=("","_tr"))
        .sort_values("Date")
        .set_index("Date")
    )

    return df_all, is_est, pc_est, tr_est

def fit_models_nk(is_est: pd.DataFrame, pc_est: pd.DataFrame, tr_est: pd.DataFrame, include_policy_smoothing: bool):
    X_is = pd.DataFrame({
        "Output Gap L1": is_est["Output Gap L1"],
        "Real Rate L1":  is_est["Real Rate L1"]
    }, index=is_est.index)
    X_is = sm.add_constant(X_is, has_constant="add")
    y_is = is_est["Output Gap"]
    mdl_is = sm.OLS(y_is, X_is).fit()

    X_pc = pd.DataFrame({
        "Inflation Rate L1": pc_est["Inflation Rate L1"],
        "Output Gap L1":    pc_est["Output Gap L1"]
    }, index=pc_est.index)
    X_pc = sm.add_constant(X_pc, has_constant="add")
    y_pc = pc_est["Inflation Rate"]
    mdl_pc = sm.OLS(y_pc, X_pc).fit()

    cols_tr = {"Inflation Gap": tr_est["Inflation Gap"], "Output Gap": tr_est["Output Gap"]}
    if include_policy_smoothing:
        cols_tr["Nominal Rate L1"] = tr_est["Nominal Rate L1"]
    X_tr = pd.DataFrame(cols_tr, index=tr_est.index)
    X_tr = sm.add_constant(X_tr, has_constant="add")
    y_tr = tr_est["Nominal Interest Rate"]
    mdl_tr = sm.OLS(y_tr, X_tr).fit()

    rho_hat = float(mdl_tr.params.get("Nominal Rate L1", 0.0)) if include_policy_smoothing else 0.0
    rho_hat = float(np.clip(rho_hat, 0.0, 0.99)) if include_policy_smoothing else 0.0

    return {"is": mdl_is, "pc": mdl_pc, "tr": mdl_tr, "rho_hat": rho_hat}

def simulate_nk(
    T: int,
    models: Dict,
    y0: float, pi0: float, i0: float,
    include_policy_smoothing: bool,
    shock_block: str = "None",
    shock_size: float = 0.0,
    shock_time: int = 1,
    shock_decay: float = 0.5  # NEW: decay rate for shocks
):
    y = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
    y[0] = y0; pi[0] = pi0; i[0] = i0

    m_is = models["is"]; m_pc = models["pc"]; m_tr = models["tr"]

    # Create decaying shock array
    shock_array = np.zeros(T)
    if shock_block != "None" and shock_time < T:
        for t in range(shock_time, T):
            shock_array[t] = shock_size * (shock_decay ** (t - shock_time))

    for t in range(1, T):
        real_rate_l1 = (i[t-1] - pi[t-1])
        Xis = row_from_params(m_is.params.index, {"Output Gap L1": y[t-1], "Real Rate L1": real_rate_l1})
        y[t] = float(m_is.predict(Xis).iloc[0])

        Xpc = row_from_params(m_pc.params.index, {"Inflation Rate L1": pi[t-1], "Output Gap L1": y[t-1]})
        pi[t] = float(m_pc.predict(Xpc).iloc[0])

        tr_vals = {"Inflation Gap": (pi[t] - 0.0), "Output Gap": y[t]}
        if "Nominal Rate L1" in m_tr.params.index:
            tr_vals["Nominal Rate L1"] = i[t-1]
        Xtr = row_from_params(m_tr.params.index, tr_vals)
        i[t] = float(m_tr.predict(Xtr).iloc[0])

        # Apply decaying shock
        if shock_block == "IS":
            y[t] += shock_array[t]
        elif shock_block == "Phillips":
            pi[t] += shock_array[t]
        elif shock_block == "Taylor":
            i[t] += shock_array[t]

    return y, pi, i

# =========================
# Sidebar Configuration
# =========================
with st.sidebar:
    st.header("📊 Model Selection")
    model_choice = st.selectbox(
        "Choose model version",
        ["Original (DSGE.xlsx)", "Simple NK (built-in)", "New Keynesian (DSGE_Model2.xlsx)"],
        index=0,
        help="Select the DSGE model variant to simulate"
    )

    st.header("⏱️ Simulation Settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1,
                  help="Number of quarters to simulate")

    neutral_rate_pct = st.number_input(
        "Baseline neutral policy rate (% annual)",
        value=2.00, step=0.25, format="%.2f",
        help="Used for display purposes in Simple NK model"
    )

# =========================
# Model-specific execution
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        with st.sidebar:
            st.header("📁 Data Source")
            xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
            fallback = Path(__file__).parent / "DSGE.xlsx"

            st.header("🎛️ Model Parameters")
            rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05,
                               help="Degree of interest rate smoothing in Taylor rule")

            st.subheader("Inflation Target")
            use_sample_mean = st.checkbox("Use sample mean of DlogCPI as π*", value=False)
            if not use_sample_mean:
                target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)

            st.header("💥 Shock Configuration")
            shock_target = st.selectbox(
                "Apply shock to",
                ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
                index=0
            )
            
            col1, col2 = st.columns(2)
            with col1:
                is_shock_size_pp = st.number_input("IS shock (pp)", value=0.50, step=0.10, format="%.2f")
                pc_shock_size_pp = st.number_input("Phillips shock (pp)", value=0.10, step=0.05, format="%.2f")
            with col2:
                policy_shock_bp_abs = st.number_input("Policy shock (bp)", value=25, step=5, format="%d")
                shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1)
            
            shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

            st.header("📋 Variable Selection")
            with st.expander("Customize regressors", expanded=False):
                IS_ALL = ["DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"]
                PC_ALL = ["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
                TR_ALL = ["Nominal_Rate_L1", "Inflation_Gap", "DlogGDP"]
                is_selected = st.multiselect("IS regressors:", IS_ALL, default=IS_ALL)
                pc_selected = st.multiselect("Phillips regressors:", PC_ALL, default=PC_ALL)
                tr_selected = st.multiselect("Taylor regressors:", TR_ALL, default=TR_ALL)

            st.header("📈 Display Options")
            units_mode_original = st.radio(
                "Taylor rate units",
                ["Deviation (pp)", "Level (% annual)"],
                index=1,
                help="Show policy rate as deviation from neutral or actual level"
            )

        # Main execution
        file_source = xlf if xlf is not None else fallback
        df_all, df_est = load_and_prepare_original(file_source)

        if use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
        else:
            pi_star_quarterly = (target_annual_pct / 100.0) / 4.0

        # Display info tabs
        tab1, tab2, tab3 = st.tabs(["📊 Simulation Results", "📐 Model Equations", "📋 Estimation Details"])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("π* (quarterly)", f"{pi_star_quarterly:.4f}")
            with col2:
                st.metric("Sample Size", len(df_est))
            with col3:
                neutral_original_pct = float(df_est["Nominal Rate"].mean() * 100.0)
                st.metric("Neutral Rate", f"{neutral_original_pct:.2f}%")

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
                    pol_arr[t0] = (policy_shock_bp_abs / 10000.0)
                    for k in range(t0 + 1, T): pol_arr[k] = rho * pol_arr[k - 1]
                elif target == "Taylor (Policy easing)":
                    pol_arr[t0] = -(policy_shock_bp_abs / 10000.0)
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

            # Prepare plotting series
            if units_mode_original == "Deviation (pp)":
                i0_plot = (i0 - i_mean_dec) * 100.0
                iS_plot = (iS - i_mean_dec) * 100.0
                i_title = "Policy Rate — deviation from neutral (pp)"
            else:
                i0_plot = i0 * 100.0
                iS_plot = iS * 100.0
                i_title = "Nominal Policy Rate (% annual)"

            # Create interactive plot
            fig = create_interactive_plots(
                np.arange(T),
                [g0*100, p0*100, i0_plot],
                [gS*100, pS*100, iS_plot],
                ["Real GDP Growth (DlogGDP, %)", "Inflation (DlogCPI, %)", i_title],
                ["%", "%", "pp" if units_mode_original == "Deviation (pp)" else "%"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("Original DSGE Model Equations")
            
            m_is = models_o["model_is"]
            m_pc = models_o["model_pc"]
            m_tr = models_o["model_tr"]
            
            st.subheader("IS Curve (Output/Demand)")
            st.latex(r"""
            \Delta \log GDP_t = \alpha_0 + \alpha_1 \Delta \log GDP_{t-1} + \alpha_2 (i_{t-2} - \pi_{t-2}) + \text{controls} + \varepsilon_t^{IS}
            """)
            st.caption("Where controls include fiscal deficit, REER, and commodity prices")
            
            st.subheader("Phillips Curve (Inflation)")
            st.latex(r"""
            \Delta \log CPI_t = \beta_0 + \beta_1 \Delta \log CPI_{t-1} + \beta_2 \Delta \log GDP_{t-1} + \text{controls} + \varepsilon_t^{PC}
            """)
            st.caption("Where controls include exchange rate and energy prices")
            
            st.subheader("Taylor Rule (Monetary Policy)")
            st.latex(r"""
            i_t = \rho i_{t-1} + (1-\rho)[\alpha^* + \phi_\pi (\pi_t - \pi^*) + \phi_y \Delta \log GDP_t] + \varepsilon_t^{MP}
            """)
            st.caption(f"Estimated ρ = {models_o['rho_hat']:.3f}, φ_π = {models_o['phi_pi_star']:.3f}, φ_y = {models_o['phi_g_star']:.3f}")

        with tab3:
            with st.expander("IS Curve Regression"):
                st.text(m_is.summary().as_text())
            with st.expander("Phillips Curve Regression"):
                st.text(m_pc.summary().as_text())
            with st.expander("Taylor Rule Regression"):
                st.text(m_tr.summary().as_text())

    elif model_choice == "Simple NK (built-in)":
        with st.sidebar:
            st.header("🎛️ NK Model Parameters")
            
            with st.expander("📚 Parameter Explanations", expanded=True):
                st.markdown("""
                **Demand Block:**
                - **σ**: Intertemporal elasticity of substitution (higher = less sensitive to interest rates)
                - **ρx**: Output gap persistence (higher = more inertia)
                - **ρr**: Natural rate shock persistence
                
                **Supply Block:**
                - **κ**: Phillips curve slope (higher = more responsive to output gap)
                - **γπ**: Inflation persistence (higher = more backward-looking)
                - **ρu**: Cost-push shock persistence
                
                **Policy Block:**
                - **φπ**: Taylor rule response to inflation (>1 for stability)
                - **φx**: Taylor rule response to output gap
                - **ρi**: Interest rate smoothing
                """)
            
            st.subheader("Demand Parameters")
            sigma = st.slider("σ (IES inverse)", 0.2, 5.0, 1.00, 0.05,
                             help="Inverse of intertemporal elasticity of substitution")
            rho_x = st.slider("ρx (Output persistence)", 0.0, 0.98, 0.50, 0.02,
                            help="Degree of output gap persistence")
            rho_r = st.slider("ρr (Demand shock persist.)", 0.0, 0.98, 0.80, 0.02,
                            help="Persistence of natural rate shocks")
            
            st.subheader("Supply Parameters")
            kappa = st.slider("κ (Phillips slope)", 0.01, 0.50, 0.10, 0.01,
                            help="Sensitivity of inflation to output gap")
            gamma_pi = st.slider("γπ (Inflation inertia)", 0.0, 0.95, 0.50, 0.05,
                                help="Degree of inflation persistence")
            rho_u = st.slider("ρu (Cost shock persist.)", 0.0, 0.98, 0.50, 0.02,
                            help="Persistence of cost-push shocks")
            
            st.subheader("Policy Parameters")
            phi_pi = st.slider("φπ (Inflation response)", 1.0, 3.0, 1.50, 0.05,
                              help="Policy response to inflation deviations")
            phi_x = st.slider("φx (Output response)", 0.00, 1.00, 0.125, 0.005,
                            help="Policy response to output gap")
            rho_i = st.slider("ρi (Rate smoothing)", 0.0, 0.98, 0.80, 0.02,
                            help="Degree of interest rate smoothing")

            st.header("💥 Shock Configuration")
            shock_type_nk = st.selectbox("Shock type", 
                                        ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], 
                                        index=0)
            shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
            shock_quarter_nk = st.slider("Shock timing", 1, T-1, 1, 1)
            shock_persist_nk = st.slider("Shock persistence", 0.0, 0.98, 0.80, 0.02)
            
            st.header("📈 Display Options")
            units_mode = st.radio("Policy rate units", 
                                 ["Deviation (pp)", "Level (% annual)"], 
                                 index=0)

        # Main execution
        tab1, tab2 = st.tabs(["📊 Simulation Results", "📐 Model Equations"])

        with tab1:
            P = NKParamsSimple(
                sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
                rho_i=rho_i, rho_x=rho_x, rho_r=rho_r, rho_u=rho_u,
                gamma_pi=gamma_pi
            )
            model = SimpleNK3EqBuiltIn(P)
            
            label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
            code = label_to_code[shock_type_nk]
            t0 = max(0, min(T-1, shock_quarter_nk - 1))

            h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, shock_persist_nk)
            h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, shock_persist_nk)

            i0_plot, iS_plot = i0.copy(), iS.copy()
            if units_mode == "Level (% annual)":
                i0_plot = neutral_rate_pct + i0_plot
                iS_plot = neutral_rate_pct + iS_plot
                i_title = "Nominal Policy Rate (% annual)"
            else:
                i_title = "Policy Rate Deviation (pp)"

            fig = create_interactive_plots(
                h,
                [x0, pi0, i0_plot],
                [xS, piS, iS_plot],
                ["Output Gap (pp)", "Inflation (pp)", i_title],
                ["pp", "pp", "%" if units_mode == "Level (% annual)" else "pp"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.header("Simple New Keynesian Model Equations")
            
            st.subheader("IS Curve (Dynamic IS)")
            st.latex(r"""
            \hat{y}_t = \rho_x \hat{y}_{t-1} - \frac{1}{\sigma}(i_t - \mathbb{E}_t[\pi_{t+1}] - r_t^n) + \varepsilon_t^{IS}
            """)
            st.caption(f"Current calibration: ρ_x = {rho_x:.2f}, σ = {sigma:.2f}")
            
            st.subheader("Phillips Curve (NKPC)")
            st.latex(r"""
            \pi_t = \gamma_\pi \pi_{t-1} + \kappa \hat{y}_t + u_t
            """)
            st.caption(f"Current calibration: γ_π = {gamma_pi:.2f}, κ = {kappa:.3f}")
            
            st.subheader("Taylor Rule")
            st.latex(r"""
            i_t = \rho_i i_{t-1} + (1-\rho_i)[\phi_\pi \pi_t + \phi_x \hat{y}_t] + \varepsilon_t^{MP}
            """)
            st.caption(f"Current calibration: ρ_i = {rho_i:.2f}, φ_π = {phi_pi:.2f}, φ_x = {phi_x:.3f}")
            
            st.subheader("Shock Processes")
            st.latex(r"""
            r_t^n = \rho_r r_{t-1}^n + \varepsilon_t^r, \quad u_t = \rho_u u_{t-1} + \varepsilon_t^u
            """)
            st.caption(f"Current calibration: ρ_r = {rho_r:.2f}, ρ_u = {rho_u:.2f}")

    else:  # New Keynesian (DSGE_Model2.xlsx)
        with st.sidebar:
            st.header("📁 Data Source")
            xlf2 = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"], key="upload_nk")
            fallback2 = Path(__file__).parent / "DSGE_Model2.xlsx"

            st.header("🎛️ NK Model Options")
            include_policy_smoothing = st.checkbox("Include policy smoothing (ρi)", value=False,
                                                  help="Add lagged interest rate to Taylor rule")
            
            st.header("💥 Shock Configuration")
            nk_block = st.selectbox("Shock target", ["None", "IS", "Phillips", "Taylor"], index=0)
            nk_size = st.number_input("Shock size", value=0.00, step=0.10, format="%.2f",
                                    help="Size in units of the target variable")
            nk_time = st.slider("Shock timing", 1, T-1, 1, 1)
            shock_decay_nk = st.slider("Shock decay rate", 0.0, 0.9, 0.5, 0.05,
                                      help="How quickly the shock fades (0=instant, 0.9=slow)")
            
            st.header("📈 Display Options")
            units_mode_nk = st.radio(
                "Taylor rate units",
                ["Deviation (pp)", "Actual (decimal)"],
                index=1,
                help="Show as deviation from neutral or actual decimal rate"
            )

        # Main execution
        file_source2 = xlf2 if xlf2 is not None else fallback2
        df_all, is_est, pc_est, tr_est = load_and_prepare_nk(file_source2)

        tab1, tab2, tab3 = st.tabs(["📊 Simulation Results", "📐 Model Equations", "📋 Estimation Details"])

        with tab1:
            models_nk = fit_models_nk(is_est, pc_est, tr_est, include_policy_smoothing)

            y0 = float(is_est["Output Gap"].mean())
            pi0 = float(pc_est["Inflation Rate"].mean())
            i0 = float(tr_est["Nominal Interest Rate"].mean())

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Initial Output Gap", f"{y0:.2f} pp")
            with col2:
                st.metric("Initial Inflation", f"{pi0*100:.2f}%")
            with col3:
                st.metric("Neutral Rate", f"{i0*100:.2f}%")

            yS, piS, iS = simulate_nk(
                T, models_nk, y0, pi0, i0,
                include_policy_smoothing=include_policy_smoothing,
                shock_block=nk_block, shock_size=nk_size, shock_time=nk_time,
                shock_decay=shock_decay_nk
            )

            y0s, pi0s, i0s = simulate_nk(
                T, models_nk, y0, pi0, i0,
                include_policy_smoothing=include_policy_smoothing,
                shock_block="None", shock_size=0.0, shock_time=nk_time
            )

            # Prepare plotting series
            if units_mode_nk == "Deviation (pp)":
                i0_plot = (i0s - i0) * 100.0
                iS_plot = (iS - i0) * 100.0
                i_title = "Policy Rate Deviation (pp)"
                i_label = "pp"
            else:
                i0_plot = i0s
                iS_plot = iS
                i_title = "Nominal Policy Rate (decimal)"
                i_label = "decimal"

            fig = create_interactive_plots(
                np.arange(T),
                [y0s, pi0s*100, i0_plot],
                [yS, piS*100, iS_plot],
                ["Output Gap (pp)", "Inflation Rate (%)", i_title],
                ["pp", "%", i_label]
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show actual rate changes
            if units_mode_nk == "Actual (decimal)" and nk_block == "Taylor":
                with st.expander("Policy Rate Response Analysis"):
                    rate_change = (iS[nk_time] - i0s[nk_time]) * 10000  # in basis points
                    st.info(f"Central bank {'raises' if rate_change > 0 else 'lowers'} rate by {abs(rate_change):.1f} basis points in response to the shock")

        with tab2:
            st.header("New Keynesian Model Equations")
            
            mi, mp, mt = models_nk["is"], models_nk["pc"], models_nk["tr"]
            
            st.subheader("IS Curve (Output Gap)")
            st.latex(r"""
            \hat{y}_t = \alpha_0 + \alpha_1 \hat{y}_{t-1} + \alpha_2 (i_{t-1} - \pi_{t-1}) + \varepsilon_t^{IS}
            """)
            st.caption(f"Estimated: α₁ = {mi.params.get('Output Gap L1', 0):.3f}, α₂ = {mi.params.get('Real Rate L1', 0):.3f}")
            
            st.subheader("Phillips Curve")
            st.latex(r"""
            \pi_t = \beta_0 + \beta_1 \pi_{t-1} + \beta_2 \hat{y}_{t-1} + \varepsilon_t^{PC}
            """)
            st.caption(f"Estimated: β₁ = {mp.params.get('Inflation Rate L1', 0):.3f}, β₂ = {mp.params.get('Output Gap L1', 0):.3f}")
            
            st.subheader("Taylor Rule")
            if include_policy_smoothing:
                st.latex(r"""
                i_t = \rho i_{t-1} + (1-\rho)[\delta + \psi_\pi (\pi_t - \pi^*) + \psi_y \hat{y}_t] + \varepsilon_t^{MP}
                """)
                st.caption(f"Estimated: ρ = {models_nk['rho_hat']:.3f}, ψ_π = {mt.params.get('Inflation Gap', 0):.3f}, ψ_y = {mt.params.get('Output Gap', 0):.3f}")
            else:
                st.latex(r"""
                i_t = \delta + \psi_\pi (\pi_t - \pi^*) + \psi_y \hat{y}_t + \varepsilon_t^{MP}
                """)
                st.caption(f"Estimated: ψ_π = {mt.params.get('Inflation Gap', 0):.3f}, ψ_y = {mt.params.get('Output Gap', 0):.3f}")

        with tab3:
            with st.expander("IS Curve Regression"):
                st.text(mi.summary().as_text())
            with st.expander("Phillips Curve Regression"):
                st.text(mp.summary().as_text())
            with st.expander("Taylor Rule Regression"):
                st.text(mt.summary().as_text())

except Exception as e:
    st.error(f"⚠️ Error: {e}")
    with st.expander("Debug Information"):
        st.exception(e)





