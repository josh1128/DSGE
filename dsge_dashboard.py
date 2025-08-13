# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that can run:
#   1) Original model (DSGE.xlsx): DlogGDP, Dlog_CPI, Taylor
#   2) NK model (DSGE_Model2.xlsx): Output Gap, Inflation Rate, Taylor
#   3) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
#
# Notes:
# - No steady-state lines/logic
# - "Scenario" renamed to "Shock"
# - Original & NK(Excel) plots: GDP/CPI in %, Nominal rate in DECIMAL
# - Simple NK (built-in): x, π, i all in percentage points (pp)
# - For Original/NK(Excel): nominal rate auto-converted from percent to DECIMAL
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional

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
    "- **Original / NK (Excel)**: GDP/CPI are plotted in **%**; **Nominal rate is plotted in decimal units**.\n"
    "- **Simple NK (built-in)**: Output gap, inflation, and nominal rate are all in **percentage points (pp)**.\n"
    "- Charts display **Baseline** vs **Shock** impulse responses."
)

# =========================
# Helpers (shared)
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    """
    Ensure a rate is in DECIMAL units.
    If the series looks like % levels (median abs > 1), divide by 100.
    """
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:  # typical % level like 3.34
        return s / 100.0
    return s

def _normalize_name(s: str) -> str:
    return pd.Series([s]).str.strip().str.replace(r"\s+", " ", regex=True).str.lower().iloc[0]

CANONICAL_MAP = {
    "date": "Date",
    "output gap": "Output Gap",
    "nominal interest rate": "Nominal Rate",
    "nominal rate": "Nominal Rate",
    "inflation rate": "Inflation Rate",
    "inflation gap": "Inflation Gap",
    "foreign demand": "Foreign Demand",
    "non-energy": "Non-Energy",
    "non energy": "Non-Energy",
    "energy": "Energy",
    "reer": "REER",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Trim/collapse spaces and map to canonical names when recognized (NK path)."""
    cleaned = []
    for c in df.columns:
        norm = _normalize_name(str(c))
        cleaned.append(CANONICAL_MAP.get(norm, pd.Series([c]).str.strip().str.replace(r"\s+", " ", regex=True).iloc[0]))
    out = df.copy()
    out.columns = cleaned
    return out

# =========================
# NEW: Simple NK (built-in) model
# =========================
@dataclass
class NKParamsSimple:
    # All in percentage points (pp)
    sigma: float = 1.00     # spending sensitivity (IS)
    kappa: float = 0.10     # NK Phillips slope
    phi_pi: float = 1.50    # Taylor weight on inflation
    phi_x: float = 0.125    # Taylor weight on output gap
    rho_i: float = 0.80     # interest rate smoothing
    rho_x: float = 0.50     # output gap persistence
    rho_r: float = 0.80     # demand (natural-rate) shock persistence
    rho_u: float = 0.50     # cost-push shock persistence
    gamma_pi: float = 0.50  # inflation inertia

class SimpleNK3EqBuiltIn:
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(self, shock: str = "demand", T: int = 24, size_pp: float = 1.0,
            t0: int = 0, rho_override: Optional[float] = None):
        """
        IRF for shocks ∈ {"demand","cost","policy"}.
        - size_pp: shock size in percentage points
        - t0: quarter when the shock hits (0-based)
        - rho_override: if provided, overrides rho_r (demand) or rho_u (cost)
        Returns: horizon, x(pp), pi(pp), i(pp)
        """
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
            rho_sh = None  # leave as one-off policy surprise
        else:
            raise ValueError("shock must be 'demand', 'cost', or 'policy'.")

        for t in range(T):
            # propagate the AR(1) demand/cost shock
            if t > t0:
                if shock == "demand":
                    r_nat[t] += rho_sh * r_nat[t-1]
                elif shock == "cost":
                    u[t] += rho_sh * u[t-1]
                # policy shock remains one-off

            x_lag  = x[t-1]  if t > 0 else 0.0
            pi_lag = pi[t-1] if t > 0 else 0.0
            i_lag  = i[t-1]  if t > 0 else 0.0

            gamma_pi = p.gamma_pi
            # compact algebra to solve contemporaneously for x_t
            A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
            B_const = (p.rho_i * i_lag
                       + ((1 - p.rho_i) * p.phi_pi * gamma_pi - gamma_pi) * pi_lag
                       + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                       + e_i[t])
            denom = 1.0 + (A_x / p.sigma)
            num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
            x[t] = num / max(denom, 1e-8)

            pi[t] = gamma_pi * pi_lag + p.kappa * x[t] + u[t]
            i[t]  = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

        return np.arange(T), x, pi, i

# =========================
# Sidebar: choose model + controls
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox(
        "Choose model version",
        ["Original (DSGE.xlsx)", "NK (DSGE_Model2.xlsx)", "Simple NK (built-in)"],
        index=0
    )

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)

    # File uploads + rho_sim only for Excel-backed models
    xlf = None
    fallback = None
    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
        fallback = Path(__file__).parent / "DSGE.xlsx"
        rho_sim = st.slider("Policy smoothing ρ (Taylor, 0 = fast, 0.95 = slow)", 0.0, 0.95, 0.80, 0.05)

        st.header("Shock")
        shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
        shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
        shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

    elif model_choice == "NK (DSGE_Model2.xlsx)":
        xlf = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"], key="upload_nk")
        fallback = Path(__file__).parent / "DSGE_Model2.xlsx"
        rho_sim = st.slider("Policy smoothing ρ (Taylor, 0 = fast, 0.95 = slow)", 0.0, 0.95, 0.80, 0.05)

        st.header("Shock")
        shock_target = st.selectbox("Apply shock to", ["None", "IS (Output Gap)", "Phillips (Inflation)"], index=0)
        is_shock_size_pp = st.number_input("IS shock (Δ Output Gap, pp)", value=0.50, step=0.10, format="%.2f")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ Inflation, pp)", value=0.10, step=0.05, format="%.2f")
        shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
        shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

    else:  # Simple NK (built-in)
        # Parameter sliders
        st.header("Simple NK parameters (pp units)")
        sigma = st.slider("σ (IS sensitivity; larger 1/σ ⇒ stronger rate effect)", 0.2, 5.0, 1.00, 0.05)
        kappa = st.slider("κ (Phillips slope)", 0.01, 0.50, 0.10, 0.01)
        phi_pi = st.slider("φπ (policy weight on inflation)", 1.0, 3.0, 1.50, 0.05)
        phi_x  = st.slider("φx (policy weight on output gap)", 0.00, 1.00, 0.125, 0.005)
        rho_i  = st.slider("ρi (policy smoothing)", 0.0, 0.98, 0.80, 0.02)
        rho_x  = st.slider("ρx (output persistence)", 0.0, 0.98, 0.50, 0.02)
        rho_r  = st.slider("ρr (demand shock persistence)", 0.0, 0.98, 0.80, 0.02)
        rho_u  = st.slider("ρu (cost shock persistence)", 0.0, 0.98, 0.50, 0.02)
        gamma_pi = st.slider("γπ (inflation inertia)", 0.0, 0.95, 0.50, 0.05)

        st.header("Shock")
        shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
        shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
        shock_quarter_nk = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
        shock_persist_nk = st.slider("Shock persistence ρ_shock (for demand/cost)", 0.0, 0.98, 0.80, 0.02)

# =========================
# ORIGINAL MODEL (DSGE.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path):
    if file_like_or_path is None:
        raise FileNotFoundError("No file provided. Upload DSGE.xlsx or include it beside this script.")
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    # Read sheets
    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    # Dates (YYYY-MM)
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    # Merge
    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Convert nominal to DECIMAL if it's in percent levels
    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

    # Build lags/transforms (Dlog series are already DECIMAL)
    df["DlogGDP_L1"]        = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]       = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]   = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L2_data",
        # IS externals
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        # Phillips externals (lagged)
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    return df, df_est

@st.cache_data(show_spinner=True)
def fit_models_original(df_est):
    # IS
    X_is = sm.add_constant(df_est[[
        "DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"
    ]])
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # Phillips
    X_pc = sm.add_constant(df_est[[
        "Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]])
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor (partial adjustment) — all in DECIMALS
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1", "Dlog_CPI", "DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    b0   = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal_Rate_L1"]), 0.99)
    bpi  = float(model_tr.params["Dlog_CPI"])
    bg   = float(model_tr.params["DlogGDP"])

    alpha_star  = b0  / (1 - rhoh)
    phi_pi_star = bpi / (1 - rhoh)
    phi_g_star  = bg  / (1 - rhoh)

    return {
        "model_is": model_is,
        "model_pc": model_pc,
        "model_tr": model_tr,
        "alpha_star": alpha_star,
        "phi_pi_star": phi_pi_star,
        "phi_g_star": phi_g_star
    }

def build_shocks_original(T, target, is_size_pp, pc_size_pp, t0, rho):
    """
    Inputs are in pp for UI. Convert to DECIMAL for Dlog variables.
    """
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

def simulate_original(T, rho_sim, df_est, models, means, i_mean_dec, real_rate_mean_dec, is_shock_arr=None, pc_shock_arr=None):
    g = np.zeros(T)  # DlogGDP (decimal)
    p = np.zeros(T)  # DlogCPI (decimal)
    i = np.zeros(T)  # Nominal rate (decimal)

    # Initialize at sample means (no explicit steady state)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean_dec

        Xis = pd.DataFrame([{
            "const": 1.0,
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L2_data": rr_lag2,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": means["Dlog_Energy"],
            "Dlog_NonEnergy": means["Dlog_NonEnergy"],
        }])
        g[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }])
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        i_star = alpha_star + phi_pi_star * p[t] + phi_g_star * g[t]
        i[t]   = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return g, p, i

# =========================
# NK MODEL (DSGE_Model2.xlsx)
# =========================
def _read_sheets_flex(excel_src):
    xl = pd.ExcelFile(excel_src)
    def _find(name):
        for s in xl.sheet_names:
            if s.strip().lower() == name.strip().lower():
                return s
        return None
    name_is = _find("IS Curve") or xl.sheet_names[0]
    name_pc = _find("Phillips") or xl.sheet_names[1]
    name_tr = _find("Taylor") or xl.sheet_names[2]
    is_df = standardize_columns(pd.read_excel(excel_src, sheet_name=name_is))
    pc_df = standardize_columns(pd.read_excel(excel_src, sheet_name=name_pc))
    tr_df = standardize_columns(pd.read_excel(excel_src, sheet_name=name_tr))
    return is_df, pc_df, tr_df

@st.cache_data(show_spinner=True)
def load_and_prepare_model2(file_like_or_path):
    if file_like_or_path is None:
        raise FileNotFoundError("No file provided. Upload DSGE_Model2.xlsx or include it beside this script.")
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    is_df, pc_df, tr_df = _read_sheets_flex(excel_src)

    def parse_date(series):
        out = pd.to_datetime(series, format="%Y-%m", errors="coerce")
        if out.isna().all():
            try:
                return pd.PeriodIndex(series.astype(str), freq="Q").to_timestamp(how="end")
            except Exception:
                return pd.to_datetime(series, errors="coerce")
        return out

    for df in (is_df, pc_df, tr_df):
        if "Date" not in df.columns:
            raise KeyError("Each sheet must contain a 'Date' column.")
        df["Date"] = parse_date(df["Date"])

    is_cols = ["Date", "Output Gap", "Nominal Rate", "Inflation Rate", "Foreign Demand", "Non-Energy", "Energy", "REER"]
    pc_cols = ["Date", "Inflation Rate", "Output Gap", "Foreign Demand", "Non-Energy", "Energy", "REER"]
    tr_cols = ["Date", "Nominal Rate", "Inflation Gap", "Output Gap"]
    for need, df, label in [(is_cols, is_df, "IS Curve"), (pc_cols, pc_df, "Phillips"), (tr_cols, tr_df, "Taylor")]:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns on '{label}' sheet: {missing}")

    tr_df = tr_df.rename(columns={"Nominal Rate": "Nominal Rate_Taylor", "Output Gap": "Output Gap_Taylor"})

    merged = (
        is_df.merge(pc_df, on="Date", how="inner", suffixes=("_IS", "_PC"))
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date").set_index("Date")
    )

    # Canonical frame (prefer IS externals + Taylor nominal rate)
    df = pd.DataFrame(index=merged.index)
    df["Output Gap"]     = merged["Output Gap_IS"]     # pp
    df["Inflation Rate"] = merged["Inflation Rate_IS"] # may be decimal or %
    df["Nominal Rate"]   = merged["Nominal Rate_Taylor"] if "Nominal Rate_Taylor" in merged else merged["Nominal Rate_IS"]
    df["Foreign Demand"] = merged["Foreign Demand_IS"]
    df["Non-Energy"]     = merged["Non-Energy_IS"]
    df["Energy"]         = merged["Energy_IS"]
    df["REER"]           = merged["REER_IS"]

    # Fill NAs (interpolate + edge-fill)
    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # Convert rates to DECIMAL if needed
    df["Inflation Rate"] = ensure_decimal_rate(df["Inflation Rate"])
    df["Nominal Rate"]   = ensure_decimal_rate(df["Nominal Rate"])

    # Lags & helpers
    df["Real Rate"]         = df["Nominal Rate"] - df["Inflation Rate"]  # decimal
    df["Output Gap_L1"]     = df["Output Gap"].shift(1)                  # pp
    df["Inflation Rate_L1"] = df["Inflation Rate"].shift(1)              # decimal
    df["Nominal Rate_L1"]   = df["Nominal Rate"].shift(1)                # decimal
    df["Real Rate_L1"]      = df["Real Rate"].shift(1)                   # decimal

    req_for_est = [
        "Output Gap", "Output Gap_L1", "Real Rate_L1",
        "Inflation Rate", "Inflation Rate_L1",
        "Nominal Rate", "Nominal Rate_L1",
        "Foreign Demand", "Non-Energy", "Energy", "REER",
    ]
    df_est = df.dropna(subset=req_for_est).copy()
    if df_est.empty:
        raise ValueError("After interpolation, no rows remain for estimation — check your data coverage.")

    return df, df_est

@st.cache_data(show_spinner=True)
def fit_models_model2(df_est):
    # IS: y_t (pp) on y_{t-1} (pp), r_{t-1} (decimal), externals (L1)
    X_is = sm.add_constant(pd.DataFrame({
        "Output Gap_L1": df_est["Output Gap_L1"],
        "Real Rate_L1": df_est["Real Rate_L1"],
        "Foreign Demand_L1": df_est["Foreign Demand"].shift(1),
        "NonEnergy_L1": df_est["Non-Energy"].shift(1),
        "Energy_L1": df_est["Energy"].shift(1),
        "REER_L1": df_est["REER"].shift(1),
    }))
    y_is = df_est["Output Gap"]
    is_ok = X_is.dropna().index.intersection(y_is.dropna().index)
    model_is = sm.OLS(y_is.loc[is_ok], X_is.loc[is_ok]).fit()

    # Phillips: pi_t (decimal) on pi_{t-1} (decimal), y_{t-1} (pp) + externals
    X_pc = sm.add_constant(pd.DataFrame({
        "Inflation Rate_L1": df_est["Inflation Rate_L1"],
        "Output Gap_L1": df_est["Output Gap_L1"],  # trained in pp
        "Foreign Demand_L1": df_est["Foreign Demand"].shift(1),
        "NonEnergy_L1": df_est["Non-Energy"].shift(1),
        "Energy_L1": df_est["Energy"].shift(1),
        "REER_L1": df_est["REER"].shift(1),
    }))
    y_pc = df_est["Inflation Rate"]
    pc_ok = X_pc.dropna().index.intersection(y_pc.dropna().index)
    model_pc = sm.OLS(y_pc.loc[pc_ok], X_pc.loc[pc_ok]).fit()

    # Taylor: i_t (decimal) on i_{t-1} (decimal), inflation gap (decimal), y_t (pp)
    infl_gap = df_est["Inflation Rate"] - float(df_est["Inflation Rate"].mean())
    X_tr = sm.add_constant(pd.DataFrame({
        "Nominal Rate_L1": df_est["Nominal Rate_L1"],
        "Inflation Gap": infl_gap,
        "Output Gap": df_est["Output Gap"],  # trained in pp
    }))
    y_tr = df_est["Nominal Rate"]
    tr_ok = X_tr.dropna().index.intersection(y_tr.dropna().index)
    model_tr = sm.OLS(y_tr.loc[tr_ok], X_tr.loc[tr_ok]).fit()

    b0   = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal Rate_L1"]), 0.99)
    bpi  = float(model_tr.params["Inflation Gap"])
    by   = float(model_tr.params["Output Gap"])

    alpha_star  = b0  / (1 - rhoh)
    phi_pi_star = bpi / (1 - rhoh)
    phi_y_star  = by  / (1 - rhoh)  # per pp of output gap

    return {
        "model_is": model_is,
        "model_pc": model_pc,
        "model_tr": model_tr,
        "alpha_star": alpha_star,
        "phi_pi_star": phi_pi_star,
        "phi_y_star": phi_y_star  # per pp (simulation scales as needed)
    }

def build_shocks_model2(T, target, is_size_pp, pc_size_pp, t0, rho):
    """
    Output gap shock is in pp; Inflation shock entered in pp but converted to DECIMAL.
    """
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Output Gap)":
        is_arr[t0] = is_size_pp  # pp
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Inflation)":
        pc_arr[t0] = pc_size_pp / 100.0  # decimal
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

def simulate_model2(T, rho_sim, df_est, models, means2, anchors, is_shock_arr=None, pc_shock_arr=None):
    """
    OPTION A:
      - 'Output Gap' stays in pp for plotting and IS equation.
      - Convert Output Gap to FRACTION (pp/100) when feeding into:
         * Phillips curve (activity term)
         * Taylor rule (activity term)
    """
    y = np.zeros(T); p = np.zeros(T); i = np.zeros(T)

    # Initialize at sample means
    y[0] = anchors["y_mean_pp"]
    p[0] = anchors["pi_mean_dec"]
    i[0] = anchors["i_mean_dec"]

    model_is = models["model_is"]; model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_y_star = models["phi_y_star"]  # per pp

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        # --- IS (uses pp and real rate in decimals) ---
        rr_l1 = (i[t - 1] - p[t - 1])  # decimal

        Xis = pd.DataFrame([{
            "const": 1.0,
            "Output Gap_L1": y[t - 1],                 # pp (as trained)
            "Real Rate_L1": rr_l1,
            "Foreign Demand_L1": means2["Foreign Demand"],
            "NonEnergy_L1": means2["Non-Energy"],
            "Energy_L1": means2["Energy"],
            "REER_L1": means2["REER"],
        }])
        y[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        # --- Phillips (feed OG in FRACTION units) ---
        og_frac_lag = y[t - 1] / 100.0  # pp -> fraction
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Inflation Rate_L1": p[t - 1],
            "Output Gap_L1": og_frac_lag,              # scaled at runtime
            "Foreign Demand_L1": means2["Foreign Demand"],
            "NonEnergy_L1": means2["Non-Energy"],
            "Energy_L1": means2["Energy"],
            "REER_L1": means2["REER"],
        }])
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        # --- Taylor (feed OG in FRACTION units) ---
        og_frac_now = y[t] / 100.0  # pp -> fraction
        pi_gap_t = p[t] - anchors["pi_mean_dec"]
        i_star = alpha_star + phi_pi_star * pi_gap_t + (phi_y_star * og_frac_now)
        i[t] = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return y, p, i

# =========================
# Run the chosen model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        # Load & fit
        file_source = xlf if xlf is not None else fallback
        df_all, df_est = load_and_prepare_original(file_source)
        models_o = fit_models_original(df_est)

        # Anchors & means
        i_mean_dec = float(df_est["Nominal Rate"].mean())
        real_rate_mean_dec = float(df_est["Real_Rate_L2_data"].mean())
        means_o = {
            "Dlog FD_Lag1":       float(df_est["Dlog FD_Lag1"].mean()),
            "Dlog_REER":          float(df_est["Dlog_REER"].mean()),
            "Dlog_Energy":        float(df_est["Dlog_Energy"].mean()),
            "Dlog_NonEnergy":     float(df_est["Dlog_NonEnergy"].mean()),
            "Dlog_Reer_L2":       float(df_est["Dlog_Reer_L2"].mean()),
            "Dlog_Energy_L1":     float(df_est["Dlog_Energy_L1"].mean()),
            "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
        }

        # Shocks & simulate
        is_arr, pc_arr = build_shocks_original(T, shock_target, is_shock_size_pp, pc_shock_size_pp, shock_quarter, shock_persist)
        g0, p0, i0 = simulate_original(T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec)
        gS, pS, iS = simulate_original(T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, is_arr, pc_arr)

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        quarters = np.arange(T); vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

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
        axes[2].set_title("Nominal Policy Rate (decimal)"); axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Diagnostics
        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(models_o["model_is"].summary().as_text())
            st.write("**Phillips Curve**"); st.text(models_o["model_pc"].summary().as_text())
            st.write("**Taylor Rule**"); st.text(models_o["model_tr"].summary().as_text())

    elif model_choice == "NK (DSGE_Model2.xlsx)":
        # Load & fit
        file_source = xlf if xlf is not None else fallback
        df_all, df_est = load_and_prepare_model2(file_source)
        models2 = fit_models_model2(df_est)

        # Anchors & means
        anchors = {
            "i_mean_dec": float(df_est["Nominal Rate"].mean()),   # decimal
            "y_mean_pp":  float(df_est["Output Gap"].mean()),     # pp
            "pi_mean_dec":float(df_est["Inflation Rate"].mean()), # decimal
        }
        means2 = {
            "Foreign Demand": float(df_est["Foreign Demand"].mean()),
            "Non-Energy": float(df_est["Non-Energy"].mean()),
            "Energy": float(df_est["Energy"].mean()),
            "REER": float(df_est["REER"].mean()),
        }

        # Shocks & simulate
        is_arr2, pc_arr2 = build_shocks_model2(T, shock_target, is_shock_size_pp, pc_shock_size_pp, shock_quarter, shock_persist)
        y0, p0, i0 = simulate_model2(T, rho_sim, df_est, models2, means2, anchors)
        yS, pS, iS = simulate_model2(T, rho_sim, df_est, models2, means2, anchors, is_arr2, pc_arr2)

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        quarters = np.arange(T); vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(quarters, y0, label="Baseline", linewidth=2)
        axes[0].plot(quarters, yS, label="Shock", linewidth=2)
        axes[0].axvline(shock_quarter, **vline_kwargs)
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
        axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation (%)"); axes[1].set_ylabel("%")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, iS, label="Shock", linewidth=2)
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (decimal)"); axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Diagnostics
        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(models2["model_is"].summary().as_text())
            st.write("**Phillips Curve**"); st.text(models2["model_pc"].summary().as_text())
            st.write("**Taylor Rule**"); st.text(models2["model_tr"].summary().as_text())

    else:
        # =========================
        # Simple NK (built-in) branch
        # =========================
        # Build params from sliders
        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
            rho_i=rho_i, rho_x=rho_x, rho_r=rho_r, rho_u=rho_u, gamma_pi=gamma_pi
        )
        model = SimpleNK3EqBuiltIn(P)

        # Map UI shock label -> internal code
        label_to_code = {
            "Demand (IS)": "demand",
            "Cost-push (Phillips)": "cost",
            "Policy (Taylor)": "policy",
        }
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))  # convert to 0-based

        # Baseline (size 0) vs Shock
        h, x0, pi0, i0 = model.irf(shock=code, T=T, size_pp=0.0, t0=t0, rho_override=shock_persist_nk)
        h, xS, piS, iS = model.irf(shock=code, T=T, size_pp=shock_size_pp_nk, t0=t0, rho_override=shock_persist_nk)

        # Plot IRFs (all in pp)
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(h, x0, label="Baseline", linewidth=2)
        axes[0].plot(h, xS, label="Shock", linewidth=2)
        axes[0].axvline(t0, **vline_kwargs)
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(h, pi0, label="Baseline", linewidth=2)
        axes[1].plot(h, piS, label="Shock", linewidth=2)
        axes[1].axvline(t0, **vline_kwargs)
        axes[1].set_title("Inflation (pp)"); axes[1].set_ylabel("pp")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(h, i0, label="Baseline", linewidth=2)
        axes[2].plot(h, iS, label="Shock", linewidth=2)
        axes[2].axvline(t0, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (pp)"); axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("pp")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Parameter readout (handy for screenshots/repro)
        with st.expander("Simple NK parameters (current)"):
            st.json({
                "sigma": sigma, "kappa": kappa, "phi_pi": phi_pi, "phi_x": phi_x,
                "rho_i": rho_i, "rho_x": rho_x, "rho_r": rho_r, "rho_u": rho_u, "gamma_pi": gamma_pi,
                "shock": code, "size_pp": shock_size_pp_nk, "t0": t0, "rho_shock": shock_persist_nk
            })

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()

