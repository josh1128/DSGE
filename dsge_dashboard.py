# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Sidebar toggles to include/exclude regressors in each curve
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, and Taylor (tightening/easing)
#      - Policy shock behavior selector:
#          • Add after smoothing (default)
#          • Add to target (inside 1−ρ)
#          • Force local jump (override)
#      - LaTeX equations shown below charts (auto-updates to reflect selected vars)
#   2) Simple NK: 3-eq NK DSGE-lite
#      - NEW: Sidebar "Data source" to choose **Simulated (built-in)** or
#              **Real (Excel: DSGE_Model2.xlsx)** with your specified columns
#      - For Real, we fit OLS for IS, Phillips, Taylor using your sheets/columns,
#        then forward-simulate IRFs with the same shock controls.
#      - "Snap-back (no persistence)" option makes x_t & π_t one-period while
#        KEEPING policy smoothing ρ_i so i_t decays geometrically (Simulated mode only).
#      - Toggle to show policy rate in **levels (% annual)** instead of deviations (pp).
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
    "- Use the sidebar to **toggle variables** in each curve."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    """Convert percent-style rates (e.g., 3.2) to decimal (0.032) if needed."""
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

# =========================
# Simple NK (built-in simulator)
# =========================
@dataclass
class NKParamsSimple:
    sigma: float = 1.00   # σ: demand sensitivity to real rate (higher = less sensitive)
    kappa: float = 0.10   # κ: slope of NK Phillips curve
    phi_pi: float = 1.50  # φπ: policy response to inflation
    phi_x: float = 0.125  # φx: policy response to output gap
    rho_i: float = 0.80   # ρi: interest rate smoothing
    rho_x: float = 0.50   # ρx: persistence of output gap
    rho_r: float = 0.80   # ρr: persistence of demand (natural-rate) shock
    rho_u: float = 0.50   # ρu: persistence of cost-push shock
    gamma_pi: float = 0.50  # γπ: inflation inertia

class SimpleNK3EqBuiltIn:
    """Tiny 3-equation NK model used for quick IRFs without loading Excel."""
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(self, shock: str = "demand", T: int = 24, size_pp: float = 1.0, t0: int = 0, rho_override: Optional[float] = None):
        """
        Generate impulse responses for output gap (x), inflation (pi), and rate (i)
        to a chosen shock type with optional shock persistence override.
        All variables are in **percentage points** (pp) deviations from baseline.
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

            # Simple static contemporaneous block (linearized intuition)
            # IS via Taylor and Phillips, solved reduced-form for x_t
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
# ORIGINAL MODEL (DSGE.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read Excel, merge sheets, set dates/units, and create basic lagged fields used by the regressions."""
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

    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
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

    def safe_div(num, den):
        return num / den if abs(den) > 1e-8 else np.nan

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

        eps = policy_shock_arr[t]  # decimal (e.g., 0.0025 = 25 bp)
        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0:
                i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0:
                i_raw = min(i_raw, i[t - 1] - abs(eps))

        i[t] = float(i_raw)

    return g, p, i

# =========================
# NEW: NK — Real data loader/fit/sim
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_nk_real(file_like_or_path, pi_star_quarterly: float):
    """
    Load DSGE_Model2.xlsx and prepare columns:
      Sheets:
        - "IS Curve": Date, Output Gap, Nominal Interest Rate, Inflation Rate, Foreign Demand, Non-Energy, Energy, REER
        - "Phillips Curve": Date, Inflation Rate, Output Gap, Foreign Demand, Non-Energy, Energy, REER
        - "Taylor": Date, Nominal Interest Rate, Inflation Gap, Output Gap  (we will recompute Inflation Gap)
    Returns df_all (joined) and df_est (dropna on required cols).
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
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips Curve")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    # Normalize names (strip spaces)
    def clean_cols(df):
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]
        return df

    is_df = clean_cols(is_df)
    pc_df = clean_cols(pc_df)
    tr_df = clean_cols(tr_df)

    # Merge on Date to a single timeline
    df = (
        is_df.merge(pc_df, on="Date", suffixes=("_IS", "_PC"), how="outer")
             .merge(tr_df, on="Date", suffixes=("", "_TR"), how="outer")
             .sort_values("Date")
             .set_index("Date")
    )

    # Identify/rename commonly used columns (left side and drivers)
    # IS block columns:
    #   Output Gap (use IS sheet version if both exist)
    if "Output Gap_IS" in df.columns:
        df["Output_Gap"] = df["Output Gap_IS"]
    elif "Output Gap" in df.columns:
        df["Output_Gap"] = df["Output Gap"]
    else:
        raise KeyError("Missing 'Output Gap' in IS Curve sheet.")

    # Rates
    # Prefer IS sheet versions for alignment; fall back to others if necessary
    col_nominal = None
    for cand in ["Nominal Interest Rate_IS", "Nominal Interest Rate", "Nominal Interest Rate_TR"]:
        if cand in df.columns:
            col_nominal = cand; break
    if col_nominal is None:
        raise KeyError("Missing 'Nominal Interest Rate' in IS/Taylor sheets.")
    df["Nominal_Rate_dec"] = ensure_decimal_rate(df[col_nominal])

    col_infl = None
    for cand in ["Inflation Rate_IS", "Inflation Rate_PC", "Inflation Rate"]:
        if cand in df.columns:
            col_infl = cand; break
    if col_infl is None:
        raise KeyError("Missing 'Inflation Rate' in IS/Phillips sheets.")
    df["Inflation_dec"] = ensure_decimal_rate(df[col_infl])

    # Exogenous drivers (use IS names where available)
    for name_is, alias in [
        ("Foreign Demand_IS", "Foreign_Demand"),
        ("Non-Energy_IS", "Non_Energy"),
        ("Energy_IS", "Energy"),
        ("REER_IS", "REER"),
    ]:
        if name_is in df.columns:
            df[alias] = df[name_is]
        else:
            # fallbacks from Phillips sheet
            fallback = name_is.replace("_IS", "_PC")
            if fallback in df.columns:
                df[alias] = df[fallback]
            else:
                # last resort: a generic name without suffix if present
                base = name_is.replace("_IS", "")
                if base in df.columns:
                    df[alias] = df[base]

    # Lags and gaps
    df["Output_Gap_L1"] = df["Output_Gap"].shift(1)
    df["Inflation_L1_dec"] = df["Inflation_dec"].shift(1)
    df["Nominal_L1_dec"] = df["Nominal_Rate_dec"].shift(1)
    df["Real_Rate_dec"] = df["Nominal_Rate_dec"] - df["Inflation_dec"]
    df["Inflation_Gap"] = df["Inflation_dec"] - float(pi_star_quarterly)

    # Required columns for estimation
    req = [
        "Output_Gap", "Output_Gap_L1",
        "Inflation_dec", "Inflation_L1_dec",
        "Nominal_Rate_dec", "Nominal_L1_dec",
        "Real_Rate_dec",
        "Foreign_Demand", "Non_Energy", "Energy", "REER",
        "Inflation_Gap",
    ]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for NK-Real: {missing}")

    df_est = df.dropna(subset=req).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for NK-Real columns. Check DSGE_Model2.xlsx.")

    return df, df_est

def fit_models_nk_real(df_est: pd.DataFrame):
    """
    Fit OLS for:
      IS: Output_Gap ~ const + Output_Gap_L1 + Real_Rate_dec + Foreign_Demand + Non_Energy + Energy + REER
      Phillips: Inflation_dec ~ const + Inflation_L1_dec + Output_Gap + Foreign_Demand + Non_Energy + Energy + REER
      Taylor: Nominal_Rate_dec ~ const + Nominal_L1_dec + Inflation_Gap + Output_Gap
    """
    # IS
    X_is = df_est[["Output_Gap_L1", "Real_Rate_dec", "Foreign_Demand", "Non_Energy", "Energy", "REER"]]
    X_is = sm.add_constant(X_is, has_constant="add")
    y_is = df_est["Output_Gap"]
    m_is = sm.OLS(y_is, X_is).fit()

    # Phillips
    X_pc = df_est[["Inflation_L1_dec", "Output_Gap", "Foreign_Demand", "Non_Energy", "Energy", "REER"]]
    X_pc = sm.add_constant(X_pc, has_constant="add")
    y_pc = df_est["Inflation_dec"]
    m_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor
    X_tr = df_est[["Nominal_L1_dec", "Inflation_Gap", "Output_Gap"]]
    X_tr = sm.add_constant(X_tr, has_constant="add")
    y_tr = df_est["Nominal_Rate_dec"]
    m_tr = sm.OLS(y_tr, X_tr).fit()

    # Extract partial-adjustment form
    b0 = float(m_tr.params.get("const", 0.0))
    rhoi = float(m_tr.params.get("Nominal_L1_dec", 0.0))
    rhoi = min(max(rhoi, 0.0), 0.99)

    def safe_div(num, den):
        return num / den if abs(den) > 1e-8 else np.nan

    alpha_star = safe_div(b0, (1 - rhoi))
    bpi = float(m_tr.params.get("Inflation_Gap", 0.0))
    bx  = float(m_tr.params.get("Output_Gap", 0.0))
    phi_pi_star = safe_div(bpi, (1 - rhoi)) if "Inflation_Gap" in m_tr.params.index else np.nan
    phi_x_star  = safe_div(bx,  (1 - rhoi)) if "Output_Gap" in m_tr.params.index else np.nan

    return {
        "m_is": m_is, "m_pc": m_pc, "m_tr": m_tr,
        "rho_i_hat": rhoi,
        "alpha_star": alpha_star, "phi_pi_star": phi_pi_star, "phi_x_star": phi_x_star
    }

def simulate_nk_real(
    T: int,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float],
    is_shock_pp: float = 0.0,
    pc_shock_pp: float = 0.0,
    policy_shock_bp: float = 0.0,
    t0: int = 0,
    shock_persist: float = 0.0,
    policy_mode: str = "Add after smoothing (standard)"
):
    """
    Forward-simulate using fitted NK-Real OLS models. Units:
      - Output_Gap is in pct points (we keep as given)
      - Inflation_dec and Nominal_Rate_dec are in decimals
      - Shocks: IS (pp added to Output_Gap), Phillips (pp added to Inflation; convert to decimal),
                Policy (basis points added to nominal rate; convert to decimal)
    """
    m_is = models["m_is"]; m_pc = models["m_pc"]; m_tr = models["m_tr"]
    rho_i_hat = models["rho_i_hat"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_x_star = models["phi_x_star"]

    x = np.zeros(T)   # Output gap, pp
    pi = np.zeros(T)  # Inflation, decimal
    i = np.zeros(T)   # Nominal rate, decimal

    # Initialize with in-sample means for stability
    x[0]  = means["Output_Gap"]
    pi[0] = means["Inflation_dec"]
    i[0]  = means["Nominal_Rate_dec"]

    # Build shock arrays
    is_arr = np.zeros(T); pc_arr = np.zeros(T); pol_arr = np.zeros(T)
    if is_shock_pp != 0.0:
        is_arr[t0] = is_shock_pp / 1.0
        for k in range(t0 + 1, T): is_arr[k] = shock_persist * is_arr[k - 1]
    if pc_shock_pp != 0.0:
        pc_arr[t0] = (pc_shock_pp / 100.0)  # convert pp to decimal for inflation
        for k in range(t0 + 1, T): pc_arr[k] = shock_persist * pc_arr[k - 1]
    if policy_shock_bp != 0.0:
        pol_arr[t0] = policy_shock_bp / 10000.0
        for k in range(t0 + 1, T): pol_arr[k] = shock_persist * pol_arr[k - 1]

    for t in range(1, T):
        rr_lag = (i[t-1] - pi[t-1])

        # --- IS prediction: x_t
        vals_is = {
            "Output_Gap_L1": x[t-1],
            "Real_Rate_dec": rr_lag,
            "Foreign_Demand": means["Foreign_Demand"],
            "Non_Energy": means["Non_Energy"],
            "Energy": means["Energy"],
            "REER": means["REER"],
        }
        Xis = row_from_params(m_is.params.index, vals_is)
        x[t] = float(m_is.predict(Xis).iloc[0]) + is_arr[t]

        # --- Phillips prediction: pi_t (decimal)
        vals_pc = {
            "Inflation_L1_dec": pi[t-1],
            "Output_Gap": x[t-1],
            "Foreign_Demand": means["Foreign_Demand"],
            "Non_Energy": means["Non_Energy"],
            "Energy": means["Energy"],
            "REER": means["REER"],
        }
        Xpc = row_from_params(m_pc.params.index, vals_pc)
        pi[t] = float(m_pc.predict(Xpc).iloc[0]) + pc_arr[t]

        # --- Taylor star level (decimal)
        pi_gap_t = (pi[t] - means["pi_star_quarterly"])
        i_star = alpha_star \
                 + (0.0 if np.isnan(phi_pi_star) else phi_pi_star) * pi_gap_t \
                 + (0.0 if np.isnan(phi_x_star)  else phi_x_star)  * x[t]

        eps = pol_arr[t]
        if policy_mode.startswith("Add after"):
            i_raw = rho_i_hat * i[t - 1] + (1 - rho_i_hat) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_i_hat * i[t - 1] + (1 - rho_i_hat) * (i_star + eps)
        else:
            i_raw = rho_i_hat * i[t - 1] + (1 - rho_i_hat) * i_star + eps
            if eps > 0:
                i_raw = max(i_raw, i[t - 1] + abs(eps))
            elif eps < 0:
                i_raw = min(i_raw, i[t - 1] - abs(eps))

        i[t] = float(i_raw)

    return x, pi, i

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox("Choose model version", ["Original (DSGE.xlsx)", "Simple NK"], index=0)

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original", help="If omitted, the app looks for 'DSGE.xlsx' next to this script.")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05, help="How much the policy rate inherits from its own past. Higher ρ ⇒ more persistence.")

        st.header("Inflation target for Taylor")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False, help="If checked, π* is the average of your sample's quarterly inflation.")
        if use_sample_mean:
            target_annual_pct = None
            st.caption("π* will be set to sample mean (quarterly) after data loads.")
        else:
            target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1, help="Annualized target inflation; we convert this to a quarterly decimal.")
        st.divider()

        st.header("Shock")
        shock_target = st.selectbox(
            "Apply shock to",
            ["None", "IS (Demand)", "Phillips (Supply)", "Taylor (Policy tightening)", "Taylor (Policy easing)"],
            index=0,
            help="Choose which block is directly shocked. Policy shocks move the rate by the bp size below."
        )
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f", help="One-time bump to GDP growth (percentage points).")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f", help="One-time bump to inflation (percentage points).")
        policy_shock_bp_abs = st.number_input("Policy shock size (absolute bp)", value=25, step=5, format="%d", help="Size of the policy rate shock in basis points (25 bp = 0.25%).")
        shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1, 1, help="Quarter index at which the shock hits.")
        shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05, help="How much the shock decays each period. 0 = one-and-done.")

        st.header("Policy shock behavior")
        policy_mode = st.radio(
            "Choose how the policy shock is applied",
            ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
            index=0,
            help=("How policy shocks enter the partial-adjustment rule:\n"
                  "• **Add after smoothing**: i_t = ρ i_{t-1} + (1−ρ) i*_t + ε^pol_t\n"
                  "• **Add to target**: i_t = ρ i_{t-1} + (1−ρ)(i*_t + ε^pol_t)\n"
                  "• **Force local jump**: Ensures a minimum jump by the shock size at t.")
        )

        st.divider()
        st.header("Variable selection (include/exclude)")
        IS_ALL = ["DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"]
        PC_ALL = ["Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"]
        TR_ALL = ["Nominal_Rate_L1", "Inflation_Gap", "DlogGDP"]

        with st.expander("IS Curve regressors", expanded=True):
            is_selected = st.multiselect("Use these variables in the IS regression:", IS_ALL, default=IS_ALL, key="is_vars")
        with st.expander("Phillips Curve regressors", expanded=True):
            pc_selected = st.multiselect("Use these variables in the Phillips regression:", PC_ALL, default=PC_ALL, key="pc_vars")
        with st.expander("Taylor Rule regressors", expanded=True):
            tr_selected = st.multiselect("Use these variables in the Taylor regression:", TR_ALL, default=TR_ALL, key="tr_vars")

    else:
        # ======= Parameter → Curve map (quick card) =======
        st.info("**Which parameters affect which curve?**  \n"
                "• **IS (Demand)**: σ, ρx, ρr  \n"
                "• **Phillips (Supply)**: κ, γπ, ρu  \n"
                "• **Taylor Rule (Policy)**: φπ, φx, ρi")

        # Data source toggle for Simple NK
        st.header("Simple NK — Data source")
        nk_data_source = st.radio("Use data from:", ["Simulated (built-in)", "Real (Excel: DSGE_Model2.xlsx)"], index=0)
        # Common display option for policy rate UNITS
        units_mode = st.radio("Policy rate units", ["Deviation (pp)", "Level (% annual)"], index=0)

        if nk_data_source == "Simulated (built-in)":
            st.header("Simple NK parameters (pp units)")
            # -------- IS (Demand) --------
            st.subheader("IS Curve (Demand)")
            sigma = st.slider("σ — Demand sensitivity denominator", 0.2, 5.0, 1.00, 0.05)
            rho_x = st.slider("ρx — Output persistence", 0.0, 0.98, 0.50, 0.02)
            rho_r = st.slider("ρr — Demand-shock persistence (r^n_t)", 0.0, 0.98, 0.80, 0.02)

            # -------- Phillips (Supply) --------
            st.subheader("Phillips Curve (Supply)")
            kappa = st.slider("κ — Phillips slope", 0.01, 0.50, 0.10, 0.01)
            gamma_pi = st.slider("γπ — Inflation inertia", 0.0, 0.95, 0.50, 0.05)
            rho_u = st.slider("ρu — Cost-push shock persistence (u_t)", 0.0, 0.98, 0.50, 0.02)

            # -------- Taylor (Policy) --------
            st.subheader("Taylor Rule (Policy)")
            phi_pi = st.slider("φπ — Response to inflation", 1.0, 3.0, 1.50, 0.05)
            phi_x = st.slider("φx — Response to output gap", 0.00, 1.00, 0.125, 0.005)
            rho_i = st.slider("ρi — Policy rate smoothing", 0.0, 0.98, 0.80, 0.02)

            # ---- Shock controls ----
            st.divider()
            st.header("Shock")
            shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
            shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
            shock_quarter_nk = st.slider("Shock timing t", 1, T-1, 1, 1)
            shock_persist_nk = st.slider("Shock persistence ρ_shock (for demand/cost)", 0.0, 0.98, 0.80, 0.02)

            # ---- Snap-back option ----
            snapback = st.checkbox("Snap-back (no x/π persistence; keep ρi)", value=True)

        else:
            st.header("NK Real data source")
            xlf_nk = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"], key="upload_nk_real",
                                      help="If omitted, the app looks for 'DSGE_Model2.xlsx' next to this script (github folder).")
            fallback_nk = Path(__file__).parent / "DSGE_Model2.xlsx"

            st.header("Inflation target π* (for Inflation Gap in Taylor)")
            target_annual_pct_nk = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1,
                                             help="Annualized target; converted to quarterly decimal for the NK-Real equations.")

            st.divider()
            st.header("Shock")
            shock_type_nk_real = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
            shock_size_pp_nk_real = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
            shock_quarter_nk_real = st.slider("Shock timing t", 1, T-1, 1, 1)
            shock_persist_nk_real = st.slider("Shock persistence ρ_shock", 0.0, 0.98, 0.50, 0.02)
            policy_mode_nk_real = st.radio("Policy shock behavior",
                                           ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
                                           index=0)

    # Baseline (neutral) level for showing policy rate in % level (for both NK modes)
    neutral_rate_pct = st.number_input(
        "Baseline (neutral) nominal policy rate — % annual",
        value=2.00, step=0.25, format="%.2f",
        help="Used only for plotting NK policy rate in level (%)."
    )

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        # ---- Load
        file_source = xlf if 'xlf' in locals() and xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)

        # ---- π* (quarterly decimal)
        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
        else:
            annual_pct = target_annual_pct if 'target_annual_pct' in locals() and target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

        # ---- Fit
        models_o = fit_models_original(df_est, pi_star_quarterly, is_selected, pc_selected, tr_selected)

        # ---- Anchors & means
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

        # ---- Build shocks & simulate
        is_arr, pc_arr, pol_arr = build_shocks_original(
            T, shock_target, is_shock_size_pp, pc_shock_size_pp, policy_shock_bp_abs, shock_quarter, shock_persist
        )
        g0, p0, i0 = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            policy_mode=policy_mode
        )
        gS, pS, iS = simulate_original(
            T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, pi_star_quarterly,
            is_shock_arr=is_arr, pc_shock_arr=pc_arr, policy_shock_arr=pol_arr, policy_mode=policy_mode
        )

        # ---- Plot
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

        # ---- Equations / diagnostics
        st.subheader("Estimated Equations (Original model)")
        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
        rho_hat = models_o["rho_hat"]

        is_terms = []
        pretty_map_is = {
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Real_Rate_L2_data": r"RR_{t-2}",
            "Dlog FD_Lag1": r"\Delta \log FD_{t-1}",
            "Dlog_REER": r"\Delta \log REER_t",
            "Dlog_Energy": r"\Delta \log Energy_t",
            "Dlog_NonEnergy": r"\Delta \log NonEnergy_t",
        }
        for k, v in m_is.params.items():
            if k == "const": continue
            is_terms.append((float(v), pretty_map_is.get(k, k)))
        st.markdown("**IS Curve (\\(\\Delta \\log GDP_t\\))**")
        st.latex(build_latex_equation(float(m_is.params.get("const", 0.0)), is_terms, r"\Delta \log GDP_t", r"\varepsilon_t"))

        pc_terms = []
        pretty_map_pc = {
            "Dlog_CPI_L1": r"\Delta \log CPI_{t-1}",
            "DlogGDP_L1": r"\Delta \log GDP_{t-1}",
            "Dlog_Reer_L2": r"\Delta \log REER_{t-2}",
            "Dlog_Energy_L1": r"\Delta \log Energy_{t-1}",
            "Dlog_Non_Energy_L1": r"\Delta \log NonEnergy_{t-1}",
        }
        for k, v in m_pc.params.items():
            if k == "const": continue
            pc_terms.append((float(v), pretty_map_pc.get(k, k)))
        st.markdown("**Phillips Curve (\\(\\Delta \log CPI_t\\))**")
        st.latex(build_latex_equation(float(m_pc.params.get("const", 0.0)), pc_terms, r"\Delta \log CPI_t", r"u_t"))

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
            st.write("**IS Curve**"); st.text(m_is.summary().as_text())
            st.write("**Phillips Curve**"); st.text(m_pc.summary().as_text())
            st.write("**Taylor Rule**"); st.text(m_tr.summary().as_text())

    else:
        # =========================
        # SIMPLE NK block
        # =========================
        if nk_data_source == "Simulated (built-in)":
            # Apply snap-back: x and π have no inertia, shock is one-period.
            # Keep policy smoothing ρ_i to get a geometric decay in i_t.
            P = NKParamsSimple(
                sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
                rho_i=rho_i,
                rho_x=(0.0 if snapback else rho_x),
                rho_r=rho_r, rho_u=rho_u,
                gamma_pi=(0.0 if snapback else gamma_pi)
            )
            model = SimpleNK3EqBuiltIn(P)
            label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
            code = label_to_code[shock_type_nk]
            t0 = max(0, min(T-1, (shock_quarter_nk - 1)))

            st.info("**Model key (Simple NK — Simulated):**  "
                    r"$x_t$ = output gap (pp),  "
                    r"$\pi_t$ = inflation (pp),  "
                    r"$i_t$ = policy rate (pp).")

            rho_for_shock = 0.0 if snapback else shock_persist_nk

            # Baseline vs Shock
            h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
            h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

            # Policy series plotting units
            i0_plot, iS_plot = i0.copy(), iS.copy()
            i_ylabel = "pp"
            if units_mode == "Level (% annual)":
                i0_plot = neutral_rate_pct + i0_plot
                iS_plot = neutral_rate_pct + iS_plot
                i_ylabel = "%"

            # Plot
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

            axes[2].plot(h, i0_plot, linewidth=2, label="Baseline")
            axes[2].plot(h, iS_plot, linewidth=2, label="Shock")
            axes[2].axvline(t0, **vline_kwargs)
            axes[2].set_title("Nominal Policy Rate (i_t)")
            axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel(i_ylabel)
            axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

            plt.tight_layout(); st.pyplot(fig)

            with st.expander("Simple NK equations (Simulated)"):
                st.latex(r"x_t = \rho_x x_{t-1} \;-\; \frac{1}{\sigma}\big( i_t - \pi_{t+1} - r^n_t \big)")
                st.latex(r"\pi_t = \gamma_\pi \pi_{t-1} \;+\; \kappa x_t \;+\; u_t")
                st.latex(r"i_t = \rho_i i_{t-1} \;+\; (1-\rho_i)(\phi_\pi \pi_t + \phi_x x_t) \;+\; \varepsilon^i_t")

        else:
            # ===== NK using Real data =====
            file_source_nk = xlf_nk if ('xlf_nk' in locals() and xlf_nk is not None) else (fallback_nk if 'fallback_nk' in locals() else None)
            pi_star_quarterly_nk = (target_annual_pct_nk / 100.0) / 4.0

            df_all_nk, df_est_nk = load_and_prepare_nk_real(file_source_nk, pi_star_quarterly_nk)
            models_nk = fit_models_nk_real(df_est_nk)

            # Means / anchors
            means_nk = {
                "Output_Gap": float(df_est_nk["Output_Gap"].mean()),
                "Inflation_dec": float(df_est_nk["Inflation_dec"].mean()),
                "Nominal_Rate_dec": float(df_est_nk["Nominal_Rate_dec"].mean()),
                "Foreign_Demand": float(df_est_nk["Foreign_Demand"].mean()),
                "Non_Energy": float(df_est_nk["Non_Energy"].mean()),
                "Energy": float(df_est_nk["Energy"].mean()),
                "REER": float(df_est_nk["REER"].mean()),
                "pi_star_quarterly": float(pi_star_quarterly_nk),
            }

            # Map shock choices
            label_to_target = {"Demand (IS)": "is", "Cost-push (Phillips)": "pc", "Policy (Taylor)": "pol"}
            target = label_to_target[shock_type_nk_real]
            t0 = max(0, min(T-1, (shock_quarter_nk_real - 1)))

            is_pp = shock_size_pp_nk_real if target == "is" else 0.0
            pc_pp = shock_size_pp_nk_real if target == "pc" else 0.0
            pol_bp = 100.0 * shock_size_pp_nk_real if target == "pol" else 0.0  # simple mapping: 1pp -> 100bp

            x0, pi0, i0 = simulate_nk_real(
                T, models_nk, means_nk,
                is_shock_pp=0.0, pc_shock_pp=0.0, policy_shock_bp=0.0,
                t0=t0, shock_persist=shock_persist_nk_real, policy_mode=policy_mode_nk_real
            )
            xS, piS, iS = simulate_nk_real(
                T, models_nk, means_nk,
                is_shock_pp=is_pp, pc_shock_pp=pc_pp, policy_shock_bp=pol_bp,
                t0=t0, shock_persist=shock_persist_nk_real, policy_mode=policy_mode_nk_real
            )

            # Prepare plotting
            quarters = np.arange(T)
            i_ylabel = "pp"
            # For NK-Real, we'll show:
            #  - Output gap: as-is (pp)
            #  - Inflation: convert decimal to % (×100)
            #  - Policy rate: either deviation (we'll show level by adding neutral, for readability)
            i0_plot = i0.copy()
            iS_plot = iS.copy()
            if units_mode == "Level (% annual)":
                i0_plot = i0_plot * 100.0
                iS_plot = iS_plot * 100.0
                i_ylabel = "%"
            else:
                # Deviation mode: show as pp deviations from baseline ~approx by subtracting baseline mean
                base = means_nk["Nominal_Rate_dec"] * 100.0  # in %
                i0_plot = (i0 * 100.0) - base
                iS_plot = (iS * 100.0) - base
                i_ylabel = "pp"

            plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
            fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

            axes[0].plot(quarters, x0, linewidth=2, label="Baseline")
            axes[0].plot(quarters, xS, linewidth=2, label="Shock")
            axes[0].axvline(t0, **vline_kwargs)
            axes[0].set_title("Output Gap (pp)")
            axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

            axes[1].plot(quarters, pi0*100.0, linewidth=2, label="Baseline")
            axes[1].plot(quarters, piS*100.0, linewidth=2, label="Shock")
            axes[1].axvline(t0, **vline_kwargs)
            axes[1].set_title("Inflation (% quarterly)"); axes[1].set_ylabel("%")
            axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

            axes[2].plot(quarters, i0_plot if units_mode=="Level (% annual)" else i0_plot, linewidth=2, label="Baseline")
            axes[2].plot(quarters, iS_plot if units_mode=="Level (% annual)" else iS_plot, linewidth=2, label="Shock")
            axes[2].axvline(t0, **vline_kwargs)
            axes[2].set_title("Nominal Policy Rate" + (" (level, % annual)" if units_mode=="Level (% annual)" else " (deviation, pp)"))
            axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel(i_ylabel)
            axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

            plt.tight_layout(); st.pyplot(fig)

            # Display estimated equations for NK-Real
            with st.expander("Estimated Equations (NK — Real)"):
                m_is = models_nk["m_is"]; m_pc = models_nk["m_pc"]; m_tr = models_nk["m_tr"]
                rho_i_hat = models_nk["rho_i_hat"]; alpha_star = models_nk["alpha_star"]
                phi_pi_star = models_nk["phi_pi_star"]; phi_x_star = models_nk["phi_x_star"]

                st.markdown("**IS (Output_Gap)**")
                terms_is = []
                pretty_is = {
                    "Output_Gap_L1": r"x_{t-1}",
                    "Real_Rate_dec": r"(i_{t-1} - \pi_{t-1})",
                    "Foreign_Demand": r"FD_t",
                    "Non_Energy": r"NonEnergy_t",
                    "Energy": r"Energy_t",
                    "REER": r"REER_t",
                }
                for k, v in m_is.params.items():
                    if k == "const": continue
                    terms_is.append((float(v), pretty_is.get(k, k)))
                st.latex(build_latex_equation(float(m_is.params.get("const", 0.0)), terms_is, r"x_t", r"\varepsilon_t"))

                st.markdown("**Phillips (Inflation_dec)**")
                terms_pc = []
                pretty_pc = {
                    "Inflation_L1_dec": r"\pi_{t-1}",
                    "Output_Gap": r"x_{t}",
                    "Foreign_Demand": r"FD_t",
                    "Non_Energy": r"NonEnergy_t",
                    "Energy": r"Energy_t",
                    "REER": r"REER_t",
                }
                for k, v in m_pc.params.items():
                    if k == "const": continue
                    terms_pc.append((float(v), pretty_pc.get(k, k)))
                st.latex(build_latex_equation(float(m_pc.params.get("const", 0.0)), terms_pc, r"\pi_t", r"u_t"))

                st.markdown("**Taylor (Nominal_Rate_dec, partial adjustment)**")
                st.latex(r"i_t \;=\; \rho_i\, i_{t-1} \;+\; (1-\rho_i)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
                st.latex(r"i_t^\* \;=\; \alpha^\* \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{x}^\*\,x_t")
                st.markdown(f"Estimated:  ρ_i = **{rho_i_hat:.3f}**,  α* = **{alpha_star:.3f}**,  "
                            f"φ_π* = **{phi_pi_star:.3f}**,  φ_x* = **{phi_x_star:.3f}**,  π* (quarterly dec) = **{means_nk['pi_star_quarterly']:.4f}**")

                with st.expander("OLS summaries"):
                    st.write("**IS**"); st.text(m_is.summary().as_text())
                    st.write("**Phillips**"); st.text(m_pc.summary().as_text())
                    st.write("**Taylor**"); st.text(m_tr.summary().as_text())

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()





