# dsge_dashboard.py
# -----------------------------------------------------------
# A single Streamlit app with two sections:
#   TAB 1) DSGE (IRFs):
#       - Original (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate, decimal)
#         * Taylor uses inflation gap (π_t − π*)
#         * Shocks: IS, Phillips, and Taylor (tightening/easing)
#         * Policy shock behavior selector:
#             · Add after smoothing (default)
#             · Add to target (inside 1−ρ)
#             · Force local jump (override)
#         * Equations and parameter→curve mapping shown
#       - Simple NK (built-in): 3-eq NK with clearly grouped sliders by curve
#
#   TAB 2) Economic Indicators (Plotly):
#       - Ported from Dash logic; HP filter, trend/cycle/raw render
#       - Year range slider + Real GDP overlay
#
# Notes:
# - No NK(Excel) model; only Original (Excel) + Simple NK (built-in).
# - For Original model: GDP/CPI plotted in %, policy rate in DECIMAL.
# - For Simple NK: x, π, i all in percentage points (pp).
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Macro Dashboard — DSGE & Indicators", layout="wide")
st.title("Macro Dashboard — DSGE (IRFs) & Economic Indicators")

st.markdown(
    "Use the tabs below to switch between the **DSGE (IRFs)** tools and the **Economic Indicators** graphs."
)

# =========================
# Shared helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:  # e.g., 3.2 means 3.2%
        return s / 100.0
    return s

def fmt_coef(x: float, nd: int = 3) -> str:
    s = f"{x:.{nd}f}"
    return f"+{s}" if x >= 0 else s

# =========================
# Simple NK (built-in)
# =========================
@dataclass
class NKParamsSimple:
    sigma: float = 1.00   # σ: demand sensitivity to real rate (higher = less sensitive)
    kappa: float = 0.10   # κ: slope of NK Phillips curve (how demand moves inflation)
    phi_pi: float = 1.50  # φπ: policy response to inflation
    phi_x: float = 0.125  # φx: policy response to output gap
    rho_i: float = 0.80   # ρi: interest rate smoothing
    rho_x: float = 0.50   # ρx: persistence of output gap
    rho_r: float = 0.80   # ρr: persistence of demand (natural-rate) shock
    rho_u: float = 0.50   # ρu: persistence of cost-push shock
    gamma_pi: float = 0.50  # γπ: inflation inertia

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

            # Solve contemporaneously for x_t given policy and Phillips
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

def fit_models_original(df_est: pd.DataFrame, pi_star_quarterly: float):
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

    # Taylor with inflation gap
    infl_gap = df_est["Dlog_CPI"] - pi_star_quarterly
    X_tr = sm.add_constant(pd.DataFrame({
        "Nominal_Rate_L1": df_est["Nominal_Rate_L1"],
        "Inflation_Gap": infl_gap,
        "DlogGDP": df_est["DlogGDP"],
    }))
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    b0 = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal_Rate_L1"]), 0.99)
    bpi = float(model_tr.params["Inflation_Gap"])
    bg = float(model_tr.params["DlogGDP"])

    alpha_star = b0 / (1 - rhoh)
    phi_pi_star = bpi / (1 - rhoh)
    phi_g_star = bg / (1 - rhoh)

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
    """
    Policy shock modes:
      - Add after smoothing: i_t = ρ i_{t-1} + (1−ρ) i*_t + ε_t^{pol}
      - Add to target:      i_t = ρ i_{t-1} + (1−ρ)( i*_t + ε_t^{pol} )
      - Force local jump:   compute with 'Add after smoothing', then override to ensure
                            tightening raises i_t by at least |ε| vs i_{t-1} (and vice versa for easing)
    """
    g = np.zeros(T); p = np.zeros(T); i = np.zeros(T)

    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None: policy_shock_arr = np.zeros(T)

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

        # Taylor target with inflation gap
        pi_gap_t = p[t] - pi_star_quarterly
        i_star = alpha_star + phi_pi_star * pi_gap_t + phi_g_star * g[t]

        # --- Apply policy shock according to chosen mode ---
        eps = policy_shock_arr[t]  # decimal (e.g., 0.0025 = 25 bp)

        if policy_mode.startswith("Add after"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
        elif policy_mode.startswith("Add to target"):
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * (i_star + eps)
        else:  # Force local jump (override)
            i_raw = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + eps
            if eps > 0:  # tightening
                min_jump = abs(eps)
                i_raw = max(i_raw, i[t - 1] + min_jump)
            elif eps < 0:  # easing
                min_jump = abs(eps)
                i_raw = min(i_raw, i[t - 1] - min_jump)

        i[t] = float(i_raw)

    return g, p, i

# =========================
# Economic Indicators helpers
# =========================
def apply_hp_filter(df: pd.DataFrame, column: str, prefix: Optional[str] = None,
                    log_transform: bool = False, exp_transform: bool = False) -> pd.DataFrame:
    if column not in df.columns:
        return df
    prefix = prefix or column
    series = df[column].replace(0, np.nan).dropna()
    if len(series) < 10:
        return df
    clean_series = np.log(series) if log_transform else series
    cycle, trend = hpfilter(clean_series, lamb=1600)
    if exp_transform:
        trend = np.exp(trend)
        cycle = clean_series - np.log(trend)  # not used, but keep consistent
    df[f"{prefix}_Trend"] = trend.reindex(df.index)
    df[f"{prefix}_Cycle"] = cycle.reindex(df.index)
    return df

def plot_indicator(df: pd.DataFrame, date_col: str, var: str, kind: str, title: str,
                   overlay_series: Optional[pd.Series] = None, overlay_name: str = "Overlay"):
    fig = go.Figure()
    if kind == "trend":
        fig.add_trace(go.Scatter(x=df[date_col], y=df.get(f"{var}_Trend"), mode="lines",
                                 name=f"{var} - Trend", line=dict(dash="dot")))
    elif kind == "cycle":
        fig.add_trace(go.Scatter(x=df[date_col], y=df.get(f"{var}_Cycle"), mode="lines",
                                 name=f"{var} - Cycle", line=dict(dash="dot")))
    elif kind == "raw_trend":
        fig.add_trace(go.Scatter(x=df[date_col], y=df.get(var), mode="lines",
                                 name=f"{var} - Raw"))
        fig.add_trace(go.Scatter(x=df[date_col], y=df.get(f"{var}_Trend"), mode="lines",
                                 name=f"{var} - Trend", line=dict(dash="dot")))
    else:  # raw
        fig.add_trace(go.Scatter(x=df[date_col], y=df.get(var), mode="lines",
                                 name=f"{var} - Raw"))
    if overlay_series is not None:
        fig.add_trace(go.Scatter(x=df[date_col], y=overlay_series, mode="lines",
                                 name=overlay_name))
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title="Date", yaxis_title=var, legend_title="Series",
        template="plotly_white", hovermode="x unified", margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

# =========================
# TABS
# =========================
tab1, tab2 = st.tabs(["🔵 DSGE (IRFs)", "🟢 Economic Indicators"])

# ==========================================================
# TAB 1 — DSGE (IRFs)
# ==========================================================
with tab1:
    st.subheader("Section A — DSGE IRF Dashboard")

    with st.sidebar:
        st.markdown("### DSGE Controls")
        model_choice = st.selectbox("Model", ["Original (DSGE.xlsx)", "Simple NK (built-in)"], index=0)
        T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    if model_choice == "Original (DSGE.xlsx)":
        colL, colR = st.columns([1,2], gap="large")
        with colL:
            xlf = st.file_uploader("Upload DSGE.xlsx (or place beside script)", type=["xlsx"])
            rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

            st.markdown("**Inflation target (π\*) for Taylor**")
            use_sample_mean = st.checkbox("Use sample mean of DlogCPI as π*", value=False)
            if use_sample_mean:
                target_annual_pct = None
                st.caption("π* will be set to sample mean (quarterly) after data loads.")
            else:
                target_annual_pct = st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)
            st.divider()

            st.markdown("**Shocks**")
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

            st.markdown("**Policy shock behavior**")
            policy_mode = st.radio(
                "How to apply the policy shock",
                ["Add after smoothing (standard)", "Add to target (inside 1−ρ)", "Force local jump (override)"],
                index=0
            )

        with colR:
            try:
                file_source = xlf if xlf is not None else None
                df_all, df_est = load_and_prepare_original(file_source)

                # π* (quarterly)
                if use_sample_mean:
                    pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
                    st.info(f"π* = sample mean of DlogCPI ⇒ {pi_star_quarterly:.4f} (quarterly decimal)")
                else:
                    annual_pct = target_annual_pct if target_annual_pct is not None else 2.0
                    pi_star_quarterly = (annual_pct / 100.0) / 4.0
                    st.info(f"π* = {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

                models_o = fit_models_original(df_est, pi_star_quarterly)

                # Anchors & means
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

                # Shocks & simulate
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
                axes[2].set_title("Nominal Policy Rate (decimal)"); axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("decimal")
                axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

                plt.tight_layout(); st.pyplot(fig)

                # Equations
                st.markdown("#### Estimated Equations (Original model)")
                m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
                alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
                rho_hat = models_o["rho_hat"]

                c_is = float(m_is.params["const"])
                a1 = float(m_is.params["DlogGDP_L1"]); a2 = float(m_is.params["Real_Rate_L2_data"])
                a3 = float(m_is.params["Dlog FD_Lag1"]); a4 = float(m_is.params["Dlog_REER"])
                a5 = float(m_is.params["Dlog_Energy"]); a6 = float(m_is.params["Dlog_NonEnergy"])
                st.markdown("**IS Curve (\\(\\Delta \\log GDP_t\\))**")
                st.latex(
                    r"""
                    \begin{aligned}
                    \Delta \log GDP_t &= {c} \; {a1}\,\Delta \log GDP_{t-1} \; {a2}\,RR_{t-2} \; {a3}\,\Delta \log FD_{t-1} \\
                                      &\quad {a4}\,\Delta \log REER_t \; {a5}\,\Delta \log Energy_t \; {a6}\,\Delta \log NonEnergy_t \; + \varepsilon_t
                    \end{aligned}
                    """.replace("{c}", f"{c_is:.3f}")
                     .replace("{a1}", fmt_coef(a1)).replace("{a2}", fmt_coef(a2))
                     .replace("{a3}", fmt_coef(a3)).replace("{a4}", fmt_coef(a4))
                     .replace("{a5}", fmt_coef(a5)).replace("{a6}", fmt_coef(a6))
                )

                c_pc = float(m_pc.params["const"])
                b1 = float(m_pc.params["Dlog_CPI_L1"]); b2 = float(m_pc.params["DlogGDP_L1"])
                b3 = float(m_pc.params["Dlog_Reer_L2"]); b4 = float(m_pc.params["Dlog_Energy_L1"]); b5 = float(m_pc.params["Dlog_Non_Energy_L1"])
                st.markdown("**Phillips Curve (\\(\\Delta \\log CPI_t\\))**")
                st.latex(
                    r"""
                    \begin{aligned}
                    \Delta \log CPI_t &= {c} \; {b1}\,\Delta \log CPI_{t-1} \; {b2}\,\Delta \log GDP_{t-1} \; {b3}\,\Delta \log REER_{t-2} \\
                                       &\quad {b4}\,\Delta \log Energy_{t-1} \; {b5}\,\Delta \log NonEnergy_{t-1} \; + u_t
                    \end{aligned}
                    """.replace("{c}", f"{c_pc:.3f}")
                     .replace("{b1}", fmt_coef(b1)).replace("{b2}", fmt_coef(b2))
                     .replace("{b3}", fmt_coef(b3)).replace("{b4}", fmt_coef(b4))
                     .replace("{b5}", fmt_coef(b5))
                )

                st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
                st.latex(r"i_t^\* \;=\; \alpha^\* \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,g_t")
                st.latex(
                    r"""
                    \rho = {rho}\,,\quad \alpha^\* = {a}\,,\quad \phi_{\pi}^\* = {fp}\,,\quad \phi_{g}^\* = {fg}\,,\quad \pi^\* = {pistar}
                    """.replace("{rho}", f"{rho_hat:.3f}")
                     .replace("{a}", f"{alpha_star:.3f}")
                     .replace("{fp}", f"{phi_pi_star:.3f}")
                     .replace("{fg}", f"{phi_g_star:.3f}")
                     .replace("{pistar}", f"{pi_star_quarterly:.4f}")
                )

            except Exception as e:
                st.error(f"Problem loading/running the Original model: {e}")

    else:
        # Simple NK (built-in)
        st.info("**Which parameters affect which curve?**  \n"
                "• **IS (Demand)**: σ, ρx, ρr  \n"
                "• **Phillips (Supply)**: κ, γπ, ρu  \n"
                "• **Taylor Rule (Policy)**: φπ, φx, ρi")

        colA, colB = st.columns([1,2], gap="large")
        with colA:
            st.markdown("#### IS Curve (Demand)")
            sigma = st.slider("σ — Demand sensitivity denominator", 0.2, 5.0, 1.00, 0.05)
            rho_x = st.slider("ρx — Output persistence", 0.0, 0.98, 0.50, 0.02)
            rho_r = st.slider("ρr — Demand-shock persistence (r^n_t)", 0.0, 0.98, 0.80, 0.02)
            st.caption("Lower σ or higher ρx/ρr → output gap moves more/longer after shocks.")

            st.markdown("#### Phillips Curve (Supply)")
            kappa = st.slider("κ — Phillips slope", 0.01, 0.50, 0.10, 0.01)
            gamma_pi = st.slider("γπ — Inflation inertia", 0.0, 0.95, 0.50, 0.05)
            rho_u = st.slider("ρu — Cost-push shock persistence (u_t)", 0.0, 0.98, 0.50, 0.02)
            st.caption("Higher κ/γπ or higher ρu → inflation moves more and/or fades slower.")

            st.markdown("#### Taylor Rule (Policy)")
            phi_pi = st.slider("φπ — Response to inflation", 1.0, 3.0, 1.50, 0.05)
            phi_x = st.slider("φx — Response to output gap", 0.00, 1.00, 0.125, 0.005)
            rho_i = st.slider("ρi — Policy rate smoothing", 0.0, 0.98, 0.80, 0.02)
            st.caption("Larger φπ/φx → stronger policy reaction; higher ρi → smoother, slower moves.")

            st.divider()
            st.markdown("#### Shock")
            shock_type_nk = st.selectbox("Shock type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
            shock_size_pp_nk = st.number_input("Shock size (pp)", value=1.00, step=0.25, format="%.2f")
            shock_quarter_nk = st.slider("Shock timing (t)", 1, T-1, 1, 1)
            shock_persist_nk = st.slider("Shock persistence ρ_shock (for demand/cost)", 0.0, 0.98, 0.80, 0.02)

        with colB:
            P = NKParamsSimple(sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
                               rho_i=rho_i, rho_x=rho_x, rho_r=rho_r, rho_u=rho_u, gamma_pi=gamma_pi)
            model = SimpleNK3EqBuiltIn(P)
            label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
            code = label_to_code[shock_type_nk]
            t0 = max(0, min(T-1, shock_quarter_nk - 1))

            st.caption(r"Model key: $x_t$=output gap (pp), $\pi_t$=inflation (pp), $i_t$=policy rate (pp), "
                       r"$r_t^n$=demand shock, $u_t$=cost-push shock.")

            h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, shock_persist_nk)
            h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, shock_persist_nk)

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

# ==========================================================
# TAB 2 — Economic Indicators (Plotly)
# ==========================================================
with tab2:
    st.subheader("Section B — Economic Indicator Graphs")

    # --- File controls ---
    with st.expander("Data source", expanded=True):
        st.caption("Select the Excel file and (optionally) adjust sheet names to match your workbook.")
        file_up = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: sheet_main = st.text_input("Potential Output sheet", "Potential Output")
        with c2: sheet_hours = st.text_input("Hours sheet", "Hours")
        with c3: sheet_is_curve = st.text_input("IS Curve sheet", "IS Curve")
        with c4: sheet_phillips = st.text_input("Phillips sheet", "Phillips Curve")
        with c5: sheet_taylor = st.text_input("Taylor sheet", "Taylor Rule")

    if file_up is None:
        st.warning("Upload an Excel file to render the indicator graphs.")
        st.stop()

    # --- Load workbook & sheets ---
    try:
        xls = pd.ExcelFile(file_up)
        available = set(xls.sheet_names)
        need = {sheet_main, sheet_hours, sheet_is_curve, sheet_phillips, sheet_taylor}
        missing = [s for s in need if s not in available]
        if missing:
            st.error(f"Missing sheets: {missing}. Available: {xls.sheet_names}")
            st.stop()
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

    # --- Load Potential Output Data (+ Hours) ---
    try:
        data = pd.read_excel(xls, sheet_name=sheet_main, na_values=["NA"])
        cols = ['Date', 'Population', 'Labour Force Participation', 'Labour Productivity', 'NAIRU',
                'Output_Gap_multivariate', 'Output_Gap_Integrated', 'Output_Gap_Internal', 'Real GDP Expenditure']
        for c in cols[1:]:
            if c in data.columns:
                data[c] = pd.to_numeric(data[c], errors='coerce')
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        data['Year'] = data['Date'].dt.year

        hours_df = pd.read_excel(xls, sheet_name=sheet_hours, na_values=["NA"])
        hours_df['Date'] = pd.to_datetime(hours_df['Date'], errors='coerce')
        hours_df = hours_df.dropna(subset=['Date'])
        if 'Average Hours Worked' in hours_df.columns:
            hours_df['Average Hours Worked'] = pd.to_numeric(hours_df['Average Hours Worked'], errors='coerce')

        # Merge hours backward
        data = pd.merge_asof(data.sort_values('Date'), hours_df.sort_values('Date'), on='Date', direction='backward')

        # Construct productivity properly
        data['LFP_decimal'] = data['Labour Force Participation'] / 100
        data['NAIRU_decimal'] = data['NAIRU'] / 100
        data['Total Hours Worked'] = (
            data['Population'] *
            data['LFP_decimal'] *
            (1 - data['NAIRU_decimal']) *
            data['Average Hours Worked']
        )
        data['Labour Productivity'] = data['Real GDP Expenditure'] / data['Total Hours Worked']

        # HP filters on inputs (only if column exists)
        for col in ['Labour Force Participation', 'Labour Productivity', 'Average Hours Worked', 'Unemployment']:
            if col in data.columns:
                apply_hp_filter(data, col)

        # Potential output (rescale to match base year level)
        base_year = st.number_input("Base year for scaling potential output", min_value=1970, max_value=2100, value=2017, step=1)
        if base_year in data['Year'].values:
            base_idx = data.index[data['Year'] == base_year][0]
            raw_potential_base = (
                data.loc[base_idx, 'Population'] *
                (data.loc[base_idx, 'Labour Force Participation_Trend'] / 100 if 'Labour Force Participation_Trend' in data else data.loc[base_idx, 'LFP_decimal']) *
                (1 - data.loc[base_idx, 'NAIRU_decimal']) *
                (data.loc[base_idx, 'Average Hours Worked_Trend'] if 'Average Hours Worked_Trend' in data else data.loc[base_idx, 'Average Hours Worked']) *
                (data.loc[base_idx, 'Labour Productivity_Trend'] if 'Labour Productivity_Trend' in data else data.loc[base_idx, 'Labour Productivity'])
            )
            real_gdp_base = data.loc[base_idx, 'Real GDP Expenditure']
            scaling_factor = real_gdp_base / raw_potential_base if raw_potential_base not in (0, np.nan) else 1.0
        else:
            st.warning("Base year not in the data; using scaling factor = 1.")
            scaling_factor = 1.0

        data['Potential Output'] = (
            data['Population'] *
            (data['Labour Force Participation_Trend'] / 100 if 'Labour Force Participation_Trend' in data else data['LFP_decimal']) *
            (1 - data['NAIRU_decimal']) *
            (data['Average Hours Worked_Trend'] if 'Average Hours Worked_Trend' in data else data['Average Hours Worked']) *
            (data['Labour Productivity_Trend'] if 'Labour Productivity_Trend' in data else data['Labour Productivity']) *
            scaling_factor
        )

        apply_hp_filter(data, 'Potential Output', log_transform=True, exp_transform=True)
        data['Potential Output Growth (%)'] = data['Potential Output_Trend'].pct_change() * 100
        data['Output Gap (%)'] = ((data['Real GDP Expenditure'] - data['Potential Output']) / data['Potential Output']) * 100

    except Exception as e:
        st.error(f"Problem constructing potential output: {e}")
        st.stop()

    # --- Load IS / Phillips / Taylor sheets (+ HP) ---
    def load_sheet(name: str) -> pd.DataFrame:
        df = pd.read_excel(xls, sheet_name=name, na_values=["NA"])
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').reset_index(drop=True)
        df['Year'] = df['Date'].dt.year
        for col in df.columns:
            if col not in ['Date', 'Year']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    try:
        is_curve_df = load_sheet(sheet_is_curve)
        phillips_df = load_sheet(sheet_phillips)
        taylor_df = load_sheet(sheet_taylor)
    except Exception as e:
        st.error(f"Problem loading secondary sheets: {e}")
        st.stop()

    # HP filters across selected columns
    is_curve_cols = [c for c in is_curve_df.columns if c not in ['Date', 'Year']]
    phillips_cols = [c for c in phillips_df.columns if c not in ['Date', 'Year']]
    taylor_cols = [c for c in taylor_df.columns if c not in ['Date', 'Year']]
    for col in is_curve_cols: apply_hp_filter(is_curve_df, col)
    for col in phillips_cols: apply_hp_filter(phillips_df, col)
    for col in taylor_cols: apply_hp_filter(taylor_df, col)

    # Year slider bounds
    year_min = int(min(data['Year'].min(), is_curve_df['Year'].min(), phillips_df['Year'].min(), taylor_df['Year'].min()))
    year_max = int(max(data['Year'].max(), is_curve_df['Year'].max(), phillips_df['Year'].max(), taylor_df['Year'].max()))

    # ---- Controls (left) & Plot (right)
    cL, cR = st.columns([1,2], gap="large")
    with cL:
        st.markdown("#### Select series to plot")
        # General variable list (exclude derived trend/cycle to avoid duplication)
        general_candidates = [c for c in data.columns if c not in ['Date', 'Year'] and not c.endswith('_Trend') and not c.endswith('_Cycle')]
        general = st.selectbox("General Variable (Potential/Gap/etc.)", options=sorted(general_candidates), index=general_candidates.index('Potential Output') if 'Potential Output' in general_candidates else 0)
        show_gdp = st.checkbox("Overlay Real GDP when plotting Potential Output", value=True)

        st.markdown("**Or pick a variable from a specific sheet** (overrides General):")
        pick_is = st.selectbox("IS Curve variable", options=["(none)"] + is_curve_cols, index=0)
        pick_pc = st.selectbox("Phillips variable", options=["(none)"] + phillips_cols, index=0)
        pick_tr = st.selectbox("Taylor variable", options=["(none)"] + taylor_cols, index=0)

        kind = st.radio("Data type", ["raw", "trend", "cycle", "raw_trend"], index=0, horizontal=True)
        year_range = st.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)

    with cR:
        # Resolve selection priority: Taylor > Phillips > IS > General
        if pick_tr != "(none)":
            var, df = pick_tr, taylor_df
        elif pick_pc != "(none)":
            var, df = pick_pc, phillips_df
        elif pick_is != "(none)":
            var, df = pick_is, is_curve_df
        else:
            var, df = general, data

        mask = (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])
        dfv = df.loc[mask].copy()

        overlay = None
        overlay_name = ""
        if show_gdp and var == "Potential Output" and "Real GDP Expenditure" in dfv.columns:
            overlay = dfv["Real GDP Expenditure"]
            overlay_name = "Real GDP"

        fig = plot_indicator(dfv, "Date", var, kind, f"{var} ({year_range[0]}–{year_range[1]})",
                             overlay_series=overlay, overlay_name=overlay_name if overlay is not None else None)
        st.plotly_chart(fig, use_container_width=True)


