# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Taylor rule uses an inflation gap: (π_t - π*)
#      - Supports shocks to IS, Phillips, and Taylor (policy tightening or easing)
#      - LaTeX equations (with fitted coefficients) are rendered below each chart
#   2) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
#
# Notes:
# - GDP/Inflation shown in PERCENT (%); Nominal policy rate shown in DECIMAL (e.g., 0.03 = 3%).
# - In Original model, policy shock is entered in basis points and added directly to i_t (with AR(1) persistence).
# -----------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
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
    "- **Original**: GDP & CPI are plotted in **%** (Dlog × 100); **Nominal rate** in **decimal**.  \n"
    "- **Taylor (Original)** uses **inflation gap**: \\(\\pi_t - \\pi^{\\*}\\).  \n"
    "- **Simple NK (built-in)**: Output gap, inflation, and nominal rate are in **percentage points (pp)**."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    """Ensure a rate is in DECIMAL units. If the series looks like % levels (median abs > 1), divide by 100."""
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:  # typical % level like 3.34
        return s / 100.0
    return s

def fmt_coef(x: float, nd: int = 3) -> str:
    """Format a coefficient with sign, for LaTeX concatenation."""
    s = f"{x:.{nd}f}"
    return f"+{s}" if x >= 0 else s

# =========================
# Simple NK (built-in) model
# =========================
@dataclass
class NKParamsSimple:
    # All in percentage points (pp)
    sigma: float = 1.00   # spending sensitivity (IS)
    kappa: float = 0.10   # NK Phillips slope
    phi_pi: float = 1.50  # Taylor weight on inflation
    phi_x: float = 0.125  # Taylor weight on output gap
    rho_i: float = 0.80   # interest rate smoothing
    rho_x: float = 0.50   # output gap persistence
    rho_r: float = 0.80   # demand (natural-rate) shock persistence
    rho_u: float = 0.50   # cost-push shock persistence
    gamma_pi: float = 0.50  # inflation inertia

class SimpleNK3EqBuiltIn:
    def __init__(self, params: Optional[NKParamsSimple] = None):
        self.p = params or NKParamsSimple()

    def irf(self, shock: str = "demand", T: int = 24, size_pp: float = 1.0, t0: int = 0, rho_override: Optional[float] = None):
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
            rho_sh = None
        else:
            raise ValueError("shock must be 'demand', 'cost', or 'policy'.")

        for t in range(T):
            if t > t0:
                if shock == "demand":
                    r_nat[t] += rho_sh * r_nat[t-1]
                elif shock == "cost":
                    u[t] += rho_sh * u[t-1]

            x_lag = x[t-1] if t > 0 else 0.0
            pi_lag = pi[t-1] if t > 0 else 0.0
            i_lag = i[t-1] if t > 0 else 0.0

            # compact contemporaneous solution
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
# Sidebar: choose model + controls
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox(
        "Choose model version",
        ["Original (DSGE.xlsx)", "Simple NK (built-in)"],
        index=0
    )

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)

    if model_choice == "Original (DSGE.xlsx)":
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
        fallback = Path(__file__).parent / "DSGE.xlsx"

        rho_sim = st.slider("Policy smoothing ρ (Taylor, 0 = fast, 0.95 = slow)", 0.0, 0.95, 0.80, 0.05)

        st.header("Inflation target for Taylor")
        use_sample_mean = st.checkbox("Use sample mean of DlogCPI as target π*", value=False)
        if use_sample_mean:
            st.caption("Target will be set to the sample mean of DlogCPI (quarterly decimal) after data loads.")
            target_annual_pct = None
        else:
            target_annual_pct = st.slider("π* (annual %, converted to quarterly)", 0.0, 5.0, 2.0, 0.1)
        st.divider()

        st.header("Shock")
        shock_target = st.selectbox(
            "Apply shock to",
            [
                "None",
                "IS (Demand)",
                "Phillips (Supply)",
                "Taylor (Policy tightening)",
                "Taylor (Policy easing)"
            ],
            index=0
        )
        is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
        pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
        # Policy shock magnitude in basis points (absolute, sign chosen by option)
        policy_shock_bp_abs = st.number_input(
            "Policy shock size (absolute bp)",
            value=25, step=5, format="%d",
            help="Tightening applies +bp; easing applies -bp. Added directly to i_t (decimal) with AR(1) persistence."
        )
        shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
        shock_persist = st.slider("Shock persistence ρ_shock (applies to all shock types)", 0.0, 0.95, 0.0, 0.05)

    else:
        st.header("Simple NK parameters (pp units)")
        sigma = st.slider("σ (IS sensitivity; larger 1/σ ⇒ stronger rate effect)", 0.2, 5.0, 1.00, 0.05)
        kappa = st.slider("κ (Phillips slope)", 0.01, 0.50, 0.10, 0.01)
        phi_pi = st.slider("φπ (policy weight on inflation)", 1.0, 3.0, 1.50, 0.05)
        phi_x = st.slider("φx (policy weight on output gap)", 0.00, 1.00, 0.125, 0.005)
        rho_i = st.slider("ρi (policy smoothing)", 0.0, 0.98, 0.80, 0.02)
        rho_x = st.slider("ρx (output persistence)", 0.0, 0.98, 0.50, 0.02)
        rho_r = st.slider("ρr (demand shock persistence)", 0.0, 0.98, 0.80, 0.02)
        rho_u = st.slider("ρu (cost shock persistence)", 0.0, 0.98, 0.50, 0.02)
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
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        excel_src = file_like_or_path  # uploaded file-like

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

    # Build lags/transforms (Dlog series already DECIMAL)
    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP", "DlogGDP_L1",
        "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1",
        "Real_Rate_L2_data",
        # IS externals
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        # Phillips externals (lagged)
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")
    return df, df_est


def fit_models_original(df_est: pd.DataFrame, pi_star_quarterly: float) -> Dict[str, sm.regression.linear_model.RegressionResultsWrapper]:
    """Fit IS, Phillips, and Taylor with inflation gap (π_t - π*)."""
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

    # Taylor (partial adjustment) — use inflation gap
    infl_gap = df_est["Dlog_CPI"] - pi_star_quarterly
    X_tr = sm.add_constant(pd.DataFrame({
        "Nominal_Rate_L1": df_est["Nominal_Rate_L1"],
        "Inflation_Gap": infl_gap,
        "DlogGDP": df_est["DlogGDP"],
    }))
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Convert to partial-adjustment parameters used in simulation
    b0 = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal_Rate_L1"]), 0.99)
    bpi = float(model_tr.params["Inflation_Gap"])
    bg = float(model_tr.params["DlogGDP"])

    alpha_star = b0 / (1 - rhoh)
    phi_pi_star = bpi / (1 - rhoh)
    phi_g_star = bg / (1 - rhoh)

    return {
        "model_is": model_is,
        "model_pc": model_pc,
        "model_tr": model_tr,
        "alpha_star": alpha_star,
        "phi_pi_star": phi_pi_star,
        "phi_g_star": phi_g_star,
        "rho_hat": rhoh,
        "pi_star_quarterly": float(pi_star_quarterly),
    }

def build_shocks_original(T, target, is_size_pp, pc_size_pp, policy_bp_abs, t0, rho):
    """
    Build AR(1) shock arrays for IS (DlogGDP, decimal), Phillips (DlogCPI, decimal),
    and Taylor (policy rate, decimal). Policy shock is specified in absolute basis points.
    - Tightening: +bp
    - Easing:     -bp
    """
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    pol_arr = np.zeros(T)

    if target == "IS (Demand)":
        is_arr[t0] = is_size_pp / 100.0  # pp -> decimal
        for k in range(t0 + 1, T):
            is_arr[k] = rho * is_arr[k - 1]

    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size_pp / 100.0  # pp -> decimal
        for k in range(t0 + 1, T):
            pc_arr[k] = rho * pc_arr[k - 1]

    elif target == "Taylor (Policy tightening)":
        pol_arr[t0] = (policy_bp_abs / 10000.0)  # +bp -> decimal
        for k in range(t0 + 1, T):
            pol_arr[k] = rho * pol_arr[k - 1]

    elif target == "Taylor (Policy easing)":
        pol_arr[t0] = -(policy_bp_abs / 10000.0)  # -bp -> decimal
        for k in range(t0 + 1, T):
            pol_arr[k] = rho * pol_arr[k - 1]

    return is_arr, pc_arr, pol_arr

def simulate_original(
    T: int,
    rho_sim: float,
    df_est: pd.DataFrame,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    means: Dict[str, float],
    i_mean_dec: float,
    real_rate_mean_dec: float,
    pi_star_quarterly: float,
    is_shock_arr=None,
    pc_shock_arr=None,
    policy_shock_arr=None
):
    """Simulate paths. Taylor uses inflation gap π_t - π*. Policy shock enters i_t additively (decimal)."""
    g = np.zeros(T)  # DlogGDP (decimal)
    p = np.zeros(T)  # DlogCPI (decimal)
    i = np.zeros(T)  # Nominal rate (decimal)

    # Initialize at sample means (no explicit steady state)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]; model_pc = models["model_pc"]
    alpha_star = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_g_star = models["phi_g_star"]

    if is_shock_arr is None:
        is_shock_arr = np.zeros(T)
    if pc_shock_arr is None:
        pc_shock_arr = np.zeros(T)
    if policy_shock_arr is None:
        policy_shock_arr = np.zeros(T)

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

        pi_gap_t = p[t] - pi_star_quarterly  # inflation gap
        i_star = alpha_star + phi_pi_star * pi_gap_t + phi_g_star * g[t]
        # Policy shock enters additively (decimal)
        i[t] = rho_sim * i[t - 1] + (1 - rho_sim) * i_star + policy_shock_arr[t]

    return g, p, i

# =========================
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
        # Load
        file_source = xlf if xlf is not None else (fallback if 'fallback' in locals() else None)
        df_all, df_est = load_and_prepare_original(file_source)

        # Determine π* (quarterly decimal)
        if 'use_sample_mean' in locals() and use_sample_mean:
            pi_star_quarterly = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* set to sample mean of DlogCPI: {pi_star_quarterly:.4f} (quarterly decimal)")
        else:
            annual_pct = target_annual_pct if target_annual_pct is not None else 2.0
            pi_star_quarterly = (annual_pct / 100.0) / 4.0
            st.info(f"π* set to {annual_pct:.2f}% annual ⇒ {pi_star_quarterly:.4f} quarterly (decimal)")

        # Fit models (Taylor uses inflation gap)
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

        # Build shocks & simulate
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

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        quarters = np.arange(T)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        # IS
        axes[0].plot(quarters, g0*100, label="Baseline", linewidth=2)
        axes[0].plot(quarters, gS*100, label="Shock", linewidth=2)
        axes[0].axvline(shock_quarter, **vline_kwargs)
        axes[0].set_title("Real GDP Growth (DlogGDP, %)")
        axes[0].set_ylabel("%")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        # Phillips
        axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
        axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation (DlogCPI, %)")
        axes[1].set_ylabel("%")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        # Taylor
        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, iS, label="Shock", linewidth=2)
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (decimal)")
        axes[2].set_xlabel("Quarters ahead")
        axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig)

        # ===== LaTeX equations (fitted) =====
        st.subheader("Estimated Equations (Original model, with inflation gap in Taylor)")

        # Pull fitted coefficients
        m_is = models_o["model_is"]
        m_pc = models_o["model_pc"]
        m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]
        phi_pi_star = models_o["phi_pi_star"]
        phi_g_star = models_o["phi_g_star"]
        rho_hat = models_o["rho_hat"]

        # IS
        c_is = float(m_is.params["const"])
        a1 = float(m_is.params["DlogGDP_L1"])
        a2 = float(m_is.params["Real_Rate_L2_data"])
        a3 = float(m_is.params["Dlog FD_Lag1"])
        a4 = float(m_is.params["Dlog_REER"])
        a5 = float(m_is.params["Dlog_Energy"])
        a6 = float(m_is.params["Dlog_NonEnergy"])

        st.markdown("**IS Curve (\\(\\Delta \\log GDP_t\\))**")
        st.latex(
            r"""
            \begin{aligned}
            \Delta \log GDP_t &= {c} \; {a1}\,\Delta \log GDP_{t-1} \; {a2}\,RR_{t-2} \; {a3}\,\Delta \log FD_{t-1} \\
                              &\quad {a4}\,\Delta \log REER_t \; {a5}\,\Delta \log Energy_t \; {a6}\,\Delta \log NonEnergy_t \; + \varepsilon_t
            \end{aligned}
            """.replace("{c}", f"{c_is:.3f}")
              .replace("{a1}", fmt_coef(a1))
              .replace("{a2}", fmt_coef(a2))
              .replace("{a3}", fmt_coef(a3))
              .replace("{a4}", fmt_coef(a4))
              .replace("{a5}", fmt_coef(a5))
              .replace("{a6}", fmt_coef(a6))
        )

        # Phillips
        c_pc = float(m_pc.params["const"])
        b1 = float(m_pc.params["Dlog_CPI_L1"])
        b2 = float(m_pc.params["DlogGDP_L1"])
        b3 = float(m_pc.params["Dlog_Reer_L2"])
        b4 = float(m_pc.params["Dlog_Energy_L1"])
        b5 = float(m_pc.params["Dlog_Non_Energy_L1"])

        st.markdown("**Phillips Curve (\\(\\Delta \\log CPI_t\\))**")
        st.latex(
            r"""
            \begin{aligned}
            \Delta \log CPI_t &= {c} \; {b1}\,\Delta \log CPI_{t-1} \; {b2}\,\Delta \log GDP_{t-1} \; {b3}\,\Delta \log REER_{t-2} \\
                               &\quad {b4}\,\Delta \log Energy_{t-1} \; {b5}\,\Delta \log NonEnergy_{t-1} \; + u_t
            \end{aligned}
            """.replace("{c}", f"{c_pc:.3f}")
              .replace("{b1}", fmt_coef(b1))
              .replace("{b2}", fmt_coef(b2))
              .replace("{b3}", fmt_coef(b3))
              .replace("{b4}", fmt_coef(b4))
              .replace("{b5}", fmt_coef(b5))
        )

        # Taylor with inflation gap (+ policy shock is additive in simulation)
        st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
        st.latex(
            r"""
            i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\,\Big( \alpha^\* \;+\; \phi_{\pi}^\*\,(\pi_t - \pi^\*) \;+\; \phi_{g}^\*\,g_t \Big) \;+\; \varepsilon^{\text{pol}}_t
            """
        )
        st.latex(
            r"""
            \rho = {rho}\,,\quad \alpha^\* = {a}\,,\quad \phi_{\pi}^\* = {fp}\,,\quad \phi_{g}^\* = {fg}\,,\quad \pi^\* = {pistar}
            """.replace("{rho}", f"{rho_hat:.3f}")
              .replace("{a}", f"{alpha_star:.3f}")
              .replace("{fp}", f"{phi_pi_star:.3f}")
              .replace("{fg}", f"{phi_g_star:.3f}")
              .replace("{pistar}", f"{pi_star_quarterly:.4f}")
        )

        # Diagnostics
        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(models_o["model_is"].summary().as_text())
            st.write("**Phillips Curve**"); st.text(models_o["model_pc"].summary().as_text())
            st.write("**Taylor Rule (with inflation gap)**"); st.text(models_o["model_tr"].summary().as_text())

    else:
        # =========================
        # Simple NK (built-in)
        # =========================
        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
            rho_i=rho_i, rho_x=rho_x, rho_r=rho_r, rho_u=rho_u, gamma_pi=gamma_pi
        )
        model = SimpleNK3EqBuiltIn(P)

        label_to_code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}
        code = label_to_code[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))  # convert to 0-based

        # Baseline vs Shock
        h, x0, pi0, i0 = model.irf(shock=code, T=T, size_pp=0.0, t0=t0, rho_override=shock_persist_nk)
        h, xS, piS, iS = model.irf(shock=code, T=T, size_pp=shock_size_pp_nk, t0=t0, rho_override=shock_persist_nk)

        # Plot (all in pp)
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(h, x0, label="Baseline", linewidth=2)
        axes[0].plot(h, xS, label="Shock", linewidth=2)
        axes[0].axvline(t0, **vline_kwargs)
        axes[0].set_title("Output Gap (pp)")
        axes[0].set_ylabel("pp")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(h, pi0, label="Baseline", linewidth=2)
        axes[1].plot(h, piS, label="Shock", linewidth=2)
        axes[1].axvline(t0, **vline_kwargs)
        axes[1].set_title("Inflation (pp)")
        axes[1].set_ylabel("pp")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        axes[2].plot(h, i0, label="Baseline", linewidth=2)
        axes[2].plot(h, iS, label="Shock", linewidth=2)
        axes[2].axvline(t0, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (pp)")
        axes[2].set_xlabel("Quarters ahead")
        axes[2].set_ylabel("pp")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")

        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Simple NK equations (model form)"):
            st.latex(r"x_t = \rho_x x_{t-1} - \frac{1}{\sigma}\big(i_t - \pi_{t+1} - r^n_t \big)")
            st.latex(r"\pi_t = \gamma_\pi \pi_{-1} + \kappa x_t + u_t")
            st.latex(r"i_t = \rho_i i_{t-1} + (1-\rho_i)\big(\phi_\pi \pi_t + \phi_x x_t\big) + \varepsilon^i_t")
            st.json({
                "sigma": sigma, "kappa": kappa, "phi_pi": phi_pi, "phi_x": phi_x,
                "rho_i": rho_i, "rho_x": rho_x, "rho_r": rho_r, "rho_u": rho_u,
                "gamma_pi": gamma_pi, "shock": code, "size_pp": shock_size_pp_nk,
                "t0": t0, "rho_shock": shock_persist_nk
            })

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()





