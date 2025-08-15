# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original model (DSGE.xlsx): IS (DlogGDP), Phillips (Dlog_CPI), Taylor (Nominal rate)
#      - Taylor uses inflation gap (π_t − π*)
#      - Shocks: IS, Phillips, and Taylor (tightening/easing)
#      - Policy shock behavior selector:
#          • Add after smoothing (default)
#          • Add to target (inside 1−ρ)
#          • Force local jump (override)  ← Guarantees an uptick/downtick vs last period
#      - LaTeX equations shown below charts
#   2) Simple NK (built-in): 3-eq NK DSGE-lite with tunable parameters
#      - NOW clearly shows which parameters affect which curve
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
    "- **Original**: GDP & CPI in **%** (Dlog × 100); **Nominal rate** in **decimal**.\n"
    "- **Taylor** uses **inflation gap**: \\(\\pi_t - \\pi^*\\)."
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

            # Solve contemporaneously for x_t given policy rule and Phillips
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

    else:
        # ======= Parameter → Curve map (quick card) =======
        st.info("**Which parameters affect which curve?**  \n"
                "• **IS (Demand)**: σ, ρx, ρr  \n"
                "• **Phillips (Supply)**: κ, γπ, ρu  \n"
                "• **Taylor Rule (Policy)**: φπ, φx, ρi")

        st.header("Simple NK parameters (pp units)")

        # -------- IS (Demand) --------
        st.subheader("IS Curve (Demand): controls how rates and shocks move activity (x_t)")
        sigma = st.slider("σ — Demand sensitivity denominator",
                          0.2, 5.0, 1.00, 0.05,
                          help="Affects the **IS curve**. Lower σ ⇒ a given real rate change moves x_t more; "
                               "higher σ ⇒ x_t reacts less.")
        rho_x = st.slider("ρx — Output persistence",
                          0.0, 0.98, 0.50, 0.02,
                          help="Affects the **IS curve** dynamics. Higher ρx ⇒ x_t is more persistent.")
        rho_r = st.slider("ρr — Demand-shock persistence (r^n_t)",
                          0.0, 0.98, 0.80, 0.02,
                          help="Affects **IS shock path**. Higher ρr ⇒ demand shock fades more slowly.")
        st.caption("**IS takeaway:** Lower σ or higher ρx/ρr → output gap moves more/longer after shocks.")

        # -------- Phillips (Supply) --------
        st.subheader("Phillips Curve (Supply): links activity to inflation (π_t)")
        kappa = st.slider("κ — Phillips slope",
                          0.01, 0.50, 0.10, 0.01,
                          help="Affects the **Phillips curve**. Higher κ ⇒ x_t has a bigger impact on π_t.")
        gamma_pi = st.slider("γπ — Inflation inertia",
                             0.0, 0.95, 0.50, 0.05,
                             help="Affects the **Phillips curve** persistence. Higher γπ ⇒ more carryover from π_{t-1}.")
        rho_u = st.slider("ρu — Cost-push shock persistence (u_t)",
                          0.0, 0.98, 0.50, 0.02,
                          help="Affects **Phillips shock path**. Higher ρu ⇒ cost-push shocks linger.")
        st.caption("**Phillips takeaway:** Higher κ/γπ or higher ρu → inflation moves more and/or fades slower.")

        # -------- Taylor Rule (Policy) --------
        st.subheader("Taylor Rule (Policy): sets the interest rate (i_t)")
        phi_pi = st.slider("φπ — Response to inflation",
                           1.0, 3.0, 1.50, 0.05,
                           help="Affects the **Taylor rule**. Larger φπ ⇒ stronger rate moves when inflation deviates from target.")
        phi_x = st.slider("φx — Response to output gap",
                          0.00, 1.00, 0.125, 0.005,
                          help="Affects the **Taylor rule**. Larger φx ⇒ stronger response to x_t.")
        rho_i = st.slider("ρi — Policy rate smoothing",
                          0.0, 0.98, 0.80, 0.02,
                          help="Affects **policy persistence**. Higher ρi ⇒ rates adjust more gradually.")
        st.caption("**Taylor takeaway:** Larger φπ/φx → stronger policy reaction; higher ρi → smoother, slower moves.")

        # ---- Shock controls ----
        st.divider()
        st.header("Shock")
        shock_type_nk = st.selectbox(
            "Shock type (what we 'poke')",
            ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"],
            index=0,
            help="Demand shock (r^n_t) ⇒ IS; Cost-push (u_t) ⇒ Phillips; Policy shock ⇒ Taylor."
        )
        shock_size_pp_nk = st.number_input(
            "Shock size (percentage points, pp)", value=1.00, step=0.25, format="%.2f",
            help="Size of the one-time shock at the chosen quarter, in pp (e.g., 1.00 = one percentage point)."
        )
        shock_quarter_nk = st.slider(
            "Shock timing t (quarter index)", 1, T-1, 1, 1,
            help="Quarter when the shock hits. Plot shows baseline vs. shocked paths."
        )
        shock_persist_nk = st.slider(
            "Shock persistence ρ_shock (for demand/cost)", 0.0, 0.98, 0.80, 0.02,
            help="AR(1) persistence for r^n_t or u_t. Policy shock is one-off (no AR)."
        )

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
# Run selected model
# =========================
try:
    if model_choice == "Original (DSGE.xlsx)":
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
        axes[0].set_title("Real GDP Growth (DlogGDP, %)")
        axes[0].set_ylabel("%")
        axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(quarters, p0*100, label="Baseline", linewidth=2)
        axes[1].plot(quarters, pS*100, label="Shock", linewidth=2)
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation (DlogCPI, %)")
        axes[1].set_ylabel("%")
        axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, iS, label="Shock", linewidth=2)
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (decimal)")
        axes[2].set_xlabel("Quarters ahead")
        axes[2].set_ylabel("decimal")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Readout at the shock quarter
        if shock_target.startswith("Taylor"):
            delta_i_bp = (iS - i0)[shock_quarter] * 10000.0
            st.info(f"Δ policy rate at t={shock_quarter}: {delta_i_bp:.1f} bp  |  mode: {policy_mode}  |  ρ={rho_sim:.2f}")

        # ===== LaTeX equations =====
        st.subheader("Estimated Equations (Original model)")
        m_is = models_o["model_is"]; m_pc = models_o["model_pc"]; m_tr = models_o["model_tr"]
        alpha_star = models_o["alpha_star"]; phi_pi_star = models_o["phi_pi_star"]; phi_g_star = models_o["phi_g_star"]
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
             .replace("{a1}", fmt_coef(a1)).replace("{a2}", fmt_coef(a2))
             .replace("{a3}", fmt_coef(a3)).replace("{a4}", fmt_coef(a4))
             .replace("{a5}", fmt_coef(a5)).replace("{a6}", fmt_coef(a6))
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
             .replace("{b1}", fmt_coef(b1)).replace("{b2}", fmt_coef(b2))
             .replace("{b3}", fmt_coef(b3)).replace("{b4}", fmt_coef(b4))
             .replace("{b5}", fmt_coef(b5))
        )

        # Taylor (display matches chosen mode)
        st.markdown("**Taylor Rule (partial adjustment, with inflation gap)**")
        if policy_mode.startswith("Add after"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t")
        elif policy_mode.startswith("Add to target"):
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\,\big(i_t^\* + \varepsilon^{\text{pol}}_t\big)")
        else:
            st.latex(r"i_t \;=\; \rho\, i_{t-1} \;+\; (1-\rho)\, i_t^\* \;+\; \varepsilon^{\text{pol}}_t \quad (\text{with local-jump override})")
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

        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(m_is.summary().as_text())
            st.write("**Phillips Curve**"); st.text(m_pc.summary().as_text())
            st.write("**Taylor Rule**"); st.text(m_tr.summary().as_text())

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

        # Model key displayed on the page
        st.info("**Model key (Simple NK):**  "
                r"$x_t$ = output gap (pp),  "
                r"$\pi_t$ = inflation (pp),  "
                r"$i_t$ = nominal policy rate (pp).  "
                r"$r_t^n$ = demand/natural-rate shock (pp),  "
                r"$u_t$ = cost-push shock (pp).")

        # Baseline (size 0) vs Shock
        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, shock_persist_nk)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, shock_persist_nk)

        # Plot IRFs (all in pp)
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
- **$x_t$** — Output gap (percentage points, pp), activity above/below normal.  
- **$\pi_t$** — Inflation (pp).  
- **$i_t$** — Nominal policy rate (pp).  
- **$r_t^n$** — Demand / natural-rate shock (pp), AR(1) with $\rho_r$.  
- **$u_t$** — Cost-push shock (pp), AR(1) with $\rho_u$.  
- **$\sigma$** — IS curve: lower ⇒ rates move $x_t$ more; higher ⇒ $x_t$ reacts less.  
- **$\rho_x$** — IS curve: persistence of $x_t$.  
- **$\rho_r$** — IS shock persistence.  
- **$\kappa$** — Phillips curve: strength of $x_t \to \pi_t$.  
- **$\gamma_\pi$** — Phillips curve: inflation inertia.  
- **$\rho_u$** — Phillips shock persistence.  
- **$\phi_\pi$** — Taylor rule: reaction to inflation.  
- **$\phi_x$** — Taylor rule: reaction to output gap.  
- **$\rho_i$** — Taylor rule: interest-rate smoothing.  
- **Shock size (pp)** — One-off change at time $t_0$ (e.g., +1.00 pp).
                """
            )

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()




