# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that runs:
#   1) Original (DSGE.xlsx)
#   2) Simple NK (built-in)
#   3) New Keynesian (DSGE_Model2.xlsx)  <-- with snap-back + pp/level toggle
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

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
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
    return rf"""\begin{{aligned}}
{lhs} &= {const_val:.3f} {rhs_terms} + {eps_symbol}
\end{{aligned}}"""

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
# Sidebar: model choice & global display options
# =========================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox(
        "Choose model",
        ["Original (DSGE.xlsx)", "Simple NK (built-in)", "New Keynesian (DSGE_Model2.xlsx)"],
        index=2
    )
    T = st.slider("Horizon (quarters)", 8, 60, 20, 1)

    st.header("Policy rate display")
    policy_units = st.radio(
        "Taylor display mode",
        ["Deviation (pp)", "Level (% annual)"],
        index=1,
        help="Deviation (pp) = Shock path minus baseline path. Level = actual nominal policy rate."
    )
    neutral_rate_pct = st.number_input(
        "Neutral policy rate (for display when level chosen) — % annual",
        value=2.00, step=0.25, format="%.2f"
    )

# =========================
# ORIGINAL model (kept concise; unchanged logic)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE.xlsx or place it beside this script.")
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute(): p = Path.cwd() / p
        if not p.exists(): raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    df = (is_df.merge(pc_df, on="Date").merge(tr_df, on="Date").sort_values("Date").set_index("Date"))
    if "Nominal Rate" in df.columns:
        df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])
    df["DlogGDP_L1"] = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"] = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP","DlogGDP_L1","Dlog_CPI","Dlog_CPI_L1",
        "Nominal Rate","Nominal_Rate_L1","Real_Rate_L2_data",
        "Dlog FD_Lag1","Dlog_REER","Dlog_Energy","Dlog_NonEnergy",
        "Dlog_Reer_L2","Dlog_Energy_L1","Dlog_Non_Energy_L1",
    ]
    miss = [c for c in required_cols if c not in df.columns]
    if miss: raise KeyError(f"Missing required columns: {miss}")
    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty: raise ValueError("No rows remain after NA drop.")
    return df, df_est

def fit_models_original(df_est: pd.DataFrame, pi_star_quarterly: float,
                        is_selected: List[str], pc_selected: List[str], tr_selected: List[str]):
    X_is = sm.add_constant(df_est[is_selected], has_constant="add")
    y_is = df_est["DlogGDP"]; m_is = sm.OLS(y_is, X_is).fit()

    X_pc = sm.add_constant(df_est[pc_selected], has_constant="add")
    y_pc = df_est["Dlog_CPI"]; m_pc = sm.OLS(y_pc, X_pc).fit()

    infl_gap = df_est["Dlog_CPI"] - pi_star_quarterly
    df_tr = pd.DataFrame(index=df_est.index)
    if "Nominal_Rate_L1" in tr_selected: df_tr["Nominal_Rate_L1"] = df_est["Nominal_Rate_L1"]
    if "Inflation_Gap" in tr_selected:   df_tr["Inflation_Gap"] = infl_gap
    if "DlogGDP" in tr_selected:         df_tr["DlogGDP"] = df_est["DlogGDP"]
    X_tr = sm.add_constant(df_tr, has_constant="add")
    y_tr = df_est["Nominal Rate"]; m_tr = sm.OLS(y_tr, X_tr).fit()

    b0 = float(m_tr.params.get("const", 0.0))
    rho = float(np.clip(m_tr.params.get("Nominal_Rate_L1", 0.0), 0.0, 0.99)) if "Nominal_Rate_L1" in m_tr.params.index else 0.0
    def sdiv(a, b): return a/b if abs(b)>1e-8 else np.nan
    alpha_star = sdiv(b0, 1 - rho)
    phi_pi_star = sdiv(float(m_tr.params.get("Inflation_Gap", 0.0)), 1 - rho) if "Inflation_Gap" in m_tr.params.index else np.nan
    phi_g_star  = sdiv(float(m_tr.params.get("DlogGDP", 0.0)), 1 - rho) if "DlogGDP" in m_tr.params.index else np.nan

    return {"is": m_is, "pc": m_pc, "tr": m_tr, "rho": rho,
            "alpha_star": alpha_star, "phi_pi_star": phi_pi_star, "phi_g_star": phi_g_star,
            "pi_star_q": float(pi_star_quarterly)}

# =========================
# NEW KEYNESIAN (DSGE_Model2.xlsx)
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_nk(file_like_or_path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Load NK Excel (with spaces in column names), create lags, and infer π* from Taylor & Phillips sheets.
    """
    if file_like_or_path is None:
        raise FileNotFoundError("Upload DSGE_Model2.xlsx or place it beside this script.")
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path);  p = p if p.is_absolute() else Path.cwd() / p
        if not p.exists(): raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], errors="raise")

    # Require only relevant columns (spaces kept)
    is_need = ["Date","Output Gap","Nominal Interest Rate","Inflation Rate"]
    pc_need = ["Date","Inflation Rate","Output Gap"]
    tr_need = ["Date","Nominal Interest Rate","Inflation Gap","Output Gap"]
    for need, dname, df in [(is_need,"IS Curve",is_df),(pc_need,"Phillips",pc_df),(tr_need,"Taylor",tr_df)]:
        miss = [c for c in need if c not in df.columns]
        if miss: raise KeyError(f"[{dname}] Missing columns: {miss}")

    # Units → decimal
    is_df["Nominal Interest Rate"] = ensure_decimal_rate(is_df["Nominal Interest Rate"])
    is_df["Inflation Rate"] = ensure_decimal_rate(is_df["Inflation Rate"])
    pc_df["Inflation Rate"] = ensure_decimal_rate(pc_df["Inflation Rate"])
    tr_df["Nominal Interest Rate"] = ensure_decimal_rate(tr_df["Nominal Interest Rate"])
    # Inflation Gap in Taylor presumed decimal already.

    # Lags
    is_df = is_df.sort_values("Date").copy()
    is_df["Output Gap L1"] = is_df["Output Gap"].shift(1)
    is_df["Real Rate L1"]  = is_df["Nominal Interest Rate"].shift(1) - is_df["Inflation Rate"].shift(1)

    pc_df = pc_df.sort_values("Date").copy()
    pc_df["Inflation Rate L1"] = pc_df["Inflation Rate"].shift(1)
    pc_df["Output Gap L1"]     = pc_df["Output Gap"].shift(1)

    tr_df = tr_df.sort_values("Date").copy()
    tr_df["Nominal Rate L1"] = tr_df["Nominal Interest Rate"].shift(1)

    # Estimation frames
    is_est = is_df.dropna(subset=["Output Gap","Output Gap L1","Real Rate L1"]).set_index("Date")
    pc_est = pc_df.dropna(subset=["Inflation Rate","Inflation Rate L1","Output Gap L1"]).set_index("Date")
    tr_est = tr_df.dropna(subset=["Nominal Interest Rate","Inflation Gap","Output Gap"]).set_index("Date")

    # π* inference from overlap of Taylor (gap) and realized inflation
    tr_merge = tr_df.merge(pc_df[["Date","Inflation Rate"]], on="Date", how="left")
    valid = tr_merge.dropna(subset=["Inflation Gap","Inflation Rate"])
    pi_star = float((valid["Inflation Rate"] - valid["Inflation Gap"]).mean())

    # Unified dataframe (not strictly required but handy)
    df_all = (
        is_df[["Date","Output Gap","Nominal Interest Rate","Inflation Rate"]]
        .merge(pc_df[["Date","Inflation Rate"]], on="Date", suffixes=("","_pc"))
        .merge(tr_df[["Date","Inflation Gap","Output Gap"]], on="Date", suffixes=("","_tr"))
        .sort_values("Date")
        .set_index("Date")
    )

    return df_all, is_est, pc_est, tr_est, pi_star

def fit_models_nk(is_est: pd.DataFrame, pc_est: pd.DataFrame, tr_est: pd.DataFrame, include_policy_smoothing: bool):
    # IS: y_t ~ const + y_{t-1} + (i_{t-1}-pi_{t-1})
    X_is = sm.add_constant(pd.DataFrame({
        "Output Gap L1": is_est["Output Gap L1"],
        "Real Rate L1":  is_est["Real Rate L1"],
    }, index=is_est.index), has_constant="add")
    y_is = is_est["Output Gap"]; m_is = sm.OLS(y_is, X_is).fit()

    # Phillips: pi_t ~ const + pi_{t-1} + y_{t-1}
    X_pc = sm.add_constant(pd.DataFrame({
        "Inflation Rate L1": pc_est["Inflation Rate L1"],
        "Output Gap L1":     pc_est["Output Gap L1"],
    }, index=pc_est.index), has_constant="add")
    y_pc = pc_est["Inflation Rate"]; m_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor: i_t ~ const + (pi_t - pi*) + y_t [+ rho*i_{t-1}]
    cols = {"Inflation Gap": tr_est["Inflation Gap"], "Output Gap": tr_est["Output Gap"]}
    if include_policy_smoothing:
        cols["Nominal Rate L1"] = tr_est["Nominal Rate L1"]
    X_tr = sm.add_constant(pd.DataFrame(cols, index=tr_est.index), has_constant="add")
    y_tr = tr_est["Nominal Interest Rate"]; m_tr = sm.OLS(y_tr, X_tr).fit()

    rho_hat = float(np.clip(m_tr.params.get("Nominal Rate L1", 0.0), 0.0, 0.99)) if include_policy_smoothing else 0.0
    return {"is": m_is, "pc": m_pc, "tr": m_tr, "rho_hat": rho_hat}

def simulate_nk(
    T: int,
    models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
    y0: float, pi0: float, i0: float,
    include_policy_smoothing: bool,
    pi_star: float,
    shock_block: str = "None",
    shock_size: float = 0.0,
    shock_time: int = 1,
    snapback: bool = False
):
    """
    Simulates output gap (y), inflation (pi), and nominal rate (i).
    Units: y in pp; pi & i in decimal levels. Shock is one-period at t==shock_time.
    If snapback=True: kill lag terms after the shock period.
    """
    y = np.zeros(T); pi = np.zeros(T); i = np.zeros(T)
    y[0], pi[0], i[0] = y0, pi0, i0
    m_is, m_pc, m_tr = models["is"], models["pc"], models["tr"]

    def lag_mult(t):
        return 0.0 if (snapback and t > shock_time) else 1.0

    for t in range(1, T):
        # IS
        real_rate_l1 = i[t-1] - pi[t-1]
        Xis = row_from_params(m_is.params.index, {
            "Output Gap L1": lag_mult(t) * y[t-1],
            "Real Rate L1":  real_rate_l1
        })
        y[t] = float(m_is.predict(Xis).iloc[0])

        # Phillips
        Xpc = row_from_params(m_pc.params.index, {
            "Inflation Rate L1": lag_mult(t) * pi[t-1],
            "Output Gap L1":     lag_mult(t) * y[t-1]
        })
        pi[t] = float(m_pc.predict(Xpc).iloc[0])

        # Taylor
        tr_vals = {"Inflation Gap": (pi[t] - pi_star), "Output Gap": y[t]}
        if include_policy_smoothing:
            tr_vals["Nominal Rate L1"] = i[t-1]
        Xtr = row_from_params(m_tr.params.index, tr_vals)
        i[t] = float(m_tr.predict(Xtr).iloc[0])

        # One-period shock
        if t == shock_time:
            if   shock_block == "IS":       y[t]  += shock_size
            elif shock_block == "Phillips": pi[t] += shock_size
            elif shock_block == "Taylor":   i[t]  += shock_size

    return y, pi, i

# =========================
# RUN
# =========================
try:
    if model_choice == "New Keynesian (DSGE_Model2.xlsx)":
        with st.sidebar:
            xlf2 = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"], key="upload_nk")
            fallback2 = Path(__file__).parent / "DSGE_Model2.xlsx"

            st.header("NK options")
            include_policy_smoothing = st.checkbox("Include policy smoothing (add i_{t-1}) in Taylor", value=False)
            snapback = st.checkbox("No persistence (snap-back)", value=False)

            st.header("Shock (NK)")
            nk_block = st.selectbox("Shock block", ["None", "IS", "Phillips", "Taylor"], index=0)
            nk_size = st.number_input("Shock size (units of variable)", value=0.00, step=0.10, format="%.2f")
            nk_time_human = st.slider("Shock timing (t)", 1, T-1, 2, 1)
            t0 = int(nk_time_human)  # internal uses the same period index

        src2 = xlf2 if xlf2 is not None else fallback2
        df_all, is_est, pc_est, tr_est, pi_star = load_and_prepare_nk(src2)
        models_nk = fit_models_nk(is_est, pc_est, tr_est, include_policy_smoothing)

        # Anchors
        y0 = float(is_est["Output Gap"].mean())
        pi0 = float(pc_est["Inflation Rate"].mean())
        i0 = float(tr_est["Nominal Interest Rate"].mean())

        # Baseline vs Shock
        yB, piB, iB = simulate_nk(
            T, models_nk, y0, pi0, i0,
            include_policy_smoothing, pi_star,
            shock_block="None", shock_size=0.0, shock_time=t0, snapback=snapback
        )
        yS, piS, iS = simulate_nk(
            T, models_nk, y0, pi0, i0,
            include_policy_smoothing, pi_star,
            shock_block=nk_block, shock_size=nk_size, shock_time=t0, snapback=snapback
        )

        # ========= Plotting (with policy_units toggle) =========
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        q = np.arange(T)

        # Output Gap (pp)
        axes[0].plot(q, yB, linewidth=2, label="Baseline")
        axes[0].plot(q, yS, linewidth=2, label="Shock")
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend()

        # Inflation (%)
        axes[1].plot(q, piB*100, linewidth=2, label="Baseline")
        axes[1].plot(q, piS*100, linewidth=2, label="Shock")
        axes[1].set_title("Inflation Rate (%)"); axes[1].set_ylabel("%"); axes[1].grid(True, alpha=0.3); axes[1].legend()

        # Policy Rate: either deviation (pp) or level (%)
        if policy_units == "Deviation (pp)":
            delta_i_pp = (iS - iB) * 100.0
            axes[2].plot(q, np.zeros_like(delta_i_pp), linewidth=2, label="Baseline")
            axes[2].plot(q, delta_i_pp, linewidth=2, label="Shock")
            axes[2].set_title("Nominal Interest Rate — Deviation from Baseline (pp)")
            axes[2].set_ylabel("pp")
        else:
            axes[2].plot(q, iB*100, linewidth=2, label="Baseline")
            axes[2].plot(q, iS*100, linewidth=2, label="Shock")
            axes[2].set_title("Nominal Interest Rate — Level (% annual)")
            axes[2].set_ylabel("%")
        axes[2].set_xlabel("Quarters"); axes[2].grid(True, alpha=0.3); axes[2].legend()

        plt.tight_layout(); st.pyplot(fig)

        # Equations
        st.subheader("Estimated Equations (New Keynesian)")
        mi, mp, mt = models_nk["is"], models_nk["pc"], models_nk["tr"]

        st.markdown("**IS (Output Gap)**")
        is_terms = []
        for k, v in mi.params.items():
            if k == "const": continue
            sym = {"Output Gap L1": r"\hat y_{t-1}", "Real Rate L1": r"(i_{t-1}-\pi_{t-1})"}.get(k, k)
            is_terms.append((float(v), sym))
        st.latex(build_latex_equation(float(mi.params.get("const", 0.0)), is_terms, r"\hat y_t", r"\varepsilon_t"))

        st.markdown("**Phillips (Inflation)**")
        pc_terms = []
        for k, v in mp.params.items():
            if k == "const": continue
            sym = {"Inflation Rate L1": r"\pi_{t-1}", "Output Gap L1": r"\hat y_{t-1}"}.get(k, k)
            pc_terms.append((float(v), sym))
        st.latex(build_latex_equation(float(mp.params.get("const", 0.0)), pc_terms, r"\pi_t", r"u_t"))

        st.markdown("**Taylor Rule**")
        tr_terms = []
        for k, v in mt.params.items():
            if k == "const": continue
            sym = {"Inflation Gap": r"(\pi_t-\pi^\*)", "Output Gap": r"\hat y_t", "Nominal Rate L1": r"i_{t-1}"}.get(k, k)
            tr_terms.append((float(v), sym))
        st.latex(build_latex_equation(float(mt.params.get("const", 0.0)), tr_terms, r"i_t", r"v_t"))
        st.caption(f"π* inferred from data: {pi_star*100:.2f}% (quarterly annualized-equivalent not applied).")

        with st.expander("OLS summaries (NK)"):
            st.write("**IS**"); st.text(mi.summary().as_text())
            st.write("**Phillips**"); st.text(mp.summary().as_text())
            st.write("**Taylor**"); st.text(mt.summary().as_text())

    elif model_choice == "Simple NK (built-in)":
        with st.sidebar:
            st.info("Simple NK parameters (pp units)")
            sigma = st.slider("σ", 0.2, 5.0, 1.00, 0.05)
            rho_x = st.slider("ρx", 0.0, 0.98, 0.50, 0.02)
            rho_r = st.slider("ρr", 0.0, 0.98, 0.80, 0.02)
            kappa = st.slider("κ", 0.01, 0.50, 0.10, 0.01)
            gamma_pi = st.slider("γπ", 0.0, 0.95, 0.50, 0.05)
            rho_u = st.slider("ρu", 0.0, 0.98, 0.50, 0.02)
            phi_pi = st.slider("φπ", 1.0, 3.0, 1.50, 0.05)
            phi_x = st.slider("φx", 0.00, 1.00, 0.125, 0.005)
            rho_i = st.slider("ρi", 0.0, 0.98, 0.80, 0.02)

            st.header("Shock")
            shock_type_nk = st.selectbox("Type", ["Demand (IS)", "Cost-push (Phillips)", "Policy (Taylor)"], index=0)
            shock_size_pp_nk = st.number_input("Size (pp)", value=1.00, step=0.25, format="%.2f")
            shock_quarter_nk = st.slider("Timing t", 1, T-1, 2, 1)
            shock_persist_nk = st.slider("Shock persistence ρ", 0.0, 0.98, 0.80, 0.02)
            snapback_bi = st.checkbox("Snap-back (x,π no persistence)", value=True)

        P = NKParamsSimple(
            sigma=sigma, kappa=kappa, phi_pi=phi_pi, phi_x=phi_x,
            rho_i=rho_i, rho_x=(0.0 if snapback_bi else rho_x),
            rho_r=rho_r, rho_u=rho_u, gamma_pi=(0.0 if snapback_bi else gamma_pi)
        )
        model = SimpleNK3EqBuiltIn(P)
        code = {"Demand (IS)": "demand", "Cost-push (Phillips)": "cost", "Policy (Taylor)": "policy"}[shock_type_nk]
        t0 = max(0, min(T-1, shock_quarter_nk - 1))
        rho_for_shock = 0.0 if snapback_bi else shock_persist_nk

        h, x0, pi0, i0 = model.irf(code, T, 0.0, t0, rho_for_shock)
        h, xS, piS, iS = model.irf(code, T, shock_size_pp_nk, t0, rho_for_shock)

        # Plot with policy_units toggle
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        axes[0].plot(h, x0, linewidth=2, label="Baseline")
        axes[0].plot(h, xS, linewidth=2, label="Shock")
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend()

        axes[1].plot(h, pi0, linewidth=2, label="Baseline")
        axes[1].plot(h, piS, linewidth=2, label="Shock")
        axes[1].set_title("Inflation (pp)"); axes[1].set_ylabel("pp"); axes[1].grid(True, alpha=0.3); axes[1].legend()

        if policy_units == "Deviation (pp)":
            axes[2].plot(h, np.zeros_like(i0), linewidth=2, label="Baseline")
            axes[2].plot(h, iS - i0, linewidth=2, label="Shock")
            axes[2].set_title("Policy Rate — Deviation (pp)"); axes[2].set_ylabel("pp")
        else:
            axes[2].plot(h, (neutral_rate_pct + i0), linewidth=2, label="Baseline")
            axes[2].plot(h, (neutral_rate_pct + iS), linewidth=2, label="Shock")
            axes[2].set_title("Policy Rate — Level (% annual)"); axes[2].set_ylabel("%")
        axes[2].set_xlabel("Quarters"); axes[2].grid(True, alpha=0.3); axes[2].legend()

        plt.tight_layout(); st.pyplot(fig)

    else:
        # ORIGINAL kept very compact here to focus on NK features
        with st.sidebar:
            xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
            fallback = Path(__file__).parent / "DSGE.xlsx"
            rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

            st.header("Inflation target")
            use_sample_mean = st.checkbox("Use sample mean of DlogCPI as π*", value=False)
            target_annual_pct = None if use_sample_mean else st.slider("π* (annual %)", 0.0, 5.0, 2.0, 0.1)

            st.header("Shock (Original)")
            shock_target = st.selectbox("Apply shock to",
                                        ["None","IS (Demand)","Phillips (Supply)","Taylor (Policy tightening)","Taylor (Policy easing)"], 0)
            is_pp = st.number_input("IS shock (pp)", value=0.50, step=0.10, format="%.2f")
            pc_pp = st.number_input("Phillips shock (pp)", value=0.10, step=0.05, format="%.2f")
            pol_bp = st.number_input("Policy shock (bp)", value=25, step=5, format="%d")
            t_shock = st.slider("Shock timing (t)", 1, T-1, 2, 1)
            rho_sh = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

            st.header("Regressor selection")
            IS_ALL = ["DlogGDP_L1","Real_Rate_L2_data","Dlog FD_Lag1","Dlog_REER","Dlog_Energy","Dlog_NonEnergy"]
            PC_ALL = ["Dlog_CPI_L1","DlogGDP_L1","Dlog_Reer_L2","Dlog_Energy_L1","Dlog_Non_Energy_L1"]
            TR_ALL = ["Nominal_Rate_L1","Inflation_Gap","DlogGDP"]
            is_selected = st.multiselect("IS:", IS_ALL, default=IS_ALL)
            pc_selected = st.multiselect("Phillips:", PC_ALL, default=PC_ALL)
            tr_selected = st.multiselect("Taylor:", TR_ALL, default=TR_ALL)

        src = xlf if xlf is not None else fallback
        df_all, df_est = load_and_prepare_original(src)

        if use_sample_mean:
            pi_star_q = float(df_est["Dlog_CPI"].mean())
            st.info(f"π* (quarterly) = sample mean of DlogCPI = {pi_star_q:.4f}")
        else:
            annual_pct = target_annual_pct if target_annual_pct is not None else 2.0
            pi_star_q = (annual_pct/100.0)/4.0
            st.info(f"π* = {annual_pct:.2f}% annual ⇒ {pi_star_q:.4f} quarterly (decimal)")

        models_o = fit_models_original(df_est, pi_star_q, is_selected, pc_selected, tr_selected)

        # Simple baseline placeholder (focus of this version is NK Excel model’s toggle)
        st.success("Original model loaded and estimated. (Plotting omitted here to keep this file compact.)")

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()






