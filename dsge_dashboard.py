# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app for the Original model (DSGE.xlsx):
#   - IS Curve (DlogGDP)
#   - Phillips Curve (DlogCPI)
#   - Taylor Rule (Nominal policy rate, decimal)
#
# Changes:
# - All NK model code removed.
# - Three separate plots for IS, Phillips, Taylor.
# - Equations shown under each plot in LaTeX (syntax fixed for Streamlit).
# -----------------------------------------------------------

from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt

# =========================
# Page setup
# =========================
st.set_page_config(page_title="DSGE IRF Dashboard — Original Model", layout="wide")
st.title("DSGE IRF Dashboard — Original Model (IS, Phillips, Taylor)")

st.markdown(
    "- **Dlog variables** (GDP, CPI) are shown in **percent** on the charts.\n"
    "- **Nominal policy rate** is in **decimal** (e.g., 0.035 = 3.5%).\n"
    "- Use the sidebar to add a temporary shock to IS or Phillips."
)

# =========================
# Helpers
# =========================
def ensure_decimal_rate(series: pd.Series) -> pd.Series:
    """Ensure rate is in decimal units (divide by 100 if values look like percentages)."""
    s = pd.to_numeric(series, errors="coerce")
    if np.nanmedian(np.abs(s.values)) > 1.0:
        return s / 100.0
    return s

# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Data")
    upload = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"])
    local_fallback = Path(__file__).parent / "DSGE.xlsx"

    st.header("Simulation")
    T = st.slider("Horizon (quarters)", 8, 60, 20, step=1)
    rho_sim = st.slider("Policy smoothing ρ (Taylor)", 0.0, 0.95, 0.80, 0.05)

    st.header("Shock")
    shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
    is_shock_size_pp = st.number_input("IS shock (Δ DlogGDP, pp)", value=0.50, step=0.10, format="%.2f")
    pc_shock_size_pp = st.number_input("Phillips shock (Δ DlogCPI, pp)", value=0.10, step=0.05, format="%.2f")
    shock_quarter = st.slider("Shock timing t (1 = next quarter)", 1, T - 1, 1, step=1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

# =========================
# Data loading & preparation
# =========================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path: Optional[object]):
    if file_like_or_path is None:
        raise FileNotFoundError("No file provided. Upload DSGE.xlsx or place it beside this script.")

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

    # Parse dates
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    # Merge
    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Convert nominal to decimal
    df["Nominal Rate"] = ensure_decimal_rate(df["Nominal Rate"])

    # Lags/transforms
    df["DlogGDP_L1"]        = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]       = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]   = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L2_data",
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns.")

    return df, df_est

@st.cache_data(show_spinner=True)
def fit_models_original(df_est: pd.DataFrame):
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

    # Taylor
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1", "Dlog_CPI", "DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Long-run conversion
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
        "phi_g_star": phi_g_star,
        "rhoh": rhoh
    }

def build_shocks_original(T, target, is_size_pp, pc_size_pp, t0_1based, rho):
    t0 = max(0, min(T - 1, t0_1based))
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size_pp / 100.0
        for k in range(t0 + 1, T):
            is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size_pp / 100.0
        for k in range(t0 + 1, T):
            pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

def simulate_original(T, rho_sim, df_est, models, means, i_mean_dec, real_rate_mean_dec, is_shock_arr=None, pc_shock_arr=None):
    g = np.zeros(T)  # GDP growth
    p = np.zeros(T)  # Inflation
    i = np.zeros(T)  # Interest rate

    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_mean_dec

    model_is = models["model_is"]
    model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]
    phi_pi_star = models["phi_pi_star"]
    phi_g_star  = models["phi_g_star"]

    if is_shock_arr is None:
        is_shock_arr = np.zeros(T)
    if pc_shock_arr is None:
        pc_shock_arr = np.zeros(T)

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
# Run model
# =========================
try:
    file_source = upload if upload is not None else (local_fallback if local_fallback.exists() else None)
    df_all, df_est = load_and_prepare_original(file_source)
    models_o = fit_models_original(df_est)

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

    is_arr, pc_arr = build_shocks_original(T, shock_target, is_shock_size_pp, pc_shock_size_pp, shock_quarter, shock_persist)
    g0, p0, i0 = simulate_original(T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec)
    gS, pS, iS = simulate_original(T, rho_sim, df_est, models_o, means_o, i_mean_dec, real_rate_mean_dec, is_arr, pc_arr)

    quarters = np.arange(T)
    vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

    # ---------- IS ----------
    fig_is, ax_is = plt.subplots(figsize=(12, 3.6))
    ax_is.plot(quarters, g0 * 100, label="Baseline", linewidth=2)
    ax_is.plot(quarters, gS * 100, label="Shock", linewidth=2)
    if shock_target != "None":
        ax_is.axvline(shock_quarter, **vline_kwargs)
    ax_is.set_title("IS Curve — Real GDP Growth (%)")
    ax_is.set_ylabel("%")
    ax_is.set_xlabel("Quarters ahead")
    ax_is.grid(True, alpha=0.3)
    ax_is.legend()
    st.pyplot(fig_is)
    st.latex(r"\textbf{IS:}\quad \Delta \log GDP_t = \beta_0 + \beta_1\,\Delta \log GDP_{t-1} + \beta_2\,(i_{t-2} - \pi_{t-2}) + \beta_3\,\Delta \log FD_{t-1} + \beta_4\,\Delta \log REER_t + \beta_5\,\Delta \log Energy_t + \beta_6\,\Delta \log NonEnergy_t + \varepsilon_t")

    # ---------- Phillips ----------
    fig_pc, ax_pc = plt.subplots(figsize=(12, 3.6))
    ax_pc.plot(quarters, p0 * 100, label="Baseline", linewidth=2)
    ax_pc.plot(quarters, pS * 100, label="Shock", linewidth=2)
    if shock_target != "None":
        ax_pc.axvline(shock_quarter, **vline_kwargs)
    ax_pc.set_title("Phillips Curve — Inflation (%)")
    ax_pc.set_ylabel("%")
    ax_pc.set_xlabel("Quarters ahead")
    ax_pc.grid(True, alpha=0.3)
    ax_pc.legend()
    st.pyplot(fig_pc)
    st.latex(r"\textbf{Phillips:}\quad \Delta \log CPI_t = \gamma_0 + \gamma_1\,\Delta \log CPI_{t-1} + \gamma_2\,\Delta \log GDP_{t-1} + \gamma_3\,\Delta \log REER_{t-2} + \gamma_4\,\Delta \log Energy_{t-1} + \gamma_5\,\Delta \log NonEnergy_{t-1} + u_t")

    # ---------- Taylor ----------
    fig_tr, ax_tr = plt.subplots(figsize=(12, 3.6))
    ax_tr.plot(quarters, i0, label="Baseline", linewidth=2)
    ax_tr.plot(quarters, iS, label="Shock", linewidth=2)
    if shock_target != "None":
        ax_tr.axvline(shock_quarter, **vline_kwargs)
    ax_tr.set_title("Taylor Rule — Nominal Policy Rate (decimal)")
    ax_tr.set_ylabel("decimal")
    ax_tr.set_xlabel("Quarters ahead")
    ax_tr.grid(True, alpha=0.3)
    ax_tr.legend()
    st.pyplot(fig_tr)
    st.latex(r"\textbf{Taylor:}\quad i_t = \rho\, i_{t-1} + (1-\rho)\,[\alpha + \phi_{\pi}\,\Delta \log CPI_t + \phi_g\,\Delta \log GDP_t] + \nu_t")
    st.latex(r"\textbf{Long-run target:}\quad i_t^{*} = \alpha^{*} + \phi_{\pi}^{*}\,\Delta \log CPI_t + \phi_g^{*}\,\Delta \log GDP_t,\quad \alpha^{*}=\frac{\alpha}{1-\rho},\ \phi_{\pi}^{*}=\frac{\phi_{\pi}}{1-\rho},\ \phi_g^{*}=\frac{\phi_g}{1-\rho}")

except Exception as e:
    st.error(f"Problem loading or running the model: {e}")
    st.stop()


