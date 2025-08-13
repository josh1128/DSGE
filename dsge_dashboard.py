# dsge_dashboard.py
# -----------------------------------------------------------
# DSGE simulation using IS curve, Phillips curve, Taylor rule
# IS: Output Gap dynamics
# Phillips: Inflation vs Output Gap (lagged)
# Taylor: Nominal rate responds to inflation gap + lagged output gap
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="DSGE Model Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — Output Gap, Inflation, Policy Rate")

st.markdown(
    "Use the sidebar to pick **shock type** and size. "
    "IS shock hits output gap directly, Phillips shock hits inflation directly. "
    "Policy responds via Taylor rule using lagged output gap."
)

# =========================================
# Data loading
# =========================================
with st.sidebar:
    st.header("Data source")
    xlf = st.file_uploader("Upload DSGE_Model2.xlsx", type=["xlsx"])
    st.caption("Sheets: 'IS Curve', 'Phillips', 'Taylor'")

    local_fallback = Path(__file__).parent / "DSGE_Model2.xlsx"

@st.cache_data(show_spinner=True)
def load_and_prepare(file_like_or_path):
    # Handle file source
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

    # Strip spaces from column names
    for df in (is_df, pc_df, tr_df):
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Merge
    df = (
        is_df.merge(pc_df, on="Date", how="inner", suffixes=("_is", "_pc"))
             .merge(tr_df, on="Date", how="inner", suffixes=("", "_tr"))
             .sort_values("Date")
             .set_index("Date")
    )

    # Interpolate missing values
    df = df.interpolate(method="linear")

    # Create lags
    df["Output_Gap_L1"] = df["Output Gap"].shift(1)
    df["Output_Gap_L2"] = df["Output Gap"].shift(2)  # For Taylor rule
    df["Inflation_Rate_L1"] = df["Inflation Rate_is"].shift(1)
    df["Nominal_Rate_L1"] = df["Nominal Interest Rate"].shift(1)

    # Drop NA rows for estimation
    df_est = df.dropna().copy()
    return df, df_est

file_source = xlf if xlf is not None else local_fallback
try:
    df_all, df_est = load_and_prepare(file_source)
except Exception as e:
    st.error(f"Problem loading data: {e}")
    st.stop()

# =========================================
# Estimate models
# =========================================
@st.cache_data(show_spinner=True)
def fit_models(df_est):
    # IS curve
    X_is = sm.add_constant(df_est[[
        "Output_Gap_L1", "Foreign Demand", "Non-Energy", "Energy", "REER"
    ]])
    y_is = df_est["Output Gap"]
    model_is = sm.OLS(y_is, X_is).fit()

    # Phillips curve (uses Output Gap L1)
    X_pc = sm.add_constant(df_est[[
        "Inflation_Rate_L1", "Output_Gap_L1", "Foreign Demand", "Non-Energy", "Energy", "REER"
    ]])
    y_pc = df_est["Inflation Rate_is"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor rule (uses Output Gap L2)
    X_tr = sm.add_constant(df_est[[
        "Nominal_Rate_L1", "Inflation Gap", "Output_Gap_L2"
    ]])
    y_tr = df_est["Nominal Interest Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    return model_is, model_pc, model_tr

model_is, model_pc, model_tr = fit_models(df_est)

# =========================================
# Sidebar: simulation controls
# =========================================
with st.sidebar:
    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", 8, 60, 20)
    rho_sim = st.slider("Policy smoothing ρ", 0.0, 0.95, 0.25, 0.05)

    st.header("Shock")
    shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
    shock_quarter = st.slider("Shock timing (t)", 1, T-1, 1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)
    is_shock_size = st.number_input("IS shock size (Δ Output Gap)", value=1.0, step=0.1)
    pc_shock_size = st.number_input("Phillips shock size (Δ Inflation)", value=0.000, step=0.001)

# =========================================
# Build shock arrays
# =========================================
def build_shocks(T, target, is_size, pc_size, t0, rho):
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size
        for k in range(t0+1, T):
            is_arr[k] = rho * is_arr[k-1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size
        for k in range(t0+1, T):
            pc_arr[k] = rho * pc_arr[k-1]
    return is_arr, pc_arr

is_shock_arr, pc_shock_arr = build_shocks(
    T, shock_target, is_shock_size, pc_shock_size, shock_quarter, shock_persist
)

# =========================================
# Simulation
# =========================================
def simulate(T, rho_sim, is_shock_arr=None, pc_shock_arr=None):
    g = np.zeros(T)  # Output Gap
    p = np.zeros(T)  # Inflation
    i = np.zeros(T)  # Nominal Rate

    g[0] = df_est["Output Gap"].mean()
    p[0] = df_est["Inflation Rate_is"].mean()
    i[0] = df_est["Nominal Interest Rate"].mean()

    if is_shock_arr is None:
        is_shock_arr = np.zeros(T)
    if pc_shock_arr is None:
        pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        # IS curve
        Xis = pd.DataFrame([{
            "const": 1.0,
            "Output_Gap_L1": g[t-1],
            "Foreign Demand": df_est["Foreign Demand"].mean(),
            "Non-Energy": df_est["Non-Energy"].mean(),
            "Energy": df_est["Energy"].mean(),
            "REER": df_est["REER"].mean()
        }])
        g[t] = model_is.predict(Xis).iloc[0] + is_shock_arr[t]

        # Phillips curve
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Inflation_Rate_L1": p[t-1],
            "Output_Gap_L1": g[t-1],
            "Foreign Demand": df_est["Foreign Demand"].mean(),
            "Non-Energy": df_est["Non-Energy"].mean(),
            "Energy": df_est["Energy"].mean(),
            "REER": df_est["REER"].mean()
        }])
        p[t] = model_pc.predict(Xpc).iloc[0] + pc_shock_arr[t]

        # Taylor rule (uses Output_Gap_L2)
        lagged_gap = g[t-2] if t >= 2 else g[0]
        Xtr = pd.DataFrame([{
            "const": 1.0,
            "Nominal_Rate_L1": i[t-1],
            "Inflation Gap": p[t] - 0.02,  # target = 2%
            "Output_Gap_L2": lagged_gap
        }])
        i_star = model_tr.predict(Xtr).iloc[0]
        i[t] = rho_sim * i[t-1] + (1 - rho_sim) * i_star

    return g, p, i

# Run baseline vs shock
g0, p0, i0 = simulate(T, rho_sim)
g1, p1, i1 = simulate(T, rho_sim, is_shock_arr, pc_shock_arr)

# =========================================
# Plot
# =========================================
fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axes[0].plot(g0, label="Baseline")
axes[0].plot(g1, label="Shock")
axes[0].set_ylabel("Output Gap")
axes[0].legend()

axes[1].plot(p0, label="Baseline")
axes[1].plot(p1, label="Shock")
axes[1].set_ylabel("Inflation Rate")

axes[2].plot(i0, label="Baseline")
axes[2].plot(i1, label="Shock")
axes[2].set_ylabel("Nominal Rate (level)")
axes[2].set_xlabel("Quarters")

st.pyplot(fig)

