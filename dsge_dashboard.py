# dsge_dashboard.py
# -----------------------------------------------------------
# This app lets you upload an Excel file, estimate 3 simple
# macro relationships (IS, Phillips, Taylor), and simulate
# what happens to GDP growth, inflation, and the policy rate
# after a "shock." Results are shown as charts.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm            # OLS
import streamlit as st                  # UI
import matplotlib.pyplot as plt         # charts
from pathlib import Path                # locating Excel file
import matplotlib.dates as mdates       # nice date ticks

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="DSGE Model Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — IS (Demand), Phillips (Supply), Taylor (Policy)")

st.markdown(
    "Use the sidebar to pick shock type and size. "
    "**IS shock** hits GDP growth directly. **Phillips shock** hits inflation directly. "
    "Policy responds via a partial-adjustment Taylor rule."
)

# =========================================
# Data source
# =========================================
with st.sidebar:
    st.header("Data source")
    xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"])
    st.caption("Sheets required: 'IS Curve', 'Phillips', 'Taylor' (Date format: YYYY-MM)")
    # If no upload, fall back to a file named DSGE.xlsx beside this script
    local_fallback = Path(__file__).parent / "DSGE.xlsx"

@st.cache_data(show_spinner=True)
def load_and_prepare(file_like_or_path):
    """
    Loads Excel, merges the 3 sheets on Date, creates lags/transforms,
    and returns (df_all, df_est).
    """
    if file_like_or_path is None:
        raise FileNotFoundError(
            "No file provided. Upload DSGE.xlsx or include DSGE.xlsx in the repo folder."
        )

    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path  # uploaded file-like

    # Read required sheets
    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    # Parse dates like "2010-03"
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    # Merge on Date
    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Lags / transforms used in models
    df["DlogGDP_L1"]        = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]       = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]   = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    # Required columns (adjust to your Excel headers if needed)
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
        raise KeyError(
            f"Missing required columns in merged dataframe: {missing}. "
            "Adjust required_cols/X matrices to match your Excel headers."
        )

    # Drop rows with NAs in required columns to make OLS fit cleanly
    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    return df, df_est

# Pick source: upload wins, else local fallback
file_source = xlf if xlf is not None else local_fallback

try:
    df_all, df_est = load_and_prepare(file_source)
except Exception as e:
    st.error(f"Problem loading data: {e}")
    st.stop()

# =========================================
# Estimate equations (OLS)
# =========================================
@st.cache_data(show_spinner=True)
def fit_models(df_est):
    # --- IS curve: GDP growth today ~ last quarter's growth + old real rate + externals
    X_is = sm.add_constant(df_est[[
        "DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"
    ]])
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # --- Phillips: inflation today ~ last inflation + last GDP growth + price externals
    X_pc = sm.add_constant(df_est[[
        "Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]])
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # --- Taylor (partial adjustment): policy rate today ~ last rate + current inflation + current GDP growth
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1", "Dlog_CPI", "DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Convert to long-run targets for the policy rule
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

models = fit_models(df_est)

# Baseline means for simulations
i_neutral      = float(df_est["Nominal Rate"].mean())
real_rate_mean = float(df_est["Real_Rate_L2_data"].mean())
means = {
    "Dlog FD_Lag1":       float(df_est["Dlog FD_Lag1"].mean()),
    "Dlog_REER":          float(df_est["Dlog_REER"].mean()),
    "Dlog_Energy":        float(df_est["Dlog_Energy"].mean()),
    "Dlog_NonEnergy":     float(df_est["Dlog_NonEnergy"].mean()),
    "Dlog_Reer_L2":       float(df_est["Dlog_Reer_L2"].mean()),
    "Dlog_Energy_L1":     float(df_est["Dlog_Energy_L1"].mean()),
    "Dlog_Non_Energy_L1": float(df_est["Dlog_Non_Energy_L1"].mean()),
}

# =========================================
# Sidebar: simulation controls
# =========================================
with st.sidebar:
    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)
    rho_sim = st.slider("Policy smoothing ρ (0 = fast, 0.95 = slow)", 0.0, 0.95, 0.25, 0.05)

    st.header("Shock")
    shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
    shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)
    is_shock_size = st.number_input("IS shock size (Δ DlogGDP)", value=1.0, step=0.1, format="%.3f")
    pc_shock_size = st.number_input("Phillips shock size (Δ DlogCPI)", value=0.000, step=0.001, format="%.3f")

    st.header("Historical plot options")
    rate_mode = st.radio("Policy rate line shows…", ["Level", "Change (Δ, pp)"], index=0, horizontal=True)

# =========================================
# Shocks
# =========================================
def build_shocks(T, target, is_size, pc_size, t0, rho):
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size
        for k in range(t0 + 1, T):
            is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size
        for k in range(t0 + 1, T):
            pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

is_shock_arr, pc_shock_arr = build_shocks(
    T, shock_target, is_shock_size, pc_shock_size, shock_quarter, shock_persist
)

# =========================================
# Simulation engine
# =========================================
def simulate(T, rho_sim, is_shock_arr=None, pc_shock_arr=None):
    g = np.zeros(T)  # DlogGDP
    p = np.zeros(T)  # DlogCPI
    i = np.zeros(T)  # Nominal rate (level)

    # Start at steady-ish values
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_neutral

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
        # Real rate with a 2-quarter lag (fallback to mean early on)
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean

        # IS block
        Xis = pd.DataFrame([{
            "const": 1.0,
            "DlogGDP_L1": g[t - 1],
            "Real_Rate_L2_data": rr_lag2,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": means["Dlog_Energy"],
            "Dlog_NonEnergy": means["Dlog_NonEnergy"],
        }])
        g[t] = model_is.predict(Xis).iloc[0] + is_shock_arr[t]

        # Phillips block
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }])
        p[t] = model_pc.predict(Xpc).iloc[0] + pc_shock_arr[t]

        # Taylor with partial adjustment
        i_star = alpha_star + phi_pi_star * p[t] + phi_g_star * g[t]
        i[t]   = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return g, p, i

# Run: baseline vs scenario
g0, p0, i0 = simulate(T=T, rho_sim=rho_sim)
g1, p1, i1 = simulate(T=T, rho_sim=rho_sim, is_shock_arr=is_shock_arr, pc_shock_arr=pc_shock_arr)

# =========================================
# Plot results (three-panel IRFs)
# =========================================
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
quarters = np.arange(T)
vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

# GDP Growth
axes[0].plot(quarters, g0, label="Baseline", linewidth=2)
axes[0].plot(quarters, g1, label="Scenario", linewidth=2)
axes[0].axhline(float(df_est["DlogGDP"].mean()), ls="--", label="Steady State")
axes[0].axvline(shock_quarter, **vline_kwargs)
axes[0].set_title("Real GDP Growth (DlogGDP)")
axes[0].set_ylabel("DlogGDP")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best")

# Inflation
axes[1].plot(quarters, p0, label="Baseline", linewidth=2)
axes[1].plot(quarters, p1, label="Scenario", linewidth=2)
axes[1].axhline(float(df_est["Dlog_CPI"].mean()), ls="--", label="Steady State")
axes[1].axvline(shock_quarter, **vline_kwargs)
axes[1].set_title("Inflation Rate (DlogCPI)")
axes[1].set_ylabel("DlogCPI")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best")

# Policy Rate (level)
axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
axes[2].plot(quarters, i1, label="Scenario", linewidth=2)
axes[2].axhline(i_neutral, ls="--", label="Neutral Rate")
axes[2].axvline(shock_quarter, **vline_kwargs)
axes[2].set_title("Nominal Policy Rate")
axes[2].set_xlabel("Quarters ahead")
axes[2].set_ylabel("Nominal Rate")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="best")

plt.tight_layout()
st.pyplot(fig)

# =========================================
# Combined Scenario Plot (optional)
# =========================================
# First differences to show nominal rate "growth" if desired
di1 = np.r_[0.0, np.diff(i1)]  # scenario Δi in percentage points

fig2 = plt.figure(figsize=(12, 4.5))
plt.plot(quarters, g1, linewidth=2, label="GDP growth (DlogGDP)")
plt.plot(quarters, p1, linewidth=2, label="CPI growth (DlogCPI)")
plt.plot(quarters, di1, linewidth=2, label="Nominal rate change (Δ, pp)")
plt.axvline(shock_quarter, **vline_kwargs)
plt.title("Combined: Growth / Changes (Scenario)")
plt.xlabel("Quarters ahead")
plt.ylabel("Growth / Change")
plt.grid(True, alpha=0.3)
plt.legend(loc="best")
plt.tight_layout()
st.pyplot(fig2)

# =========================================
# NEW: Historical time series (like your reference chart)
# =========================================
# Build a clean plotting frame from historical data
df_plot = df_all.copy().reset_index()

# Helper: scale to percent if series looks like decimals (e.g., 0.01 = 1%)
def to_percent(s: pd.Series) -> pd.Series:
    return s * 100 if s.abs().max() < 5 else s

g_hist = to_percent(df_plot["DlogGDP"])
p_hist = to_percent(df_plot["Dlog_CPI"])

if rate_mode == "Change (Δ, pp)":
    r_hist = df_plot["Nominal Rate"].diff()          # percentage points change
    rate_label = "Nominal rate change (Δ, pp)"
else:
    r_hist = df_plot["Nominal Rate"]                 # level
    rate_label = "Nominal policy rate (level)"

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(df_plot["Date"], g_hist, linewidth=2, label="GDP growth")
ax3.plot(df_plot["Date"], p_hist, linewidth=2, label="Inflation")
ax3.plot(df_plot["Date"], r_hist, linewidth=2, label=rate_label)

ax3.set_title("Inflation, GDP Growth, and Policy Rate — Historical")
ax3.set_xlabel("Date")
ax3.set_ylabel("Percent / Percentage points")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Year ticks every ~5 years
ax3.xaxis.set_major_locator(mdates.YearLocator(base=5))
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
st.pyplot(fig3)

# =========================================
# Diagnostics
# =========================================
with st.expander("Model diagnostics (OLS summaries)"):
    st.write("**IS Curve**")
    st.text(models["model_is"].summary().as_text())
    st.write("**Phillips Curve**")
    st.text(models["model_pc"].summary().as_text())
    st.write("**Taylor Rule**")
    st.text(models["model_tr"].summary().as_text())
