# dsge_dashboard.py
# -----------------------------------------------------------
# This app lets you upload an Excel file, estimate 3 simple
# macro relationships (IS, Phillips, Taylor), and simulate
# what happens to GDP growth, inflation, and the policy rate
# after a "shock." Results are shown as charts.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm            # used to run simple regressions (OLS)
import streamlit as st                  # used to build the web app UI
import matplotlib.pyplot as plt         # used to make charts
from pathlib import Path                # helps find the Excel file on disk

# =========================================
# Page setup
# =========================================
# Sets the browser tab title and layout width
st.set_page_config(page_title="DSGE Model Dashboard", layout="wide")

# Big page title the user sees
st.title("DSGE IRF Dashboard — IS (Demand), Phillips (Supply), Taylor (Policy)")

# Short explanation at the top of the page
st.markdown(
    "Use the sidebar to pick shock type and size. "
    "**IS shock** hits GDP growth directly. **Phillips shock** hits inflation directly. "
    "Policy responds via a partial-adjustment Taylor rule."
)

# =========================================
# Data source (where the Excel file comes from)
# =========================================
with st.sidebar:
    st.header("Data source")

    # Option 1: user uploads the Excel file by clicking here
    xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"])
    st.caption("Sheets required: 'IS Curve', 'Phillips', 'Taylor' (Date format: YYYY-MM)")

    # Option 2 (fallback): if they don't upload a file, we look for DSGE.xlsx
    # in the same folder as this script.
    # If the app is on GitHub, put DSGE.xlsx in the repo with this file.
    local_fallback = Path(__file__).parent / "DSGE.xlsx"

@st.cache_data(show_spinner=True)
def load_and_prepare(file_like_or_path):
    """
    Loads the Excel file, reads the 3 sheets, merges them on Date,
    creates a few lagged variables, and returns:
      - df:     the full merged dataset
      - df_est: a cleaned version with no missing values in required columns
    If anything is missing or misnamed, we raise a clear error message.
    """

    # If no file is provided at all, stop early with a helpful message.
    if file_like_or_path is None:
        raise FileNotFoundError(
            "No file provided. Upload DSGE.xlsx or include DSGE.xlsx in the repo folder."
        )

    # If we got a path (string/Path), check that it exists. If we got an uploaded file,
    # just pass it through. (Streamlit gives us a file-like object when a user uploads.)
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        # If path is relative, resolve it relative to where we’re running
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            # Non-technical users: if you see this, the file isn’t where we expect.
            # Put DSGE.xlsx next to this .py file or upload it via the sidebar.
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path  # uploaded file object

    # Read the 3 sheets we need. Make sure your Excel sheet names match exactly.
    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    # Convert the Date column from text (e.g., "2010-03") to a real date.
    # If your Excel file uses a different format, change "%Y-%m" here.
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    # Merge the three sheets together using the Date column
    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Create lagged versions of variables we’ll use in the models.
    # A "lag" just means "previous quarter’s value".
    df["DlogGDP_L1"]        = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]       = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]   = df["Nominal Rate"].shift(1)
    # Real rate = nominal - inflation, and then take the 2-quarter lag
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    # --------------------------------------------------------
    # CHANGE ME (if your Excel column headers are different):
    # Make sure these names match your Excel sheet columns.
    # If you get a "Missing required columns" error, fix names here.
    # --------------------------------------------------------
    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L2_data",
        # IS externals
        "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy",
        # Phillips externals (lagged)
        "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]

    # If any of those columns are missing, stop with a helpful message.
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in merged dataframe: {missing}. "
            "Adjust required_cols/X matrices to match your Excel headers."
        )

    # Drop rows that have any missing values in the required columns.
    # This avoids crashes when running the regressions.
    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    return df, df_est

# Pick our data source: uploaded file wins; otherwise, use the local fallback file
file_source = xlf if xlf is not None else local_fallback

try:
    df_all, df_est = load_and_prepare(file_source)
except Exception as e:
    # Prints a big red error box in the app if something goes wrong, then stops.
    st.error(f"Problem loading data: {e}")
    st.stop()

# =========================================
# Estimate equations (OLS = simple linear regression)
# =========================================
@st.cache_data(show_spinner=True)
def fit_models(df_est):
    """
    Fits 3 separate regressions:
      1) IS curve  -> predicts GDP growth (DlogGDP)
      2) Phillips  -> predicts inflation (Dlog_CPI)
      3) Taylor    -> predicts the nominal policy rate
    Returns the fitted models and some long-run coefficients for the policy rule.
    """

    # --- IS curve: GDP growth today ~ last quarter's growth + old real rate + externals
    X_is = sm.add_constant(df_est[[
        "DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"
    ]])
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # --- Phillips: inflation today ~ last quarter's inflation + last quarter's GDP growth + price externals
    X_pc = sm.add_constant(df_est[[
        "Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]])
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # --- Taylor (partial adjustment): policy rate today ~ last rate + current inflation + current GDP growth
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1", "Dlog_CPI", "DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Turn the Taylor regression into long-run weights for a "target" rate i*
    # We cap the persistence (rhoh) at 0.99 to avoid weird behavior in simulation.
    b0   = float(model_tr.params["const"]) #intercept for taylor 
    rhoh = min(float(model_tr.params["Nominal_Rate_L1"]), 0.99) #persistence of policy rate
    bpi  = float(model_tr.params["Dlog_CPI"]) # tells you how muh the central bank raises or lower the rate when inflation changes by 1 unit 
    bg   = float(model_tr.params["DlogGDP"]) # How much the policy rate moves when GDP growth changes

    # These convert short-run regression coefficients into long-run effects
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

# Baseline means for simulations (used as steady-state anchors)
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
# Sidebar: user picks simulation knobs
# =========================================
with st.sidebar:
    st.header("Simulation settings")

    # How many quarters ahead to simulate?
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)

    # How sticky is the policy rate? Higher = slower to move.
    rho_sim = st.slider("Policy smoothing ρ (0 = fast, 0.95 = slow)", 0.0, 0.95, 0.25, 0.05)

    st.header("Shock")

    # Choose which block gets the shock, when it hits, how long it lasts, and how big it is.
    shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
    shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

    # Size of the shock:
    # - IS shock adds directly to GDP growth in that period.
    # - Phillips shock adds directly to inflation in that period.
    is_shock_size = st.number_input("IS shock size (Δ DlogGDP)", value=1.0, step=0.1, format="%.3f")
    pc_shock_size = st.number_input("Phillips shock size (Δ DlogCPI)", value=0.000, step=0.001, format="%.3f")

# =========================================
# Build shock paths over time (so they can decay each quarter)
# =========================================
def build_shocks(T, target, is_size, pc_size, t0, rho):
    """
    Creates two arrays (one for GDP growth shocks and one for inflation shocks).
    The shock starts at quarter t0, and then shrinks each quarter by factor rho.
    """
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
# Simulation engine (the heart of the app)
# =========================================
def simulate(T, rho_sim, is_shock_arr=None, pc_shock_arr=None):
    """
    Steps the economy forward T quarters.
    - g[t]: GDP growth this quarter
    - p[t]: inflation this quarter
    - i[t]: policy rate this quarter
    Uses the estimated models plus the shocks selected in the sidebar.
    """

    g = np.zeros(T)  # GDP growth (DlogGDP)
    p = np.zeros(T)  # Inflation   (DlogCPI)
    i = np.zeros(T)  # Nominal policy rate

    # Start near steady-state values (typical/average levels)
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_neutral

    model_is = models["model_is"]
    model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]
    phi_pi_star = models["phi_pi_star"]
    phi_g_star  = models["phi_g_star"]

    # If no shocks passed in, use zeros (no shock)
    if is_shock_arr is None:
        is_shock_arr = np.zeros(T)
    if pc_shock_arr is None:
        pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        # Compute the real interest rate with a 2-quarter lag.
        # Early on (t < 2) we fall back to an average so the code doesn’t crash.
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean

        # --- IS block: predict GDP growth from last quarter's growth, lagged real rate, and externals
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

        # --- Phillips block: predict inflation from last inflation, last GDP growth, and price externals
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }])
        p[t] = model_pc.predict(Xpc).iloc[0] + pc_shock_arr[t]

        # --- Taylor rule with partial adjustment:
        # First compute the "target" rate i* (what the central bank wants)
        i_star = alpha_star + phi_pi_star * p[t] + phi_g_star * g[t]
        # Then move the actual rate part of the way toward i* (rho controls how slowly)
        i[t]   = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return g, p, i

# Run two simulations:
#  - Baseline: no shocks
#  - Scenario: with your chosen shock settings
g0, p0, i0 = simulate(T=T, rho_sim=rho_sim)  # baseline (no shock)
g1, p1, i1 = simulate(T=T, rho_sim=rho_sim, is_shock_arr=is_shock_arr, pc_shock_arr=pc_shock_arr)

# =========================================
# Plot results
# =========================================
# Make chart text a bit larger
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

quarters = np.arange(T)
vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

# --- GDP Growth chart
axes[0].plot(quarters, g0, label="Baseline", linewidth=2)
axes[0].plot(quarters, g1, label="Scenario", linewidth=2)
axes[0].axhline(float(df_est["DlogGDP"].mean()), ls="--", color="gray", label="Steady State")
axes[0].axvline(shock_quarter, **vline_kwargs)
axes[0].set_title("Real GDP Growth (DlogGDP)")
axes[0].set_ylabel("DlogGDP")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best")

# --- Inflation chart
axes[1].plot(quarters, p0, label="Baseline", linewidth=2)
axes[1].plot(quarters, p1, label="Scenario", linewidth=2)
axes[1].axhline(float(df_est["Dlog_CPI"].mean()), ls="--", color="gray", label="Steady State")
axes[1].axvline(shock_quarter, **vline_kwargs)
axes[1].set_title("Inflation Rate (DlogCPI)")
axes[1].set_ylabel("DlogCPI")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best")

# --- Policy Rate chart
axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
axes[2].plot(quarters, i1, label="Scenario", linewidth=2)
axes[2].axhline(i_neutral, ls="--", color="gray", label="Neutral Rate")
axes[2].axvline(shock_quarter, **vline_kwargs)
axes[2].set_title("Nominal Policy Rate")
axes[2].set_xlabel("Quarters ahead")
axes[2].set_ylabel("Nominal Rate")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="best")

plt.tight_layout()
st.pyplot(fig)  # Show the chart inside the Streamlit app

# =========================================
# Diagnostics (for the curious)
# =========================================
# This section prints detailed regression output (coefficients, R-squared, etc.)
# Non-technical users can ignore this, but it’s helpful for power users.
with st.expander("Model diagnostics (OLS summaries)"):
    st.write("**IS Curve**")
    st.text(models["model_is"].summary().as_text())
    st.write("**Phillips Curve**")
    st.text(models["model_pc"].summary().as_text())
    st.write("**Taylor Rule**")
    st.text(models["model_tr"].summary().as_text())
