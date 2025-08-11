# dsge_dashboard.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt

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
# Load & prep data
# =========================================
with st.sidebar:
    st.header("Data source")
    default_path = r"C:\Users\AC03537\OneDrive - Alberta Central\Desktop\DSGE.xlsx"
    xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"])
    file_path = xlf if xlf is not None else default_path
    st.caption("Sheets required: 'IS Curve', 'Phillips', 'Taylor' (Date format: YYYY-MM)")

@st.cache_data(show_spinner=True)
def load_and_prepare(file_like):
    is_df = pd.read_excel(file_like, sheet_name="IS Curve")
    pc_df = pd.read_excel(file_like, sheet_name="Phillips")
    tr_df = pd.read_excel(file_like, sheet_name="Taylor")

    # Parse dates
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")

    # Merge and index
    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Lags / drivers
    df["DlogGDP_L1"]       = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]      = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]  = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"]= (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP","DlogGDP_L1","Dlog_CPI","Dlog_CPI_L1",
        "Nominal Rate","Nominal_Rate_L1","Real_Rate_L2_data",
        "Dlog FD_Lag1","Dlog_REER","Dlog_Energy","Dlog_NonEnergy",
        "Dlog_Reer_L2","Dlog_Energy_L1","Dlog_Non_Energy_L1"
    ]
    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    return df, df_est

try:
    df_all, df_est = load_and_prepare(file_path)
except Exception as e:
    st.error(f"Problem loading data: {e}")
    st.stop()

# =========================================
# Estimate equations (OLS)
# =========================================
@st.cache_data(show_spinner=True)
def fit_models(df_est):
    # IS: DlogGDP_t ~ const + DlogGDP_{t-1} + real_rate_{t-2} + externals
    X_is = sm.add_constant(df_est[[
        "DlogGDP_L1","Real_Rate_L2_data","Dlog FD_Lag1","Dlog_REER","Dlog_Energy","Dlog_NonEnergy"
    ]])
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # Phillips: DlogCPI_t ~ const + DlogCPI_{t-1} + DlogGDP_{t-1} + externals
    X_pc = sm.add_constant(df_est[[
        "Dlog_CPI_L1","DlogGDP_L1","Dlog_Reer_L2","Dlog_Energy_L1","Dlog_Non_Energy_L1"
    ]])
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor: i_t ~ const + i_{t-1} + DlogCPI_t + DlogGDP_t
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1","Dlog_CPI","DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Convert to long-run form (to control persistence in simulation)
    b0   = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal_Rate_L1"]), 0.99)  # safety cap
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

# Baselines for simulation
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
# Sidebar controls
# =========================================
with st.sidebar:
    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)
    rho_sim = st.slider("Policy smoothing ρ (0 = fast, 0.95 = slow)", 0.0, 0.95, 0.25, 0.05)

    st.header("Shock")
    shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
    shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

    # Separate inputs so units are clear
    is_shock_size = st.number_input("IS shock size (Δ DlogGDP)", value=1.0, step=0.1, format="%.3f")
    pc_shock_size = st.number_input("Phillips shock size (Δ DlogCPI)", value=0.000, step=0.001, format="%.3f")

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

is_shock_arr, pc_shock_arr = build_shocks(T, shock_target, is_shock_size, pc_shock_size, shock_quarter, shock_persist)

# =========================================
# Simulation
# =========================================
def simulate(T, rho_sim, is_shock_arr=None, pc_shock_arr=None):
    g = np.zeros(T)  # DlogGDP
    p = np.zeros(T)  # DlogCPI
    i = np.zeros(T)  # Nominal rate

    # Start near steady state
    g[0] = float(df_est["DlogGDP"].mean())
    p[0] = float(df_est["Dlog_CPI"].mean())
    i[0] = i_neutral

    model_is = models["model_is"]
    model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]
    phi_pi_star = models["phi_pi_star"]
    phi_g_star  = models["phi_g_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        # Real rate with 2-quarter lag
        rr_lag2 = (i[t-2] - p[t-2]) if t >= 2 else real_rate_mean

        # IS
        Xis = pd.DataFrame([{
            "const": 1.0,
            "DlogGDP_L1": g[t-1],
            "Real_Rate_L2_data": rr_lag2,
            "Dlog FD_Lag1": means["Dlog FD_Lag1"],
            "Dlog_REER": means["Dlog_REER"],
            "Dlog_Energy": means["Dlog_Energy"],
            "Dlog_NonEnergy": means["Dlog_NonEnergy"],
        }])
        g[t] = model_is.predict(Xis).iloc[0] + is_shock_arr[t]

        # Phillips
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Dlog_CPI_L1": p[t-1],
            "DlogGDP_L1": g[t-1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }])
        p[t] = model_pc.predict(Xpc).iloc[0] + pc_shock_arr[t]

        # Taylor (partial adjustment)
        i_star = alpha_star + phi_pi_star * p[t] + phi_g_star * g[t]
        i[t]   = rho_sim * i[t-1] + (1 - rho_sim) * i_star

    return g, p, i

# Baseline vs scenario
g0, p0, i0 = simulate(T=T, rho_sim=rho_sim)  # no shocks
g1, p1, i1 = simulate(T=T, rho_sim=rho_sim, is_shock_arr=is_shock_arr, pc_shock_arr=pc_shock_arr)

# =========================================
# Plot (bigger fonts + clear titles)
# =========================================
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

quarters = np.arange(T)
vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

# GDP Growth
axes[0].plot(quarters, g0, label="Baseline", linewidth=2)
axes[0].plot(quarters, g1, label="Scenario", linewidth=2)
axes[0].axhline(float(df_est["DlogGDP"].mean()), ls="--", color="gray", label="Steady State")
axes[0].axvline(shock_quarter, **vline_kwargs)
axes[0].set_title("Real GDP Growth (DlogGDP)")
axes[0].set_ylabel("DlogGDP")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best")

# Inflation
axes[1].plot(quarters, p0, label="Baseline", linewidth=2)
axes[1].plot(quarters, p1, label="Scenario", linewidth=2)
axes[1].axhline(float(df_est["Dlog_CPI"].mean()), ls="--", color="gray", label="Steady State")
axes[1].axvline(shock_quarter, **vline_kwargs)
axes[1].set_title("Inflation Rate (DlogCPI)")
axes[1].set_ylabel("DlogCPI")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best")

# Policy Rate
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
st.pyplot(fig)

# =========================================
# Diagnostics
# =========================================
with st.expander("Model diagnostics (OLS summaries)"):
    st.write("**IS Curve**")
    st.text(models["model_is"].summary().as_text())
    st.write("**Phillips Curve**")
    st.text(models["model_pc"].summary().as_text())
    st.write("**Taylor Rule**")
    st.text(df_est[["Nominal Rate","Nominal_Rate_L1","Dlog_CPI","DlogGDP"]].head(1))  # quick sanity view