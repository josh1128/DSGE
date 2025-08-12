# dsge_dashboard.py
# -----------------------------------------------------------
# Upload (or use local) DSGE_Model2.xlsx, estimate a simple
# NK 3-equation system (IS, Phillips, Taylor), and simulate
# IRFs to shocks. Plots: IRFs + historical lines.
# Target inflation is fixed at 2%.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# =========================================
# Config
# =========================================
TARGET_PI = 2.0  # percent

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="DSGE Model Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — IS (Output Gap), Phillips (Inflation), Taylor (Policy)")

st.markdown(
    "Upload **DSGE_Model2.xlsx** or place it beside this script. "
    "We estimate an NK-style 3-equation system and simulate responses to shocks. "
    "Missing values are linearly interpolated and edge-filled. "
    "**Inflation target is fixed at 2%.**"
)

# =========================================
# Data source
# =========================================
with st.sidebar:
    st.header("Data source")
    xlf = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"])
    st.caption(
        "Sheets & columns expected:\n"
        "• IS Curve: Date, Output Gap, Nominal Interest Rate, Inflation Rate, Foreign Demand, Non-Energy, Energy, REER\n"
        "• Phillips: Date, Inflation Rate, Output Gap, Foreign Demand, Non-Energy, Energy, REER\n"
        "• Taylor: Date, Nominal Interest Rate, Inflation Gap, Output Gap"
    )
    local_fallback = Path(__file__).parent / "DSGE_Model2.xlsx"

def _read_sheets_flex(excel_src):
    """
    Try to read by expected sheet names; if not present, read first three sheets.
    Returns (is_df, pc_df, tr_df).
    """
    xl = pd.ExcelFile(excel_src)

    def find(name):
        for s in xl.sheet_names:
            if s.strip().lower() == name.strip().lower():
                return s
        return None

    name_is = find("IS Curve")
    name_pc = find("Phillips")
    name_tr = find("Taylor")

    if all([name_is, name_pc, name_tr]):
        is_df = pd.read_excel(excel_src, sheet_name=name_is)
        pc_df = pd.read_excel(excel_src, sheet_name=name_pc)
        tr_df = pd.read_excel(excel_src, sheet_name=name_tr)
    else:
        sheet_list = xl.sheet_names[:3]
        is_df = pd.read_excel(excel_src, sheet_name=sheet_list[0])
        pc_df = pd.read_excel(excel_src, sheet_name=sheet_list[1])
        tr_df = pd.read_excel(excel_src, sheet_name=sheet_list[2])
    return is_df, pc_df, tr_df

@st.cache_data(show_spinner=True)
def load_and_prepare(file_like_or_path):
    """
    Loads Excel, reads 3 sheets, harmonizes dates, merges, interpolates,
    and constructs variables/lagged terms needed for estimation/simulation.
    Returns (df_all, df_est).
    """
    if file_like_or_path is None:
        raise FileNotFoundError("No file provided. Upload DSGE_Model2.xlsx or place it beside this script.")

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
    is_df, pc_df, tr_df = _read_sheets_flex(excel_src)

    # Parse Date (YYYY-MM or YYYY-Qx) robustly
    def parse_date(col):
        try:
            return pd.to_datetime(col, format="%Y-%m")
        except Exception:
            try:
                # Try generic parse; if like 1999Q1, coerce via PeriodIndex
                return pd.PeriodIndex(col.astype(str), freq="Q").to_timestamp(how="end")
            except Exception:
                return pd.to_datetime(col, errors="coerce")

    for df in (is_df, pc_df, tr_df):
        if "Date" not in df.columns:
            raise KeyError("Each sheet must contain a 'Date' column.")
        df["Date"] = parse_date(df["Date"])

    # Keep only the needed columns per sheet
    is_cols = ["Date", "Output Gap", "Nominal Interest Rate", "Inflation Rate", "Foreign Demand", "Non-Energy", "Energy", "REER"]
    pc_cols = ["Date", "Inflation Rate", "Output Gap", "Foreign Demand", "Non-Energy", "Energy", "REER"]
    tr_cols = ["Date", "Nominal Interest Rate", "Inflation Gap", "Output Gap"]

    for need, df, label in [
        (is_cols, is_df, "IS Curve"),
        (pc_cols, pc_df, "Phillips"),
        (tr_cols, tr_df, "Taylor"),
    ]:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns on '{label}' sheet: {missing}")

    is_df = is_df[is_cols].copy()
    pc_df = pc_df[pc_cols].copy()
    tr_df = tr_df[tr_cols].copy()

    # Merge on Date (inner join to keep aligned samples)
    df = (
        is_df.merge(pc_df, on="Date", how="inner", suffixes=("_IS", "_PC"))
             .merge(tr_df, on="Date", how="inner")
    )

    # Tidy names, prefer IS-sheet externals
    df = df.sort_values("Date").set_index("Date")
    df = df.rename(columns={
        "Output Gap_IS": "Output Gap",
        "Inflation Rate_IS": "Inflation Rate",
        "Nominal Interest Rate": "Nominal Rate",
        "Foreign Demand_IS": "Foreign Demand",
        "Non-Energy_IS": "Non-Energy",
        "Energy_IS": "Energy",
        "REER_IS": "REER",
        "Inflation Gap": "Inflation Gap"  # will be ignored in estimation; kept for reference
    })
    drop_dupes = [c for c in ["Output Gap_PC", "Inflation Rate_PC", "Foreign Demand_PC",
                              "Non-Energy_PC", "Energy_PC", "REER_PC"] if c in df.columns]
    df = df.drop(columns=drop_dupes)

    # Interpolate/extrapolate missing values; finalize with ffill/bfill
    df_interp = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # Helper variables & lags
    df_interp["Real Rate"] = df_interp["Nominal Rate"] - df_interp["Inflation Rate"]
    df_interp["Output Gap_L1"] = df_interp["Output Gap"].shift(1)
    df_interp["Inflation Rate_L1"] = df_interp["Inflation Rate"].shift(1)
    df_interp["Nominal Rate_L1"] = df_interp["Nominal Rate"].shift(1)
    df_interp["Real Rate_L1"] = df_interp["Real Rate"].shift(1)

    # For estimation sets (drop initial lag NAs)
    req_for_est = [
        "Output Gap", "Output Gap_L1", "Real Rate_L1",
        "Inflation Rate", "Inflation Rate_L1",
        "Nominal Rate", "Nominal Rate_L1",
        "Foreign Demand", "Non-Energy", "Energy", "REER",
    ]
    df_est = df_interp.dropna(subset=req_for_est).copy()
    if df_est.empty:
        raise ValueError("After interpolation, no rows remain for estimation — check your data coverage.")

    return df_interp, df_est

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
    # --- IS: Output Gap_t on lagged gap, lagged real rate, externals (L1 to reduce simultaneity)
    X_is = sm.add_constant(pd.DataFrame({
        "Output Gap_L1": df_est["Output Gap_L1"],
        "Real Rate_L1": df_est["Real Rate_L1"],
        "Foreign Demand_L1": df_est["Foreign Demand"].shift(1),
        "NonEnergy_L1": df_est["Non-Energy"].shift(1),
        "Energy_L1": df_est["Energy"].shift(1),
        "REER_L1": df_est["REER"].shift(1),
    }))
    y_is = df_est["Output Gap"]
    is_ok = X_is.dropna().index.intersection(y_is.dropna().index)
    model_is = sm.OLS(y_is.loc[is_ok], X_is.loc[is_ok]).fit()

    # --- Phillips: Inflation_t on lagged inflation, lagged output gap, externals (L1)
    X_pc = sm.add_constant(pd.DataFrame({
        "Inflation Rate_L1": df_est["Inflation Rate_L1"],
        "Output Gap_L1": df_est["Output Gap_L1"],
        "Foreign Demand_L1": df_est["Foreign Demand"].shift(1),
        "NonEnergy_L1": df_est["Non-Energy"].shift(1),
        "Energy_L1": df_est["Energy"].shift(1),
        "REER_L1": df_est["REER"].shift(1),
    }))
    y_pc = df_est["Inflation Rate"]
    pc_ok = X_pc.dropna().index.intersection(y_pc.dropna().index)
    model_pc = sm.OLS(y_pc.loc[pc_ok], X_pc.loc[pc_ok]).fit()

    # --- Taylor (partial adjustment): i_t on i_{t-1} + Inflation Gap_t + Output Gap_t
    # Inflation Gap is enforced as (Inflation Rate - 2.0)
    infl_gap = df_est["Inflation Rate"] - TARGET_PI

    X_tr = sm.add_constant(pd.DataFrame({
        "Nominal Rate_L1": df_est["Nominal Rate_L1"],
        "Inflation Gap": infl_gap,
        "Output Gap": df_est["Output Gap"],
    }))
    y_tr = df_est["Nominal Rate"]
    tr_ok = X_tr.dropna().index.intersection(y_tr.dropna().index)
    model_tr = sm.OLS(y_tr.loc[tr_ok], X_tr.loc[tr_ok]).fit()

    # Convert to long-run targets for policy rule i* = α* + φπ*π_gap + φy*y
    b0   = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal Rate_L1"]), 0.99)  # cap persistence
    bpi  = float(model_tr.params["Inflation Gap"])
    by   = float(model_tr.params["Output Gap"])

    alpha_star  = b0  / (1 - rhoh)
    phi_pi_star = bpi / (1 - rhoh)
    phi_y_star  = by  / (1 - rhoh)

    return {
        "model_is": model_is,
        "model_pc": model_pc,
        "model_tr": model_tr,
        "alpha_star": alpha_star,
        "phi_pi_star": phi_pi_star,
        "phi_y_star": phi_y_star,
        "rhoh": rhoh
    }

models = fit_models(df_est)

# Means for externals and anchors
means = {
    "Foreign Demand": float(df_est["Foreign Demand"].mean()),
    "Non-Energy": float(df_est["Non-Energy"].mean()),
    "Energy": float(df_est["Energy"].mean()),
    "REER": float(df_est["REER"].mean()),
}
i_neutral = float(df_est["Nominal Rate"].mean())
y_gap_mean = float(df_est["Output Gap"].mean())
pi_mean = float(df_est["Inflation Rate"].mean())

# =========================================
# Sidebar: simulation controls
# =========================================
with st.sidebar:
    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)
    rho_sim = st.slider("Policy smoothing ρ (0 = fast, 0.95 = slow)", 0.0, 0.95, 0.25, 0.05)

    st.header("Shock")
    shock_target = st.selectbox("Apply shock to", ["None", "IS (Output Gap)", "Phillips (Inflation)"], index=0)
    shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)
    is_shock_size = st.number_input("IS shock size (Δ Output Gap, pp)", value=0.5, step=0.1, format="%.2f")
    pc_shock_size = st.number_input("Phillips shock size (Δ Inflation, pp)", value=0.10, step=0.05, format="%.2f")

# =========================================
# Shocks
# =========================================
def build_shocks(T, target, is_size, pc_size, t0, rho):
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Output Gap)":
        is_arr[t0] = is_size
        for k in range(t0 + 1, T):
            is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Inflation)":
        pc_arr[t0] = pc_size
        for k in range(t0 + 1, T):
            pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

is_shock_arr, pc_shock_arr = build_shocks(T, shock_target, is_shock_size, pc_shock_size, shock_quarter, shock_persist)

# =========================================
# Simulation engine
# =========================================
def simulate(T, rho_sim, is_shock_arr=None, pc_shock_arr=None):
    """
    y[t]: Output Gap (pp)
    p[t]: Inflation Rate (pp)
    i[t]: Nominal Rate (level, %)
    """
    y = np.zeros(T)
    p = np.zeros(T)
    i = np.zeros(T)

    # Start from means
    y[0] = y_gap_mean
    p[0] = pi_mean
    i[0] = i_neutral

    model_is = models["model_is"]
    model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]
    phi_pi_star = models["phi_pi_star"]
    phi_y_star  = models["phi_y_star"]

    if is_shock_arr is None:
        is_shock_arr = np.zeros(T)
    if pc_shock_arr is None:
        pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        # lagged real rate from last period
        rr_l1 = (i[t - 1] - p[t - 1])

        # IS block
        Xis = pd.DataFrame([{
            "const": 1.0,
            "Output Gap_L1": y[t - 1],
            "Real Rate_L1": rr_l1,
            "Foreign Demand_L1": means["Foreign Demand"],
            "NonEnergy_L1": means["Non-Energy"],
            "Energy_L1": means["Energy"],
            "REER_L1": means["REER"],
        }])
        y[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        # Phillips block
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Inflation Rate_L1": p[t - 1],
            "Output Gap_L1": y[t - 1],
            "Foreign Demand_L1": means["Foreign Demand"],
            "NonEnergy_L1": means["Non-Energy"],
            "Energy_L1": means["Energy"],
            "REER_L1": means["REER"],
        }])
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        # Taylor rule with partial adjustment:
        # Inflation gap = p[t] - TARGET_PI
        pi_gap_t = p[t] - TARGET_PI
        i_star = alpha_star + phi_pi_star * pi_gap_t + phi_y_star * y[t]
        i[t] = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return y, p, i

# Run: baseline vs scenario
y0, p0, i0 = simulate(T=T, rho_sim=rho_sim)
y1, p1, i1 = simulate(T=T, rho_sim=rho_sim, is_shock_arr=is_shock_arr, pc_shock_arr=pc_shock_arr)

# =========================================
# Plot IRFs (three-panel)
# =========================================
plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
quarters = np.arange(T)
vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

# Output Gap
axes[0].plot(quarters, y0, label="Baseline", linewidth=2)
axes[0].plot(quarters, y1, label="Scenario", linewidth=2)
axes[0].axhline(y_gap_mean, ls="--", label="Steady State")
axes[0].axvline(shock_quarter, **vline_kwargs)
axes[0].set_title("Output Gap (pp)")
axes[0].set_ylabel("pp")
axes[0].grid(True, alpha=0.3)
axes[0].legend(loc="best")

# Inflation Rate
axes[1].plot(quarters, p0, label="Baseline", linewidth=2)
axes[1].plot(quarters, p1, label="Scenario", linewidth=2)
axes[1].axhline(pi_mean, ls="--", label="Steady State")
axes[1].axvline(shock_quarter, **vline_kwargs)
axes[1].set_title("Inflation Rate (pp)")
axes[1].set_ylabel("pp")
axes[1].grid(True, alpha=0.3)
axes[1].legend(loc="best")

# Nominal Policy Rate (level)
axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
axes[2].plot(quarters, i1, label="Scenario", linewidth=2)
axes[2].axhline(i_neutral, ls="--", label="Neutral Rate")
axes[2].axvline(shock_quarter, **vline_kwargs)
axes[2].set_title("Nominal Policy Rate (%)")
axes[2].set_xlabel("Quarters ahead")
axes[2].set_ylabel("%")
axes[2].grid(True, alpha=0.3)
axes[2].legend(loc="best")

plt.tight_layout()
st.pyplot(fig)

# =========================================
# Historical time series (Nominal Rate shown as growth %)
# =========================================
df_plot = df_all.reset_index().copy()

# Nominal interest rate growth (%): 100 * log change (guard against zeros/negatives)
rate = df_plot["Nominal Rate"].astype(float)
rate_growth = pd.Series(np.nan, index=rate.index)
valid = (rate > 0) & (rate.shift(1) > 0)
rate_growth.loc[valid] = 100 * np.log(rate.loc[valid] / rate.shift(1).loc[valid])

fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.plot(df_plot["Date"], df_plot["Output Gap"], linewidth=2, label="Output Gap (pp)")
ax3.plot(df_plot["Date"], df_plot["Inflation Rate"], linewidth=2, label="Inflation Rate (pp)")
ax3.plot(df_plot["Date"], rate_growth, linewidth=2, label="Nominal Rate growth (%)")

ax3.set_title("Historical: Output Gap, Inflation, and Nominal Rate Growth")
ax3.set_xlabel("Date")
ax3.set_ylabel("pp / %")
ax3.grid(True, alpha=0.3)
ax3.legend()

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

