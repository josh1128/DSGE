# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app:
# 1) Loads DSGE.xlsx (IS, Phillips, Taylor), estimates simple OLS models,
#    and simulates impulse responses (GDP growth, inflation, policy rate).
# 2) Optionally loads test.xlsx and provides an interactive Plotly graph viewer
#    (General/IS/Phillips/Taylor tabs) with Raw/Trend/Cycle options.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objects as go

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
# Data source (GitHub-friendly)
# =========================================
with st.sidebar:
    st.header("Data source")
    xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"])
    st.caption("Sheets required: 'IS Curve', 'Phillips', 'Taylor' (Date format: YYYY-MM)")

    # Default to DSGE.xlsx in the same folder as this script (repo)
    local_fallback = Path(__file__).parent / "DSGE.xlsx"

# Extra uploader for test.xlsx (optional)
with st.sidebar:
    st.header("Additional data (optional)")
    test_file = st.file_uploader("Upload test.xlsx (optional)", type=["xlsx"], key="test_xlf")
    test_fallback = Path(__file__).parent / "test.xlsx"  # repo-local fallback

# =========================================
# Helper: HP filter
# =========================================
def apply_hp_filter(df, column, prefix=None, log_transform=False, exp_transform=False):
    """Adds {prefix}_Trend and {prefix}_Cycle columns using HP filter."""
    if df is None or df.empty or column not in df.columns:
        return df
    prefix = prefix or column
    series = df[column].replace(0, np.nan).dropna()
    if len(series) < 10:
        return df
    clean = np.log(series) if log_transform else series
    cycle, trend = hpfilter(clean, lamb=1600)
    if exp_transform:
        trend = np.exp(trend)
    df[f"{prefix}_Trend"] = trend.reindex(df.index)
    df[f"{prefix}_Cycle"] = cycle.reindex(df.index)
    return df

# =========================================
# Load & prep DSGE.xlsx
# =========================================
@st.cache_data(show_spinner=True)
def load_and_prepare(file_like_or_path):
    # Require a source
    if file_like_or_path is None:
        raise FileNotFoundError(
            "No file provided. Upload DSGE.xlsx or include DSGE.xlsx in the repo folder."
        )

    # Resolve path if a string/Path was provided
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        # Uploaded file-like object (BytesIO)
        excel_src = file_like_or_path

    # Read sheets
    is_df = pd.read_excel(excel_src, sheet_name="IS Curve")
    pc_df = pd.read_excel(excel_src, sheet_name="Phillips")
    tr_df = pd.read_excel(excel_src, sheet_name="Taylor")

    # Parse dates (YYYY-MM)
    for df in (is_df, pc_df, tr_df):
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m", errors="raise")

    # Merge and index
    df = (
        is_df.merge(pc_df, on="Date", how="inner")
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date")
    )

    # Lags / drivers
    df["DlogGDP_L1"]        = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]       = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]   = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

    required_cols = [
        "DlogGDP", "DlogGDP_L1", "Dlog_CPI", "Dlog_CPI_L1",
        "Nominal Rate", "Nominal_Rate_L1", "Real_Rate_L2_data",
        # IS externals (must match your sheet headers)
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

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    return df, df_est

# Decide source: uploaded file first, else repo-local fallback
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
    # IS: DlogGDP_t ~ const + DlogGDP_{t-1} + real_rate_{t-2} + externals
    X_is = sm.add_constant(df_est[[
        "DlogGDP_L1", "Real_Rate_L2_data", "Dlog FD_Lag1", "Dlog_REER", "Dlog_Energy", "Dlog_NonEnergy"
    ]])
    y_is = df_est["DlogGDP"]
    model_is = sm.OLS(y_is, X_is).fit()

    # Phillips: DlogCPI_t ~ const + DlogCPI_{t-1} + DlogGDP_{t-1} + externals (lagged)
    X_pc = sm.add_constant(df_est[[
        "Dlog_CPI_L1", "DlogGDP_L1", "Dlog_Reer_L2", "Dlog_Energy_L1", "Dlog_Non_Energy_L1"
    ]])
    y_pc = df_est["Dlog_CPI"]
    model_pc = sm.OLS(y_pc, X_pc).fit()

    # Taylor (partial adjustment basis): i_t ~ const + i_{t-1} + DlogCPI_t + DlogGDP_t
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1", "Dlog_CPI", "DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

    # Long-run coefficients for target i*
    b0   = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal_Rate_L1"]), 0.99)  # cap to avoid explosive smoothing
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

# =========================================
# Build shock arrays
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

    if is_shock_arr is None:
        is_shock_arr = np.zeros(T)
    if pc_shock_arr is None:
        pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        # Real rate with 2-quarter lag (fallback to sample mean early)
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean

        # IS
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

        # Phillips
        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }])
        p[t] = model_pc.predict(Xpc).iloc[0] + pc_shock_arr[t]

        # Taylor (partial adjustment)
        i_star = alpha_star + phi_pi_star * p[t] + phi_g_star * g[t]
        i[t]   = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return g, p, i

# Baseline vs scenario
g0, p0, i0 = simulate(T=T, rho_sim=rho_sim)  # no shocks
g1, p1, i1 = simulate(T=T, rho_sim=rho_sim, is_shock_arr=is_shock_arr, pc_shock_arr=pc_shock_arr)

# =========================================
# Plot (Matplotlib) — IRFs
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
# Load test.xlsx (optional) + Plotly Graph Viewer
# =========================================
@st.cache_data(show_spinner=True)
def load_test_dataset(file_like_or_path):
    # Resolve source (uploaded file or path)
    if file_like_or_path is None:
        return None
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            return None
        excel_src = p
    else:
        excel_src = file_like_or_path  # uploaded object

    # Expected sheets
    sheet_main     = "Potential Output"
    sheet_hours    = "Hours"
    sheet_is       = "IS Curve"
    sheet_phillips = "Phillips Curve"
    sheet_taylor   = "Taylor Rule"

    xls = pd.ExcelFile(excel_src)
    required = [sheet_main, sheet_hours, sheet_is, sheet_phillips, sheet_taylor]
    missing = [s for s in required if s not in xls.sheet_names]
    if missing:
        return None

    # Load
    df_main     = pd.read_excel(xls, sheet_name=sheet_main, na_values=["NA"])
    df_hours    = pd.read_excel(xls, sheet_name=sheet_hours, na_values=["NA"])
    df_is       = pd.read_excel(xls, sheet_name=sheet_is, na_values=["NA"])
    df_phillips = pd.read_excel(xls, sheet_name=sheet_phillips, na_values=["NA"])
    df_taylor   = pd.read_excel(xls, sheet_name=sheet_taylor, na_values=["NA"])

    # Normalize columns + dates
    for d in (df_main, df_hours, df_is, df_phillips, df_taylor):
        d.columns = d.columns.str.strip()
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], format="%Y-%m", errors="coerce").fillna(
                pd.to_datetime(d["Date"], errors="coerce")
            )
            d.dropna(subset=["Date"], inplace=True)
            d.sort_values("Date", inplace=True)
            d.reset_index(drop=True, inplace=True)

    # Merge hours -> main
    if "Average Hours Worked" in df_hours.columns and "Date" in df_hours.columns and "Date" in df_main.columns:
        df_main = pd.merge_asof(
            df_main.sort_values("Date"),
            df_hours.sort_values("Date"),
            on="Date",
            direction="backward"
        )

    # Construct helpful cols if present
    if {"Population","Labour Force Participation","NAIRU","Average Hours Worked","Real GDP Expenditure"}.issubset(df_main.columns):
        df_main["LFP_decimal"]   = df_main["Labour Force Participation"] / 100
        df_main["NAIRU_decimal"] = df_main["NAIRU"] / 100
        df_main["Total Hours Worked"] = (
            df_main["Population"]
            * df_main["LFP_decimal"]
            * (1 - df_main["NAIRU_decimal"])
            * df_main["Average Hours Worked"]
        )
        if "Labour Productivity" not in df_main.columns and "Total Hours Worked" in df_main.columns:
            with np.errstate(divide="ignore", invalid="ignore"):
                df_main["Labour Productivity"] = df_main["Real GDP Expenditure"] / df_main["Total Hours Worked"]

        # HP-filter some key series (raw trend/cycle)
        for col in ["Labour Force Participation","Labour Productivity","Average Hours Worked","Real GDP Expenditure"]:
            apply_hp_filter(df_main, col)

        # Potential Output proxy if not provided
        if "Potential Output" not in df_main.columns:
            if "Real GDP Expenditure_Trend" in df_main.columns:
                df_main["Potential Output"] = df_main["Real GDP Expenditure_Trend"]
            else:
                df_main["Potential Output"] = df_main["Real GDP Expenditure"]

        apply_hp_filter(df_main, "Potential Output", log_transform=True, exp_transform=True)

        with np.errstate(divide="ignore", invalid="ignore"):
            df_main["Output Gap (%)"] = (
                (df_main["Real GDP Expenditure"] - df_main["Potential Output"])
                / df_main["Potential Output"]
            ) * 100

    def with_year(df):
        if df is None or df.empty or "Date" not in df.columns:
            return df
        out = df.copy()
        out["Year"] = out["Date"].dt.year
        return out

    return {
        "main":     with_year(df_main),
        "is":       with_year(df_is),
        "phillips": with_year(df_phillips),
        "taylor":   with_year(df_taylor),
    }

# Decide test source: uploaded first, else repo fallback (or None)
test_source = test_file if test_file is not None else (test_fallback if test_fallback.exists() else None)
test_data = load_test_dataset(test_source)

# =========================================
# Plotly Graph Viewer (for test.xlsx)
# =========================================
st.markdown("---")
st.header("Test.xlsx — Graph Viewer")

def plot_selector(frame, default_var=None, title_prefix=""):
    if frame is None or frame.empty:
        st.info("No data available for this tab.")
        return

    numeric_cols = [c for c in frame.columns
                    if c not in {"Date","Year"}
                    and frame[c].dtype.kind in "biufc"
                    and not c.endswith("_Cycle")
                    and not c.endswith("_Trend")]

    if not numeric_cols:
        st.info("No numeric columns to plot.")
        return

    col1, col2 = st.columns([2,1])
    sorted_cols = sorted(numeric_cols)
    if default_var and default_var in sorted_cols:
        default_idx = sorted_cols.index(default_var)
    else:
        default_idx = 0

    with col1:
        var = st.selectbox("Variable", sorted_cols, index=default_idx)
    with col2:
        kind = st.radio("Data type", ["Raw","Trend","Cycle","Raw + Trend"], horizontal=True)

    years = frame["Year"]
    ymin, ymax = int(years.min()), int(years.max())
    yr = st.slider("Year range", min_value=ymin, max_value=ymax, value=(ymin, ymax))
    mask = (frame["Year"] >= yr[0]) & (frame["Year"] <= yr[1])
    f = frame.loc[mask].copy()

    fig = go.Figure()
    if kind in ("Raw","Raw + Trend"):
        fig.add_trace(go.Scatter(x=f["Date"], y=f[var], mode="lines", name=f"{var} — Raw"))
    if kind in ("Trend","Raw + Trend"):
        fig.add_trace(go.Scatter(x=f["Date"], y=f.get(f"{var}_Trend"), mode="lines", name=f"{var} — Trend"))
    if kind == "Cycle":
        fig.add_trace(go.Scatter(x=f["Date"], y=f.get(f"{var}_Cycle"), mode="lines", name=f"{var} — Cycle"))

    fig.update_layout(
        title={"text": f"{title_prefix}{var} ({yr[0]}–{yr[1]})", "x":0.5},
        xaxis_title="Date",
        yaxis_title=var,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40,r=40,t=60,b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

if test_data is not None and isinstance(test_data, dict) and test_data.get("main") is not None:
    tab1, tab2, tab3, tab4 = st.tabs(["General", "IS Curve", "Phillips Curve", "Taylor Rule"])

    with tab1:
        plot_selector(test_data["main"], default_var="Potential Output", title_prefix="General — ")
        # Optional tip
        f = test_data["main"]
        if f is not None and "Real GDP Expenditure" in f.columns and "Potential Output" in f.columns:
            st.caption("Tip: Choose **Potential Output**, then switch to **Raw + Trend** to compare with GDP trend.")

    with tab2:
        plot_selector(test_data["is"], title_prefix="IS Curve — ")

    with tab3:
        plot_selector(test_data["phillips"], title_prefix="Phillips — ")

    with tab4:
        plot_selector(test_data["taylor"], title_prefix="Taylor — ")
else:
    st.info("Upload **test.xlsx** or place it in the repo next to this script to enable the Graph Viewer.")

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
