# dsge_dashboard.py
# -----------------------------------------------------------
# One Streamlit app that can run:
#   1) Original model (DSGE.xlsx): IS/DlogGDP, Phillips/DlogCPI, Taylor
#   2) NK model (DSGE_Model2.xlsx): Output Gap, Inflation, Taylor with pi* = 2%
#
# For Model2, headers are cleaned (spacing/case), and missing values are
# interpolated/extrapolated (linear + edge fill) as requested.
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================
# Global config
# =========================================
TARGET_PI = 2.0  # inflation target for NK model (percent)

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="DSGE Model Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — IS, Phillips, Taylor")

st.markdown(
    "Use the sidebar to select a model version and upload the matching Excel file.\n\n"
    "- **Original model (DSGE.xlsx)**: DlogGDP, Dlog_CPI, Nominal Rate, etc.\n"
    "- **NK model (DSGE_Model2.xlsx)**: Output Gap, Inflation Rate, Nominal Rate, externals; "
    "headers are spacing/case-robust and NAs are interpolated/extrapolated.\n"
    "Policy responds via a partial-adjustment Taylor rule."
)

# =========================================
# Helpers (Model2: robust header cleaning)
# =========================================
def _normalize_name(s: str) -> str:
    return pd.Series([s]).str.strip().str.replace(r"\s+", " ", regex=True).str.lower().iloc[0]

CANONICAL_MAP = {
    "date": "Date",
    "output gap": "Output Gap",
    "nominal interest rate": "Nominal Rate",
    "nominal rate": "Nominal Rate",
    "inflation rate": "Inflation Rate",
    "inflation gap": "Inflation Gap",
    "foreign demand": "Foreign Demand",
    "non-energy": "Non-Energy",
    "non energy": "Non-Energy",
    "energy": "Energy",
    "reer": "REER",
}

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = []
    for c in df.columns:
        norm = _normalize_name(str(c))
        cleaned.append(CANONICAL_MAP.get(norm, pd.Series([c]).str.strip().str.replace(r"\s+", " ", regex=True).iloc[0]))
    df = df.copy()
    df.columns = cleaned
    return df

# =========================================
# Sidebar: choose model + upload file + controls
# =========================================
with st.sidebar:
    st.header("Model selection")
    model_choice = st.selectbox(
        "Choose model version",
        ["Original (DSGE.xlsx)", "NK (DSGE_Model2.xlsx)"],
        index=0
    )

    if "Original" in model_choice:
        xlf = st.file_uploader("Upload DSGE.xlsx (optional)", type=["xlsx"], key="upload_original")
        fallback = Path(__file__).parent / "DSGE.xlsx"
    else:
        xlf = st.file_uploader("Upload DSGE_Model2.xlsx (optional)", type=["xlsx"], key="upload_nk")
        fallback = Path(__file__).parent / "DSGE_Model2.xlsx"

    st.header("Simulation settings")
    T = st.slider("Horizon (quarters)", min_value=8, max_value=60, value=20, step=1)
    rho_sim = st.slider("Policy smoothing ρ (0 = fast, 0.95 = slow)", 0.0, 0.95, 0.25, 0.05)

    st.header("Shock")
    if "Original" in model_choice:
        shock_target = st.selectbox("Apply shock to", ["None", "IS (Demand)", "Phillips (Supply)"], index=0)
        is_shock_label = "IS shock size (Δ DlogGDP)"
        pc_shock_label = "Phillips shock size (Δ DlogCPI)"
        is_shock_size = st.number_input(is_shock_label, value=1.0, step=0.1, format="%.3f")
        pc_shock_size = st.number_input(pc_shock_label, value=0.000, step=0.001, format="%.3f")
    else:
        shock_target = st.selectbox("Apply shock to", ["None", "IS (Output Gap)", "Phillips (Inflation)"], index=0)
        is_shock_label = "IS shock size (Δ Output Gap, pp)"
        pc_shock_label = "Phillips shock size (Δ Inflation, pp)"
        is_shock_size = st.number_input(is_shock_label, value=0.5, step=0.1, format="%.2f")
        pc_shock_size = st.number_input(pc_shock_label, value=0.10, step=0.05, format="%.2f")

    shock_quarter = st.slider("Shock timing (t)", min_value=1, max_value=T-1, value=1, step=1)
    shock_persist = st.slider("Shock persistence ρ_shock", 0.0, 0.95, 0.0, 0.05)

# Decide file source
file_source = xlf if xlf is not None else fallback

# =========================================
# ORIGINAL MODEL (DSGE.xlsx)
# =========================================
@st.cache_data(show_spinner=True)
def load_and_prepare_original(file_like_or_path):
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

    # Lags / transforms
    df["DlogGDP_L1"]        = df["DlogGDP"].shift(1)
    df["Dlog_CPI_L1"]       = df["Dlog_CPI"].shift(1)
    df["Nominal_Rate_L1"]   = df["Nominal Rate"].shift(1)
    df["Real_Rate_L2_data"] = (df["Nominal Rate"] - df["Dlog_CPI"]).shift(2)

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
        raise KeyError(f"Missing required columns: {missing}. Adjust required_cols/X matrices to match Excel headers.")

    df_est = df.dropna(subset=required_cols).copy()
    if df_est.empty:
        raise ValueError("No rows remain after dropping NA for required columns. Check your data.")

    return df, df_est

@st.cache_data(show_spinner=True)
def fit_models_original(df_est):
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

    # Taylor (partial adjustment)
    X_tr = sm.add_constant(df_est[["Nominal_Rate_L1", "Dlog_CPI", "DlogGDP"]])
    y_tr = df_est["Nominal Rate"]
    model_tr = sm.OLS(y_tr, X_tr).fit()

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

def build_shocks_original(T, target, is_size, pc_size, t0, rho):
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Demand)":
        is_arr[t0] = is_size
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Supply)":
        pc_arr[t0] = pc_size
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

def simulate_original(T, rho_sim, df_est, models, means, i_neutral, real_rate_mean, is_shock_arr=None, pc_shock_arr=None):
    g = np.zeros(T)  # DlogGDP
    p = np.zeros(T)  # DlogCPI
    i = np.zeros(T)  # Nominal rate (level)

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
        rr_lag2 = (i[t - 2] - p[t - 2]) if t >= 2 else real_rate_mean

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

        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Dlog_CPI_L1": p[t - 1],
            "DlogGDP_L1": g[t - 1],
            "Dlog_Reer_L2": means["Dlog_Reer_L2"],
            "Dlog_Energy_L1": means["Dlog_Energy_L1"],
            "Dlog_Non_Energy_L1": means["Dlog_Non_Energy_L1"],
        }])
        p[t] = model_pc.predict(Xpc).iloc[0] + pc_shock_arr[t]

        i_star = alpha_star + phi_pi_star * p[t] + phi_g_star * g[t]
        i[t]   = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return g, p, i

# =========================================
# NK MODEL (DSGE_Model2.xlsx)
# =========================================
def _read_sheets_flex(excel_src):
    xl = pd.ExcelFile(excel_src)
    def _find(name):
        for s in xl.sheet_names:
            if s.strip().lower() == name.strip().lower():
                return s
        return None
    name_is = _find("IS Curve") or xl.sheet_names[0]
    name_pc = _find("Phillips") or xl.sheet_names[1]
    name_tr = _find("Taylor") or xl.sheet_names[2]
    is_df = standardize_columns(pd.read_excel(excel_src, sheet_name=name_is))
    pc_df = standardize_columns(pd.read_excel(excel_src, sheet_name=name_pc))
    tr_df = standardize_columns(pd.read_excel(excel_src, sheet_name=name_tr))
    return is_df, pc_df, tr_df

@st.cache_data(show_spinner=True)
def load_and_prepare_model2(file_like_or_path):
    if file_like_or_path is None:
        raise FileNotFoundError("No file provided. Upload DSGE_Model2.xlsx or include it beside this script.")
    if isinstance(file_like_or_path, (str, Path)):
        p = Path(file_like_or_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(f"Could not find Excel file at: {p}")
        excel_src = p
    else:
        excel_src = file_like_or_path

    is_df, pc_df, tr_df = _read_sheets_flex(excel_src)

    def parse_date(series):
        out = pd.to_datetime(series, format="%Y-%m", errors="coerce")
        if out.isna().all():
            try:
                return pd.PeriodIndex(series.astype(str), freq="Q").to_timestamp(how="end")
            except Exception:
                return pd.to_datetime(series, errors="coerce")
        return out

    for df in (is_df, pc_df, tr_df):
        if "Date" not in df.columns:
            raise KeyError("Each sheet must contain a 'Date' column.")
        df["Date"] = parse_date(df["Date"])

    is_cols = ["Date", "Output Gap", "Nominal Rate", "Inflation Rate", "Foreign Demand", "Non-Energy", "Energy", "REER"]
    pc_cols = ["Date", "Inflation Rate", "Output Gap", "Foreign Demand", "Non-Energy", "Energy", "REER"]
    tr_cols = ["Date", "Nominal Rate", "Inflation Gap", "Output Gap"]

    for need, df, label in [(is_cols, is_df, "IS Curve"), (pc_cols, pc_df, "Phillips"), (tr_cols, tr_df, "Taylor")]:
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns on '{label}' sheet: {missing}")

    tr_df = tr_df.rename(columns={"Nominal Rate": "Nominal Rate_Taylor", "Output Gap": "Output Gap_Taylor"})

    merged = (
        is_df.merge(pc_df, on="Date", how="inner", suffixes=("_IS", "_PC"))
             .merge(tr_df, on="Date", how="inner")
             .sort_values("Date").set_index("Date")
    )

    df = pd.DataFrame(index=merged.index)
    df["Output Gap"]     = merged["Output Gap_IS"]
    df["Inflation Rate"] = merged["Inflation Rate_IS"]
    df["Nominal Rate"]   = merged["Nominal Rate_Taylor"] if "Nominal Rate_Taylor" in merged else merged["Nominal Rate_IS"]
    df["Foreign Demand"] = merged["Foreign Demand_IS"]
    df["Non-Energy"]     = merged["Non-Energy_IS"]
    df["Energy"]         = merged["Energy_IS"]
    df["REER"]           = merged["REER_IS"]
    if "Inflation Gap" in merged.columns:
        df["Inflation Gap"] = merged["Inflation Gap"]  # informational only

    # Interpolate/extrapolate missing values as requested
    df_interp = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    # Lags & helpers
    df_interp["Real Rate"]            = df_interp["Nominal Rate"] - df_interp["Inflation Rate"]
    df_interp["Output Gap_L1"]        = df_interp["Output Gap"].shift(1)
    df_interp["Inflation Rate_L1"]    = df_interp["Inflation Rate"].shift(1)
    df_interp["Nominal Rate_L1"]      = df_interp["Nominal Rate"].shift(1)
    df_interp["Real Rate_L1"]         = df_interp["Real Rate"].shift(1)

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

@st.cache_data(show_spinner=True)
def fit_models_model2(df_est):
    # IS: y_t on y_{t-1}, real rate_{t-1}, externals (L1)
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

    # Phillips: pi_t on pi_{t-1}, y_{t-1}, externals (L1)
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

    # Taylor (partial adjustment): i_t on i_{t-1}, (pi_t - 2), y_t
    infl_gap = df_est["Inflation Rate"] - TARGET_PI
    X_tr = sm.add_constant(pd.DataFrame({
        "Nominal Rate_L1": df_est["Nominal Rate_L1"],
        "Inflation Gap": infl_gap,
        "Output Gap": df_est["Output Gap"],
    }))
    y_tr = df_est["Nominal Rate"]
    tr_ok = X_tr.dropna().index.intersection(y_tr.dropna().index)
    model_tr = sm.OLS(y_tr.loc[tr_ok], X_tr.loc[tr_ok]).fit()

    b0   = float(model_tr.params["const"])
    rhoh = min(float(model_tr.params["Nominal Rate_L1"]), 0.99)
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

def build_shocks_model2(T, target, is_size, pc_size, t0, rho):
    is_arr = np.zeros(T)
    pc_arr = np.zeros(T)
    if target == "IS (Output Gap)":
        is_arr[t0] = is_size
        for k in range(t0 + 1, T): is_arr[k] = rho * is_arr[k - 1]
    elif target == "Phillips (Inflation)":
        pc_arr[t0] = pc_size
        for k in range(t0 + 1, T): pc_arr[k] = rho * pc_arr[k - 1]
    return is_arr, pc_arr

def simulate_model2(T, rho_sim, df_est, models, means2, anchors, is_shock_arr=None, pc_shock_arr=None):
    """
    y[t]: Output Gap (pp)
    p[t]: Inflation Rate (pp)
    i[t]: Nominal Rate (level, %)
    """
    y = np.zeros(T); p = np.zeros(T); i = np.zeros(T)
    y[0], p[0], i[0] = anchors["y_gap_mean"], anchors["pi_mean"], anchors["i_neutral"]

    model_is = models["model_is"]; model_pc = models["model_pc"]
    alpha_star  = models["alpha_star"]; phi_pi_star = models["phi_pi_star"]; phi_y_star = models["phi_y_star"]

    if is_shock_arr is None: is_shock_arr = np.zeros(T)
    if pc_shock_arr is None: pc_shock_arr = np.zeros(T)

    for t in range(1, T):
        rr_l1 = (i[t - 1] - p[t - 1])

        Xis = pd.DataFrame([{
            "const": 1.0,
            "Output Gap_L1": y[t - 1],
            "Real Rate_L1": rr_l1,
            "Foreign Demand_L1": means2["Foreign Demand"],
            "NonEnergy_L1": means2["Non-Energy"],
            "Energy_L1": means2["Energy"],
            "REER_L1": means2["REER"],
        }])
        y[t] = float(model_is.predict(Xis).iloc[0]) + is_shock_arr[t]

        Xpc = pd.DataFrame([{
            "const": 1.0,
            "Inflation Rate_L1": p[t - 1],
            "Output Gap_L1": y[t - 1],
            "Foreign Demand_L1": means2["Foreign Demand"],
            "NonEnergy_L1": means2["Non-Energy"],
            "Energy_L1": means2["Energy"],
            "REER_L1": means2["REER"],
        }])
        p[t] = float(model_pc.predict(Xpc).iloc[0]) + pc_shock_arr[t]

        pi_gap_t = p[t] - TARGET_PI
        i_star = alpha_star + phi_pi_star * pi_gap_t + phi_y_star * y[t]
        i[t] = rho_sim * i[t - 1] + (1 - rho_sim) * i_star

    return y, p, i

# =========================================
# Run the chosen model
# =========================================
try:
    if "Original" in model_choice:
        # Load
        df_all, df_est = load_and_prepare_original(file_source)

        # Fit
        models = fit_models_original(df_est)

        # Anchors & means
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

        # Shocks
        is_arr, pc_arr = build_shocks_original(T, shock_target, is_shock_size, pc_shock_size, shock_quarter, shock_persist)

        # Simulate
        g0, p0, i0 = simulate_original(T, rho_sim, df_est, models, means, i_neutral, real_rate_mean)
        g1, p1, i1 = simulate_original(T, rho_sim, df_est, models, means, i_neutral, real_rate_mean, is_arr, pc_arr)

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        quarters = np.arange(T); vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(quarters, g0, label="Baseline", linewidth=2)
        axes[0].plot(quarters, g1, label="Scenario", linewidth=2)
        axes[0].axhline(float(df_est["DlogGDP"].mean()), ls="--", color="gray", label="Steady State")
        axes[0].axvline(shock_quarter, **vline_kwargs)
        axes[0].set_title("Real GDP Growth (DlogGDP)"); axes[0].set_ylabel("DlogGDP"); axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(quarters, p0, label="Baseline", linewidth=2)
        axes[1].plot(quarters, p1, label="Scenario", linewidth=2)
        axes[1].axhline(float(df_est["Dlog_CPI"].mean()), ls="--", color="gray", label="Steady State")
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation Rate (DlogCPI)"); axes[1].set_ylabel("DlogCPI"); axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, i1, label="Scenario", linewidth=2)
        axes[2].axhline(i_neutral, ls="--", color="gray", label="Neutral Rate")
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate"); axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("Nominal Rate")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Diagnostics
        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(models["model_is"].summary().as_text())
            st.write("**Phillips Curve**"); st.text(models["model_pc"].summary().as_text())
            st.write("**Taylor Rule**"); st.text(models["model_tr"].summary().as_text())

    else:
        # Load
        df_all, df_est = load_and_prepare_model2(file_source)

        # Fit
        models2 = fit_models_model2(df_est)

        # Anchors & means
        means2 = {
            "Foreign Demand": float(df_est["Foreign Demand"].mean()),
            "Non-Energy": float(df_est["Non-Energy"].mean()),
            "Energy": float(df_est["Energy"].mean()),
            "REER": float(df_est["REER"].mean()),
        }
        i_neutral2 = float(df_est["Nominal Rate"].mean())
        y_gap_mean = float(df_est["Output Gap"].mean())
        pi_mean    = float(df_est["Inflation Rate"].mean())
        anchors = {"i_neutral": i_neutral2, "y_gap_mean": y_gap_mean, "pi_mean": pi_mean}

        # Shocks
        is_arr2, pc_arr2 = build_shocks_model2(T, shock_target, is_shock_size, pc_shock_size, shock_quarter, shock_persist)

        # Simulate
        y0, p0, i0 = simulate_model2(T, rho_sim, df_est, models2, means2, anchors)
        y1, p1, i1 = simulate_model2(T, rho_sim, df_est, models2, means2, anchors, is_arr2, pc_arr2)

        # Plot IRFs
        plt.rcParams.update({"axes.titlesize": 16, "axes.labelsize": 12, "legend.fontsize": 11})
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        quarters = np.arange(T); vline_kwargs = dict(color="black", linestyle=":", linewidth=1)

        axes[0].plot(quarters, y0, label="Baseline", linewidth=2)
        axes[0].plot(quarters, y1, label="Scenario", linewidth=2)
        axes[0].axhline(y_gap_mean, ls="--", color="gray", label="Steady State")
        axes[0].axvline(shock_quarter, **vline_kwargs)
        axes[0].set_title("Output Gap (pp)"); axes[0].set_ylabel("pp"); axes[0].grid(True, alpha=0.3); axes[0].legend(loc="best")

        axes[1].plot(quarters, p0, label="Baseline", linewidth=2)
        axes[1].plot(quarters, p1, label="Scenario", linewidth=2)
        axes[1].axhline(pi_mean, ls="--", color="gray", label="Steady State")
        axes[1].axvline(shock_quarter, **vline_kwargs)
        axes[1].set_title("Inflation Rate (pp)"); axes[1].set_ylabel("pp"); axes[1].grid(True, alpha=0.3); axes[1].legend(loc="best")

        axes[2].plot(quarters, i0, label="Baseline", linewidth=2)
        axes[2].plot(quarters, i1, label="Scenario", linewidth=2)
        axes[2].axhline(i_neutral2, ls="--", color="gray", label="Neutral Rate")
        axes[2].axvline(shock_quarter, **vline_kwargs)
        axes[2].set_title("Nominal Policy Rate (%)"); axes[2].set_xlabel("Quarters ahead"); axes[2].set_ylabel("%")
        axes[2].grid(True, alpha=0.3); axes[2].legend(loc="best")

        plt.tight_layout(); st.pyplot(fig)

        # Diagnostics
        with st.expander("Model diagnostics (OLS summaries)"):
            st.write("**IS Curve**"); st.text(models2["model_is"].summary().as_text())
            st.write("**Phillips Curve**"); st.text(models2["model_pc"].summary().as_text())
            st.write("**Taylor Rule**"); st.text(models2["model_tr"].summary().as_text())

except Exception as e:
    st.error(f"Problem loading or running the selected model: {e}")
    st.stop()

