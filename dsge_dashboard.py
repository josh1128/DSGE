# dsge_dashboard.py
# -----------------------------------------------------------
# Streamlit app that can run:
#   1) Original model (DSGE.xlsx): DlogGDP, Dlog_CPI, Taylor
#   2) Simple NK model: Output Gap, Inflation, Nominal Rate
#
# This version:
# - Keeps full Original model with diagnostics
# - Adds Simple NK model with sliders & parameterized equations
# - Allows shock selection for both models
# -----------------------------------------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

# =========================================
# Page setup
# =========================================
st.set_page_config(page_title="DSGE IRF Dashboard", layout="wide")
st.title("DSGE IRF Dashboard — Original & Simple NK Models")

# =========================================
# Sidebar
# =========================================
with st.sidebar:
    st.header("Model Selection")
    model_choice = st.selectbox(
        "Choose model:",
        ["Original (DSGE.xlsx)", "Simple NK (Built-in)"]
    )

# =========================================
# ---- 1) Original Model (DSGE.xlsx) ----
# =========================================
if model_choice == "Original (DSGE.xlsx)":

    with st.sidebar:
        st.subheader("Data source")
        local_fallback = Path(__file__).parent / "DSGE.xlsx"
        xlf = st.file_uploader("Upload DSGE.xlsx", type=["xlsx"])
        if xlf:
            file_path = xlf
        else:
            file_path = local_fallback

    # Load data
    is_df = pd.read_excel(file_path, sheet_name="IS Curve")
    pc_df = pd.read_excel(file_path, sheet_name="Phillips")
    tr_df = pd.read_excel(file_path, sheet_name="Taylor")

    # Regression models
    model_is = sm.OLS(is_df["DlogGDP"], sm.add_constant(is_df[["Real.Interest.Rate_L2",
                                                                "Dlog_Foreign_Demand_L1",
                                                                "Dlog_REER",
                                                                "Dlog_Commodity_Energy",
                                                                "Dlog_Commodity_Non_Energy"]])).fit()

    model_pc = sm.OLS(pc_df["DlogCPI"], sm.add_constant(pc_df[["Output_Gap_L1",
                                                               "Dlog_Commodity_Energy",
                                                               "Dlog_Commodity_Non_Energy",
                                                               "Dlog_REER",
                                                               "Dlog_WTI"]])).fit()

    model_tr = sm.OLS(tr_df["Nominal.Interest.Rate"], sm.add_constant(tr_df[["Inflation.Gap",
                                                                             "Output.Gap"]])).fit()

    st.subheader("Diagnostics — Original Model")
    st.write("**IS Curve Model Summary**")
    st.text(model_is.summary())
    st.write("**Phillips Curve Model Summary**")
    st.text(model_pc.summary())
    st.write("**Taylor Rule Model Summary**")
    st.text(model_tr.summary())

    # Sidebar — Shock settings
    with st.sidebar:
        st.subheader("Shock Settings")
        shock_type = st.selectbox("Shock type", ["IS", "Phillips", "Taylor"])
        shock_size = st.slider("Shock size", -2.0, 2.0, 1.0, 0.1)
        horizon = st.slider("Horizon (quarters)", 4, 40, 20, 1)

    # Initialize responses
    resp_is = np.zeros(horizon)
    resp_pc = np.zeros(horizon)
    resp_tr = np.zeros(horizon)

    # Apply shock
    if shock_type == "IS":
        resp_is[0] = shock_size
    elif shock_type == "Phillips":
        resp_pc[0] = shock_size
    elif shock_type == "Taylor":
        resp_tr[0] = shock_size

    # Simulate responses
    for t in range(1, horizon):
        resp_is[t] = (model_is.params[1] * resp_tr[t-2] if t >= 2 else 0) \
                     + model_is.params[2] * 0 \
                     + model_is.params[3] * 0 \
                     + model_is.params[4] * 0 \
                     + model_is.params[5] * 0 \
                     + model_is.params[0]
        resp_pc[t] = model_pc.params[1] * (resp_is[t-1] if t >= 1 else 0) \
                     + model_pc.params[2] * 0 \
                     + model_pc.params[3] * 0 \
                     + model_pc.params[4] * 0 \
                     + model_pc.params[5] * 0 \
                     + model_pc.params[0]
        resp_tr[t] = model_tr.params[1] * resp_pc[t] \
                     + model_tr.params[2] * resp_is[t] \
                     + model_tr.params[0]

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    axes[0].plot(range(horizon), resp_is, label="DlogGDP")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title("IS Curve — Response of GDP")

    axes[1].plot(range(horizon), resp_pc, label="DlogCPI", color="orange")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_title("Phillips Curve — Response of Inflation")

    axes[2].plot(range(horizon), resp_tr, label="Nominal Interest Rate", color="green")
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    axes[2].set_title("Taylor Rule — Response of Interest Rate")

    plt.tight_layout()
    st.pyplot(fig)

# =========================================
# ---- 2) Simple NK Model ----
# =========================================
elif model_choice == "Simple NK (Built-in)":

    @dataclass
    class NKParams:
        beta: float = 0.99
        sigma: float = 1.00
        kappa: float = 0.10
        phi_pi: float = 1.50
        phi_x: float = 0.125
        rho_i: float = 0.80
        rho_x: float = 0.50
        rho_r: float = 0.80
        rho_u: float = 0.50
        gamma_pi: float = 0.50

    class SimpleNK3Eq:
        def __init__(self, params: NKParams | None = None):
            self.p = params or NKParams()

        def irf(self, shock: str = "demand", T: int = 24, size: float = 1.0):
            p = self.p
            x = np.zeros(T)
            pi = np.zeros(T)
            i = np.zeros(T)
            r_nat = np.zeros(T)
            u = np.zeros(T)
            e_i = np.zeros(T)

            if shock == "demand":
                r_nat[0] = size
            elif shock == "cost":
                u[0] = size
            elif shock == "policy":
                e_i[0] = size
            else:
                raise ValueError("shock must be 'demand', 'cost', or 'policy'.")

            for t in range(T):
                if t > 0:
                    r_nat[t] += p.rho_r * r_nat[t-1]
                    u[t] += p.rho_u * u[t-1]

                x_lag = x[t-1] if t > 0 else 0.0
                pi_lag = pi[t-1] if t > 0 else 0.0
                i_lag = i[t-1] if t > 0 else 0.0

                A_x = (1 - p.rho_i) * (p.phi_pi * p.kappa + p.phi_x) - p.kappa
                B_const = (p.rho_i * i_lag
                           + ((1 - p.rho_i) * p.phi_pi * p.gamma_pi - p.gamma_pi) * pi_lag
                           + ((1 - p.rho_i) * p.phi_pi - 1.0) * u[t]
                           + e_i[t])

                denom = 1.0 + (A_x / p.sigma)
                num = (p.rho_x * x_lag) - (B_const / p.sigma) + (r_nat[t] / p.sigma)
                x[t] = num / max(denom, 1e-8)

                pi[t] = p.gamma_pi * pi_lag + p.kappa * x[t] + u[t]
                i[t] = p.rho_i * i_lag + (1 - p.rho_i) * (p.phi_pi * pi[t] + p.phi_x * x[t]) + e_i[t]

            return np.arange(T), x, pi, i

    # Sidebar controls
    with st.sidebar:
        st.subheader("NK Parameters")
        beta = st.slider("β (discount factor)", 0.90, 1.00, 0.99, 0.01)
        sigma = st.slider("σ (sensitivity to interest rates)", 0.5, 3.0, 1.0, 0.1)
        kappa = st.slider("κ (Phillips slope)", 0.01, 0.5, 0.10, 0.01)
        phi_pi = st.slider("ϕπ (Taylor response to inflation)", 0.5, 3.0, 1.5, 0.1)
        phi_x = st.slider("ϕx (Taylor response to output gap)", 0.01, 1.0, 0.125, 0.01)
        rho_i = st.slider("ρi (interest rate inertia)", 0.0, 1.0, 0.8, 0.05)
        rho_x = st.slider("ρx (output gap persistence)", 0.0, 1.0, 0.5, 0.05)
        rho_r = st.slider("ρr (demand shock persistence)", 0.0, 1.0, 0.8, 0.05)
        rho_u = st.slider("ρu (cost shock persistence)", 0.0, 1.0, 0.5, 0.05)
        gamma_pi = st.slider("γπ (inflation persistence)", 0.0, 1.0, 0.5, 0.05)

        shock_type = st.selectbox("Shock type", ["demand", "cost", "policy"])
        shock_size = st.slider("Shock size", -2.0, 2.0, 1.0, 0.1)
        horizon = st.slider("Horizon (quarters)", 4, 40, 24, 1)

    params = NKParams(beta, sigma, kappa, phi_pi, phi_x, rho_i, rho_x, rho_r, rho_u, gamma_pi)
    nk_model = SimpleNK3Eq(params)
    t_vals, x_vals, pi_vals, i_vals = nk_model.irf(shock=shock_type, T=horizon, size=shock_size)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8))
    axes[0].plot(t_vals, x_vals, label="Output gap", color="blue")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_title("Output gap (pp)")

    axes[1].plot(t_vals, pi_vals, label="Inflation", color="orange")
    axes[1].axhline(0, color="black", lw=0.8, ls="--")
    axes[1].set_title("Inflation (pp)")

    axes[2].plot(t_vals, i_vals, label="Nominal rate", color="green")
    axes[2].axhline(0, color="black", lw=0.8, ls="--")
    axes[2].set_title("Nominal interest rate (pp)")

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown(f"**IS Curve:** $x_t = {rho_x:.2f} x_{{t-1}} - (B/σ) + (r^n_t / σ)$")
    st.markdown(f"**Phillips Curve:** $π_t = {gamma_pi:.2f} π_{{t-1}} + {kappa:.2f} x_t + u_t$")
    st.markdown(f"**Taylor Rule:** $i_t = {rho_i:.2f} i_{{t-1}} + (1 - {rho_i:.2f}) ({phi_pi:.2f} π_t + {phi_x:.2f} x_t) + e^i_t$")


