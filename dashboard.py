import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from var_engine import VaREngine
from stress_testing import StressTester, SCENARIOS

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Risk Management Dashboard",
    page_icon=None,
    layout="wide",
)

st.title("Risk Management Dashboard")
st.caption("Value at Risk & Stress Testing Suite — three VaR methods, five historical crisis scenarios")

# ------------------------------------------------------------------
# Sidebar — portfolio parameters
# ------------------------------------------------------------------
st.sidebar.header("Portfolio Parameters")

portfolio_value = st.sidebar.number_input(
    "Portfolio Value ($)", min_value=10_000, max_value=100_000_000,
    value=1_000_000, step=10_000, format="%d"
)

confidence = st.sidebar.selectbox(
    "Confidence Level", options=[0.90, 0.95, 0.99], index=1,
    format_func=lambda x: f"{x:.0%}"
)

horizon = st.sidebar.selectbox(
    "Horizon (days)", options=[1, 5, 10, 21], index=0
)

beta = st.sidebar.slider("Portfolio Beta (vs S&P 500)", 0.0, 2.5, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.header("Return Parameters")
annual_return = st.sidebar.slider("Annual Return (%)", -20.0, 50.0, 10.0, 0.5)
annual_vol = st.sidebar.slider("Annual Volatility (%)", 5.0, 80.0, 20.0, 0.5)

# ------------------------------------------------------------------
# Generate synthetic returns from parameters
# ------------------------------------------------------------------
np.random.seed(42)
daily_mu = annual_return / 100 / 252
daily_sigma = annual_vol / 100 / np.sqrt(252)
n_days = 1260  # 5 years

raw = np.random.normal(daily_mu, daily_sigma, n_days)
# Add fat tails via Student-t mixture
t_shocks = np.random.standard_t(df=4, size=n_days) * daily_sigma * 0.3
returns = pd.Series(raw + t_shocks * (np.random.rand(n_days) < 0.05))

# ------------------------------------------------------------------
# VaR computation
# ------------------------------------------------------------------
engine = VaREngine(returns, portfolio_value)

hist = engine.historical_var(confidence, horizon)
param = engine.parametric_var(confidence, horizon)
mc = engine.monte_carlo_var(confidence, horizon)

results_df = engine.all_methods(confidence, horizon)

# ------------------------------------------------------------------
# Layout — row 1: KPI cards
# ------------------------------------------------------------------
st.subheader("VaR Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label=f"Historical VaR ({confidence:.0%}, {horizon}d)",
        value=f"${hist.var:,.0f}",
        delta=f"{hist.var_pct:.2f}% of portfolio",
        delta_color="inverse",
    )
    st.caption(f"CVaR (Expected Shortfall): ${hist.cvar:,.0f}")

with col2:
    st.metric(
        label=f"Parametric VaR ({confidence:.0%}, {horizon}d)",
        value=f"${param.var:,.0f}",
        delta=f"{param.var_pct:.2f}% of portfolio",
        delta_color="inverse",
    )
    st.caption(f"CVaR (Expected Shortfall): ${param.cvar:,.0f}")

with col3:
    st.metric(
        label=f"Monte Carlo VaR ({confidence:.0%}, {horizon}d)",
        value=f"${mc.var:,.0f}",
        delta=f"{mc.var_pct:.2f}% of portfolio",
        delta_color="inverse",
    )
    st.caption(f"CVaR (Expected Shortfall): ${mc.cvar:,.0f}")

# ------------------------------------------------------------------
# Row 2: return distribution + VaR lines
# ------------------------------------------------------------------
st.subheader("Return Distribution & VaR Thresholds")

fig = go.Figure()

hist_returns = returns * np.sqrt(horizon)
counts, bins = np.histogram(hist_returns, bins=80)
bin_centers = (bins[:-1] + bins[1:]) / 2

fig.add_trace(go.Bar(
    x=bin_centers * 100,
    y=counts,
    name="Simulated Returns",
    marker_color="#334155",
    opacity=0.8,
))

colors = {"Historical": "#3b82f6", "Parametric": "#f59e0b", "Monte Carlo": "#10b981"}
var_values = {
    "Historical": -hist.var_pct,
    "Parametric": -param.var_pct,
    "Monte Carlo": -mc.var_pct,
}

for label, val in var_values.items():
    fig.add_vline(
        x=val,
        line_dash="dash",
        line_color=colors[label],
        annotation_text=f"{label} VaR: {abs(val):.2f}%",
        annotation_position="top right",
    )

fig.update_layout(
    xaxis_title="Portfolio Return (%)",
    yaxis_title="Frequency",
    template="plotly_dark",
    height=380,
    showlegend=False,
    margin=dict(l=40, r=40, t=40, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------
# Row 3: method comparison table
# ------------------------------------------------------------------
st.subheader("Method Comparison")
st.dataframe(results_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# Row 4: stress testing
# ------------------------------------------------------------------
st.subheader("Historical Stress Scenarios")

tester = StressTester(portfolio_value, beta)
stress_df = tester.run_all()

col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.dataframe(stress_df, use_container_width=True, hide_index=True)

with col_right:
    losses = []
    names = []
    for name in SCENARIOS:
        r = tester.run_scenario(name)
        losses.append(abs(r.portfolio_loss))
        names.append(name.replace(" ", "<br>"))

    fig2 = go.Figure(go.Bar(
        x=losses,
        y=names,
        orientation="h",
        marker_color=["#ef4444", "#f97316", "#eab308", "#84cc16", "#3b82f6"],
        text=[f"${l:,.0f}" for l in losses],
        textposition="outside",
    ))
    fig2.update_layout(
        xaxis_title="Estimated Loss ($)",
        template="plotly_dark",
        height=320,
        margin=dict(l=10, r=80, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------------------------------
# Row 5: rolling VaR over time
# ------------------------------------------------------------------
st.subheader("Rolling 1-Year Historical VaR (95%)")

window = 252
rolling_var = []
dates = pd.date_range(end="2025-12-31", periods=len(returns), freq="B")

for i in range(window, len(returns)):
    window_returns = returns.iloc[i-window:i]
    v = -np.percentile(window_returns, 5) * portfolio_value
    rolling_var.append(v)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=dates[window:],
    y=rolling_var,
    mode="lines",
    line=dict(color="#3b82f6", width=1.5),
    name="Rolling VaR (95%)",
    fill="tozeroy",
    fillcolor="rgba(59,130,246,0.1)",
))
fig3.update_layout(
    yaxis_title="1-day VaR ($)",
    template="plotly_dark",
    height=300,
    margin=dict(l=40, r=40, t=20, b=40),
)
st.plotly_chart(fig3, use_container_width=True)

st.caption(
    "Returns generated via normal distribution with Student-t tail mixture. "
    "Stress scenario losses scaled by portfolio beta vs S&P 500. "
    "CVaR = Expected Shortfall beyond VaR threshold."
)
