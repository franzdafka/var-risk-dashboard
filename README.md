# Risk Management Dashboard — VaR & Stress Testing

A portfolio risk analytics suite implementing three Value at Risk methodologies and five historical crisis stress scenarios, with an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?logo=streamlit&logoColor=white)
![CI](https://github.com/franzdafka/var-risk-dashboard/actions/workflows/ci.yml/badge.svg)

---

## Overview

Implements the three standard VaR methods used in practice, applied to a configurable portfolio:

- **Historical Simulation** — non-parametric, empirical return distribution
- **Parametric (Delta-Normal)** — assumes normally distributed returns
- **Monte Carlo (GBM)** — 100,000 simulated paths with antithetic variates for variance reduction

Each method computes both VaR and CVaR (Expected Shortfall) — the expected loss conditional on exceeding VaR, which Basel III requires banks to report alongside VaR.

---

## Results (default: $1M portfolio, 95% confidence, 1-day horizon)

| Method | VaR ($) | VaR (%) | CVaR ($) |
|--------|---------|---------|---------|
| Historical Simulation | $18,240 | 1.82% | $25,310 |
| Parametric (Delta-Normal) | $17,890 | 1.79% | $22,460 |
| Monte Carlo (GBM) | $18,105 | 1.81% | $24,780 |

CVaR consistently exceeds VaR — capturing tail risk beyond the threshold, which VaR alone underestimates.

---

## Stress Testing — Historical Scenarios

| Scenario | Period | Market Shock | Portfolio Loss |
|----------|--------|-------------|---------------|
| 2008 Financial Crisis | Sep 2008 – Mar 2009 | -56.5% | -$565,000 |
| COVID-19 Crash | Feb – Mar 2020 | -34.0% | -$340,000 |
| 2022 Rate Hike Shock | Jan – Oct 2022 | -25.5% | -$255,000 |
| Dot-Com Bust | Mar 2000 – Oct 2002 | -49.1% | -$491,000 |
| Black Monday 1987 | 19 Oct 1987 | -22.8% | -$228,000 |

Portfolio losses scaled by user-defined beta vs S&P 500.

---

## Mathematical Foundation

### Value at Risk

For confidence level $\alpha$ and horizon $h$:

$$\text{VaR}_\alpha = -\inf\{x : P(L > x) \leq 1 - \alpha\}$$

**Parametric (Delta-Normal):**

$$\text{VaR}_\alpha = -(\mu h + z_\alpha \sigma \sqrt{h}) \cdot V$$

where $z_\alpha = \Phi^{-1}(1-\alpha)$ and $V$ is portfolio value.

**CVaR (Expected Shortfall):**

$$\text{CVaR}_\alpha = -\left(\mu h - \sigma\sqrt{h} \cdot \frac{\phi(z_\alpha)}{1-\alpha}\right) \cdot V$$

### Monte Carlo — Variance Reduction

Uses antithetic variates: for each random draw $z$, also simulates $-z$. This halves variance of the estimator without additional simulation cost.

### Square-Root-of-Time Scaling

Multi-day VaR estimated via $\text{VaR}_h = \text{VaR}_1 \times \sqrt{h}$, valid under i.i.d. return assumption.

---

## Project Structure

```
var-risk-dashboard/
├── var_engine.py       — VaREngine class (Historical, Parametric, Monte Carlo)
├── stress_testing.py   — StressTester class with 5 historical scenarios
├── dashboard.py        — Streamlit dashboard
├── tests/
│   └── test_var.py     — 9 unit tests
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Dashboard runs at `http://localhost:8501`. All parameters (portfolio value, confidence level, horizon, beta, volatility) are configurable via sidebar.

---

## Tests

```bash
pytest tests/ -v
```

9 tests covering: positive VaR outputs, CVaR ≥ VaR invariant, monotonicity in confidence and horizon, all stress scenarios, beta scaling.

---

## Key Concepts

| Concept | Implementation |
|---------|---------------|
| VaR | Three methods: Historical, Parametric, Monte Carlo |
| CVaR / Expected Shortfall | Computed for all methods; Basel III standard |
| Antithetic variates | Variance reduction in Monte Carlo |
| Fat tails | Student-t mixture in return generation |
| Stress testing | Five historical crisis scenarios with beta scaling |
| Square-root-of-time | Multi-day horizon scaling |

---

## Stack

Python · NumPy · SciPy · pandas · Streamlit · Plotly · pytest
