import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict

@dataclass
class StressResult:
    scenario: str
    period: str
    market_shock: float       # % shock applied
    portfolio_loss: float     # absolute loss
    portfolio_loss_pct: float # % loss
    description: str

# Historical shock magnitudes (max drawdown over crisis period)
SCENARIOS: Dict[str, dict] = {
    "2008 Financial Crisis": {
        "period": "Sep 2008 – Mar 2009",
        "equity_shock": -0.565,   # S&P 500 peak-to-trough
        "vol_spike": 3.2,         # VIX multiplier
        "description": "Lehman Brothers collapse; S&P 500 fell 56.5% peak-to-trough. "
                        "Credit markets froze, interbank lending halted.",
    },
    "COVID-19 Crash": {
        "period": "Feb 2020 – Mar 2020",
        "equity_shock": -0.340,   # S&P 500 drawdown
        "vol_spike": 4.1,
        "description": "Fastest 30% drawdown in S&P 500 history (33 days). "
                        "VIX reached 85.47 on 18 March 2020.",
    },
    "2022 Rate Hike Shock": {
        "period": "Jan 2022 – Oct 2022",
        "equity_shock": -0.255,
        "vol_spike": 1.8,
        "description": "Fed hiked rates 425bps in 12 months. S&P 500 fell 25%, "
                        "Bloomberg Global Aggregate Bond Index fell 16% — worst year since 1976.",
    },
    "Dot-Com Bust": {
        "period": "Mar 2000 – Oct 2002",
        "equity_shock": -0.491,
        "vol_spike": 2.5,
        "description": "Nasdaq fell 78% peak-to-trough. S&P 500 down 49%. "
                        "Tech-heavy portfolios experienced catastrophic losses.",
    },
    "Black Monday 1987": {
        "period": "19 Oct 1987",
        "equity_shock": -0.228,   # single-day S&P 500
        "vol_spike": 5.0,
        "description": "Single largest one-day percentage drop in S&P 500 history: -22.8%. "
                        "Triggered by portfolio insurance strategies and program trading.",
    },
}

class StressTester:
    """
    Applies historical stress scenarios to a portfolio.
    Uses equity beta to scale index shocks to portfolio losses.
    """

    def __init__(self, portfolio_value: float = 1_000_000, beta: float = 1.0):
        """
        Parameters
        ----------
        portfolio_value : float
        beta : float
            Portfolio beta vs equity market (1.0 = market-neutral assumption)
        """
        self.portfolio_value = portfolio_value
        self.beta = beta

    def run_scenario(self, scenario_name: str) -> StressResult:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        s = SCENARIOS[scenario_name]
        shock = s["equity_shock"] * self.beta
        loss = shock * self.portfolio_value
        loss_pct = shock * 100

        return StressResult(
            scenario=scenario_name,
            period=s["period"],
            market_shock=s["equity_shock"] * 100,
            portfolio_loss=loss,
            portfolio_loss_pct=loss_pct,
            description=s["description"],
        )

    def run_all(self) -> pd.DataFrame:
        rows = []
        for name in SCENARIOS:
            r = self.run_scenario(name)
            rows.append({
                "Scenario": r.scenario,
                "Period": r.period,
                "Market Shock": f"{r.market_shock:.1f}%",
                "Portfolio Loss ($)": f"${abs(r.portfolio_loss):,.0f}",
                "Portfolio Loss (%)": f"{r.portfolio_loss_pct:.1f}%",
            })
        return pd.DataFrame(rows)

    def worst_case(self) -> StressResult:
        results = [self.run_scenario(n) for n in SCENARIOS]
        return min(results, key=lambda r: r.portfolio_loss)
