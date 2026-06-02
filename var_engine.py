import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Optional

@dataclass
class VaRResult:
    method: str
    confidence_level: float
    horizon_days: int
    var: float          # as positive number (loss)
    cvar: float         # Expected Shortfall
    portfolio_value: float
    var_pct: float      # VaR as % of portfolio
    cvar_pct: float

class VaREngine:
    """
    Value at Risk engine implementing three standard methods:
    - Historical Simulation
    - Parametric (Delta-Normal)
    - Monte Carlo
    """

    def __init__(self, returns: pd.Series, portfolio_value: float = 1_000_000):
        """
        Parameters
        ----------
        returns : pd.Series
            Daily portfolio returns (decimal, e.g. 0.01 for 1%)
        portfolio_value : float
            Current portfolio value in base currency
        """
        self.returns = returns.dropna()
        self.portfolio_value = portfolio_value
        self.mu = returns.mean()
        self.sigma = returns.std()

    # ------------------------------------------------------------------
    # 1. Historical Simulation
    # ------------------------------------------------------------------
    def historical_var(self, confidence: float = 0.95, horizon: int = 1) -> VaRResult:
        """
        Non-parametric VaR from empirical return distribution.
        Scales to multi-day horizon by square-root-of-time rule.
        """
        scaled = self.returns * np.sqrt(horizon)
        var_return = np.percentile(scaled, (1 - confidence) * 100)
        cvar_return = scaled[scaled <= var_return].mean()

        var_abs = -var_return * self.portfolio_value
        cvar_abs = -cvar_return * self.portfolio_value

        return VaRResult(
            method="Historical Simulation",
            confidence_level=confidence,
            horizon_days=horizon,
            var=var_abs,
            cvar=cvar_abs,
            portfolio_value=self.portfolio_value,
            var_pct=-var_return * 100,
            cvar_pct=-cvar_return * 100,
        )

    # ------------------------------------------------------------------
    # 2. Parametric (Delta-Normal)
    # ------------------------------------------------------------------
    def parametric_var(self, confidence: float = 0.95, horizon: int = 1) -> VaRResult:
        """
        Assumes normally distributed returns.
        VaR = -(mu*h - z * sigma * sqrt(h)) * portfolio_value
        """
        z = stats.norm.ppf(1 - confidence)
        mu_h = self.mu * horizon
        sigma_h = self.sigma * np.sqrt(horizon)

        var_return = -(mu_h + z * sigma_h)
        cvar_return = -(mu_h - sigma_h * stats.norm.pdf(z) / (1 - confidence))

        var_abs = var_return * self.portfolio_value
        cvar_abs = cvar_return * self.portfolio_value

        return VaRResult(
            method="Parametric (Delta-Normal)",
            confidence_level=confidence,
            horizon_days=horizon,
            var=var_abs,
            cvar=cvar_abs,
            portfolio_value=self.portfolio_value,
            var_pct=var_return * 100,
            cvar_pct=cvar_return * 100,
        )

    # ------------------------------------------------------------------
    # 3. Monte Carlo
    # ------------------------------------------------------------------
    def monte_carlo_var(
        self,
        confidence: float = 0.95,
        horizon: int = 1,
        n_simulations: int = 100_000,
        seed: Optional[int] = 42,
    ) -> VaRResult:
        """
        Simulates return paths assuming GBM with fitted mu and sigma.
        Uses antithetic variates for variance reduction.
        """
        rng = np.random.default_rng(seed)
        half = n_simulations // 2
        z = rng.standard_normal((half, horizon))
        z = np.concatenate([z, -z], axis=0)  # antithetic variates

        daily_returns = self.mu + self.sigma * z
        path_returns = daily_returns.sum(axis=1)  # sum over horizon

        var_return = np.percentile(path_returns, (1 - confidence) * 100)
        cvar_return = path_returns[path_returns <= var_return].mean()

        var_abs = -var_return * self.portfolio_value
        cvar_abs = -cvar_return * self.portfolio_value

        return VaRResult(
            method="Monte Carlo (GBM)",
            confidence_level=confidence,
            horizon_days=horizon,
            var=var_abs,
            cvar=cvar_abs,
            portfolio_value=self.portfolio_value,
            var_pct=-var_return * 100,
            cvar_pct=-cvar_return * 100,
        )

    # ------------------------------------------------------------------
    # Convenience: run all three
    # ------------------------------------------------------------------
    def all_methods(self, confidence: float = 0.95, horizon: int = 1) -> pd.DataFrame:
        results = [
            self.historical_var(confidence, horizon),
            self.parametric_var(confidence, horizon),
            self.monte_carlo_var(confidence, horizon),
        ]
        rows = []
        for r in results:
            rows.append({
                "Method": r.method,
                "Confidence": f"{r.confidence_level:.0%}",
                "Horizon (days)": r.horizon_days,
                "VaR ($)": f"${r.var:,.0f}",
                "CVaR ($)": f"${r.cvar:,.0f}",
                "VaR (%)": f"{r.var_pct:.2f}%",
                "CVaR (%)": f"{r.cvar_pct:.2f}%",
            })
        return pd.DataFrame(rows)
