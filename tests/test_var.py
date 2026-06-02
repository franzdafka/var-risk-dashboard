import numpy as np
import pandas as pd
import pytest
from var_engine import VaREngine
from stress_testing import StressTester, SCENARIOS

@pytest.fixture
def engine():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))
    return VaREngine(returns, portfolio_value=1_000_000)

def test_historical_var_positive(engine):
    result = engine.historical_var(0.95, 1)
    assert result.var > 0
    assert result.cvar >= result.var

def test_parametric_var_positive(engine):
    result = engine.parametric_var(0.95, 1)
    assert result.var > 0

def test_monte_carlo_var_positive(engine):
    result = engine.monte_carlo_var(0.95, 1, n_simulations=10_000)
    assert result.var > 0

def test_higher_confidence_higher_var(engine):
    var_95 = engine.historical_var(0.95, 1).var
    var_99 = engine.historical_var(0.99, 1).var
    assert var_99 > var_95

def test_longer_horizon_higher_var(engine):
    var_1d = engine.parametric_var(0.95, 1).var
    var_10d = engine.parametric_var(0.95, 10).var
    assert var_10d > var_1d

def test_all_methods_returns_dataframe(engine):
    df = engine.all_methods(0.95, 1)
    assert len(df) == 3
    assert "Method" in df.columns

def test_stress_all_scenarios():
    tester = StressTester(1_000_000, beta=1.0)
    df = tester.run_all()
    assert len(df) == len(SCENARIOS)

def test_stress_losses_negative():
    tester = StressTester(1_000_000, beta=1.0)
    for name in SCENARIOS:
        result = tester.run_scenario(name)
        assert result.portfolio_loss < 0

def test_higher_beta_larger_loss():
    t1 = StressTester(1_000_000, beta=0.5).run_scenario("COVID-19 Crash")
    t2 = StressTester(1_000_000, beta=1.5).run_scenario("COVID-19 Crash")
    assert abs(t2.portfolio_loss) > abs(t1.portfolio_loss)
