import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from risk_engine import RiskEngine

class TestRiskEngine(unittest.TestCase):
    
    def setUp(self):
        # Create a series of normal returns (mean=0, std=0.01)
        # Using fixed seed for reproducibility
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        self.daily_returns = pd.Series(returns)
        self.engine = RiskEngine(self.daily_returns)
        
    def test_var_historical(self):
        # 95% Historical VaR ~ 1.645 std devs for normal dist
        # 0.01 * 1.645 = 0.01645
        var_95 = self.engine.calculate_var_historical(0.95)
        self.assertGreater(var_95, 0.015)
        self.assertLess(var_95, 0.018)
        
    def test_var_parametric(self):
        var_95 = self.engine.calculate_var_parametric(0.95)
        # Should be very close to 0.01645
        self.assertAlmostEqual(var_95, 0.01645, delta=0.002)
        
    def test_expected_shortfall(self):
        cvar_95 = self.engine.calculate_expected_shortfall(0.95)
        # CVaR should always be greater than VaR (the tail average is worse than the cutoff)
        var_95 = self.engine.calculate_var_historical(0.95)
        self.assertGreater(cvar_95, var_95)
        
    def test_stress_test(self):
        beta = 1.2
        nav = 1_000_000
        df = self.engine.run_stress_test(beta, nav)
        
        # Check specific scenario: Market -10% => Portfolio -12%
        correction = df[df['Scenario'] == "Correction (-10%)"].iloc[0]
        expected_ret = -0.10 * 1.2
        self.assertAlmostEqual(correction['Est. Portfolio Return'], expected_ret)
        self.assertAlmostEqual(correction['Est. PnL Impact'], expected_ret * nav)

    def test_monte_carlo(self):
        nav = 1_000_000
        n_sims = 1000
        days = 10
        
        # Run simulation
        result = self.engine.run_monte_carlo_simulation(nav, n_sims, days)
        
        # Check structure
        self.assertIn('paths', result)
        self.assertIn('var_95_mc', result)
        self.assertEqual(result['paths'].shape, (n_sims, days))
        
        # Check values
        # Since mean return is 0, expected value should be close to NAV
        # We allow some variance due to random walk
        self.assertGreater(result['expected_value'], nav * 0.98)
        self.assertLess(result['expected_value'], nav * 1.02)
        
        # VaR should be positive
        self.assertGreater(result['var_95_mc'], 0)

if __name__ == '__main__':
    unittest.main()
