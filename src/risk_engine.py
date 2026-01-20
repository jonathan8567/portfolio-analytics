import pandas as pd
import numpy as np
from scipy import stats

class RiskEngine:
    """
    Advanced Risk Analytics Engine.
    Calculates Value-at-Risk (VaR), Expected Shortfall (CVaR), and Stress Scenarios.
    """
    
    def __init__(self, daily_returns: pd.Series, risk_free_rate: float = 0.0):
        """
        Args:
            daily_returns: pd.Series of daily portfolio returns.
            risk_free_rate: Annualized risk-free rate (decimal).
        """
        self.returns = daily_returns.dropna()
        self.rf = risk_free_rate / 252 # Daily RF approximation
        
    def calculate_var_historical(self, confidence: float = 0.95) -> float:
        """
        Calculates Historical Value-at-Risk.
        Ex: 95% VaR = 5th percentile of return distribution.
        Returns positive float representing loss % (e.g. 0.02 for 2%).
        """
        if self.returns.empty:
            return 0.0
            
        percentile = (1.0 - confidence) * 100
        # Percentile gives negative return (e.g. -0.02). We want positive loss magnitude.
        var_return = np.percentile(self.returns, percentile)
        return abs(var_return)

    def calculate_var_parametric(self, confidence: float = 0.95) -> float:
        """
        Calculates Parametric (Normal) Value-at-Risk.
        VaR = (Mean - Z * Sigma)
        """
        if self.returns.empty:
            return 0.0
            
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # Z-score for confidence (e.g. 1.645 for 95%)
        z_score = stats.norm.ppf(confidence)
        
        # VaR cutoff return
        var_return = mu - (z_score * sigma)
        
        # Return loss magnitude
        return abs(var_return) if var_return < 0 else 0.0

    def calculate_expected_shortfall(self, confidence: float = 0.95) -> float:
        """
        Calculates Conditional VaR (CVaR) / Expected Shortfall.
        Average of all returns worse than the VaR cutoff.
        """
        if self.returns.empty:
            return 0.0
            
        var_cutoff = np.percentile(self.returns, (1.0 - confidence) * 100)
        tail_losses = self.returns[self.returns <= var_cutoff]
        
        if tail_losses.empty:
            return 0.0
            
        return abs(tail_losses.mean())

    def run_stress_test(self, beta: float, current_nav: float) -> pd.DataFrame:
        """
        Runs predefined stress scenarios based on Portfolio Beta.
        Returns DataFrame with Impact analysis.
        """
        scenarios = {
            "Black Monday (-20%)": -0.20,
            "GFC Crash (-50%)": -0.50,
            "Correction (-10%)": -0.10,
            "Tech Rally (+15%)": 0.15,
            "Mild Recession (-5%)": -0.05
        }
        
        results = []
        for name, shock in scenarios.items():
            est_return = shock * beta
            est_pnl = est_return * current_nav
            
            results.append({
                "Scenario": name,
                "Market Shock": shock,
                "Est. Portfolio Return": est_return,
                "Est. PnL Impact": est_pnl
            })
            
        return pd.DataFrame(results)

    def run_monte_carlo_simulation(self, current_nav: float, n_sims: int = 10000, days: int = 20) -> dict:
        """
        Runs Monte Carlo Simulation using Geometric Brownian Motion.
        
        Args:
            current_nav: Current Portfolio Net Asset Value.
            n_sims: Number of simulation paths (default 10,000).
            days: Number of days to simulate forward (default 20).
            
        Returns:
            dict: {
                'paths': np.array (n_sims, days),
                'final_values': np.array (n_sims,),
                'var_95_mc': float (loss amount),
                'percentiles': dict
            }
        """
        if self.returns.empty:
            return {}
            
        # 1. Calibrate Model Parameters
        mu = self.returns.mean()
        sigma = self.returns.std()
        
        # 2. Generate Random Paths
        # GBM: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)
        # We simulate daily steps
        dt = 1 # 1 day
        
        # Generate random Z-scores
        np.random.seed(42) # Reproducibility
        z_scores = np.random.normal(size=(n_sims, days))
        
        # Calculate daily drift and diffusion terms
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * z_scores
        
        # Calculate daily returns for each step
        daily_returns_sim = np.exp(drift + diffusion)
        
        # Accumulate returns to get Price Paths
        price_paths = np.zeros((n_sims, days))
        price_paths[:, 0] = current_nav * daily_returns_sim[:, 0]
        
        for t in range(1, days):
            price_paths[:, t] = price_paths[:, t-1] * daily_returns_sim[:, t]
            
        # 3. Analyze Results
        final_values = price_paths[:, -1]
        
        # Calculate Profit/Loss
        pnl = final_values - current_nav
        
        # Monte Carlo VaR (95%)
        # 5th percentile of PnL
        var_95_pnl = np.percentile(pnl, 5)
        
        # Expected Value (Mean)
        expected_val = np.mean(final_values)
        
        return {
            'paths': price_paths, # Full paths for plotting "Cone" or "Spaghetti"
            'final_values': final_values,
            'var_95_mc': abs(var_95_pnl) if var_95_pnl < 0 else 0.0,
            'expected_value': expected_val,
            'days': days
        }

    def run_sensitivity_test(self, holdings_df: pd.DataFrame, current_nav: float) -> pd.DataFrame:
        """
        Runs sensitivity analysis on Top 3 Holdings.
        Simulates a specific idiosyncratic shock (e.g. -15%) to each top position.
        """
        if holdings_df is None or holdings_df.empty:
            return pd.DataFrame()
            
        # Sort by weight desc
        top_holdings = holdings_df.sort_values('Weight %', ascending=False).head(3)
        
        results = []
        shock_pct = -0.15 # 15% drop
        
        for _, row in top_holdings.iterrows():
            ticker = row['Ticker']
            weight = row['Weight %'] / 100.0
            pk_val = row['Market Value']
            
            # Impact = Position Value * Shock
            pnl_impact = pk_val * shock_pct
            
            # Portfolio Return Impact = Weight * Shock
            portfolio_ret_impact = weight * shock_pct
            
            results.append({
                "Scenario": f"{ticker} Crash (-15%)",
                "Market Shock": shock_pct, # Proxy for the stock's move
                "Est. Portfolio Return": portfolio_ret_impact,
                "Est. PnL Impact": pnl_impact
            })
            
        return pd.DataFrame(results)
