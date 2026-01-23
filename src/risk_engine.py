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

    # ==================== PHASE 7: CORRELATION-BASED STRESS TESTING ====================
    
    @staticmethod
    def compute_covariance_matrix(asset_returns_df: pd.DataFrame) -> tuple:
        """
        Computes the Covariance and Correlation Matrix from asset returns.
        
        Args:
            asset_returns_df: DataFrame with columns = Tickers, index = Dates.
            
        Returns:
            tuple: (cov_matrix: pd.DataFrame, corr_matrix: pd.DataFrame)
        """
        if asset_returns_df.empty or asset_returns_df.shape[1] < 2:
            return pd.DataFrame(), pd.DataFrame()
            
        cov_matrix = asset_returns_df.cov()
        corr_matrix = asset_returns_df.corr()
        
        return cov_matrix, corr_matrix

    @staticmethod
    def run_multivariate_stress_test(
        asset_returns_df: pd.DataFrame,
        weights: np.ndarray,
        current_nav: float,
        n_sims: int = 5000,
        days: int = 20,
        correlation_shock: float = 0.0
    ) -> dict:
        """
        Runs Monte Carlo Simulation using Multivariate Normal Distribution.
        Accounts for asset correlations from historical data.
        
        Args:
            asset_returns_df: DataFrame of daily returns (cols=tickers).
            weights: np.array of current portfolio weights (same order as columns).
            current_nav: Current Portfolio Value.
            n_sims: Number of simulations.
            days: Forecast horizon.
            correlation_shock: If > 0, increase all off-diagonal correlations 
                               towards this value (e.g. 0.9 for "Panic" regime).
        
        Returns:
            dict: Contains 'var_95', 'cvar_95', 'expected_nav', 'paths' for visualization.
        """
        if asset_returns_df.empty or asset_returns_df.shape[1] < 2:
            return {}
            
        n_assets = asset_returns_df.shape[1]
        
        # 1. Estimate Parameters
        mu = asset_returns_df.mean().values  # (n_assets,)
        cov_matrix = asset_returns_df.cov().values  # (n_assets, n_assets)
        
        # 2. Apply Correlation Shock (Regime Switching)
        if correlation_shock > 0:
            # Extract volatilities (diagonal sqrt)
            vols = np.sqrt(np.diag(cov_matrix))
            # Compute correlation matrix
            corr_matrix = cov_matrix / np.outer(vols, vols)
            # Shock: move off-diagonal towards correlation_shock
            shocked_corr = np.where(
                np.eye(n_assets) == 1, 
                1.0, # Diagonal stays 1
                corr_matrix * (1 - correlation_shock) + correlation_shock
            )
            # Rebuild covariance matrix
            cov_matrix = shocked_corr * np.outer(vols, vols)
        
        # 3. Cholesky Decomposition
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback: Add small diagonal to ensure positive definiteness
            cov_matrix += np.eye(n_assets) * 1e-8
            L = np.linalg.cholesky(cov_matrix)
        
        # 4. Simulate Correlated Returns
        np.random.seed(42)  # Reproducibility
        
        # Generate uncorrelated standard normal shocks: (n_sims, days, n_assets)
        z = np.random.normal(size=(n_sims, days, n_assets))
        
        # Correlate shocks: z_corr = z @ L.T
        z_corr = np.einsum('ijk,lk->ijl', z, L)  # each (days, n_assets) gets correlated
        
        # Add drift to get daily returns
        daily_returns = mu + z_corr  # Broadcasting (n_sims, days, n_assets)
        
        # 5. Aggregate to Portfolio Level
        # Portfolio return per day = sum(asset_return * weight)
        portfolio_daily_returns = np.einsum('ijk,k->ij', daily_returns, weights)  # (n_sims, days)
        
        # Compound to get NAV paths
        cumulative_returns = np.cumprod(1 + portfolio_daily_returns, axis=1)
        nav_paths = current_nav * cumulative_returns  # (n_sims, days)
        
        # 6. Compute Risk Metrics
        final_navs = nav_paths[:, -1]
        pnl = final_navs - current_nav
        
        var_95 = np.percentile(pnl, 5)  # 5th percentile of P&L
        cvar_95 = pnl[pnl <= var_95].mean() if np.any(pnl <= var_95) else var_95
        expected_nav = np.mean(final_navs)
        
        return {
            'var_95': abs(var_95) if var_95 < 0 else 0.0,
            'cvar_95': abs(cvar_95) if cvar_95 < 0 else 0.0,
            'expected_nav': expected_nav,
            'paths': nav_paths,
            'days': days,
            'correlation_shock': correlation_shock
        }

