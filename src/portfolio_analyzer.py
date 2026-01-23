import pandas as pd
import numpy as np
from datetime import datetime

class PortfolioAnalyzer:
    """
    Core engine for portfolio analysis.
    Handles position calculation, valuation, and metric generation.
    """
    
    def __init__(self, market_data_manager):
        self.mdm = market_data_manager

    def process_portfolio(self, trades_df: pd.DataFrame, benchmark_ticker: str = 'QQQ', slippage_bps: float = 0.0, commission_bps: float = 10.0, initial_capital: float = 1_000_000.0):
        """
        Main workflow to calculate portfolio history and metrics.
        Args:
            commission_bps: Transaction commission in basis points (default 10.0).
        """
        if trades_df.empty:
            return {}

        # ... (Time Scope, Universe, Market Data steps remain unchanged)
        # 1. Time Scope
        start_date = trades_df['Date'].min()
        end_date = datetime.now()
        
        # 2. Get Universe
        tickers = trades_df['Ticker'].unique().tolist()
        
        # 3. Fetch Market Data
        # Include benchmark and FX
        fx_ticker = 'USDKRW Curncy'
        all_tickers = list(set(tickers + [benchmark_ticker, fx_ticker]))
        price_df = self.mdm.fetch_history(all_tickers, start_date, end_date)
        
        if price_df.empty:
            raise ValueError("No market data fetched. Check internet or tickers.")
            
        # 4. Calculate Daily Positions and Cash Flow
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        price_df = price_df.reindex(full_date_range).ffill()
        
        trades_df['DateOnly'] = trades_df['Date'].dt.normalize()
        
        # 4a. Positions (Share Count)
        share_changes = trades_df.pivot_table(index='DateOnly', columns='Ticker', values='Shares', aggfunc='sum').fillna(0)
        share_changes = share_changes.reindex(full_date_range, fill_value=0)
        holdings_calendar = share_changes.cumsum()
        holdings = holdings_calendar.reindex(price_df.index, method='ffill')

        # 4b. Cash Flow & Costs
        # Traded_Total_Value is signed cash flow. Negative = Outflow (Buy), Positive = Inflow (Sell)
        trades_df['TradeValue'] = trades_df['Traded_Total_Value'].abs()
        
        # A. Slippage (Cost on all trades)
        trades_df['SlippageCost'] = trades_df['TradeValue'] * (slippage_bps / 10000.0)
        
        # B. Commission (Cost on all trades)
        trades_df['CommissionCost'] = trades_df['TradeValue'] * (commission_bps / 10000.0)
        
        # C. Tax (0.3% only on SELLS for Taiwan Stocks)
        # Heuristic: Ticker ends with '.TW' or is purely numeric with 4 digits (e.g. '2330')
        def calculate_tax(row):
            # Only Tax on SELLS (Positive CashInflow, i.e., Traded_Total_Value > 0)
            if row['Traded_Total_Value'] > 0:
                ticker = str(row['Ticker'])
                is_tw = ticker.endswith('.TW') or (ticker.isdigit() and len(ticker) == 4) or 'TWSE' in ticker
                if is_tw:
                    return row['TradeValue'] * 0.003
            return 0.0

        trades_df['TaxCost'] = trades_df.apply(calculate_tax, axis=1)
        
        # Net Cash Effect = Principal - Costs
        # Note: Principal is signed. Costs are always positive outflows (subtracted).
        trades_df['PrincipalCashFlow'] = trades_df['Traded_Total_Value']
        trades_df['NetCashEffect'] = trades_df['PrincipalCashFlow'] - trades_df['SlippageCost'] - trades_df['CommissionCost'] - trades_df['TaxCost']
        
        cash_changes = trades_df.groupby('DateOnly')['NetCashEffect'].sum()
        
        # Reindex to full calendar range
        cash_changes = cash_changes.reindex(full_date_range, fill_value=0)
        cash_calendar = initial_capital + cash_changes.cumsum()
        
        # Sample cash on Trading Days
        cash_series = cash_calendar.reindex(price_df.index, method='ffill')
        
        # Initialize daily_df with index
        daily_df = pd.DataFrame(index=price_df.index)
        for t in tickers:
            daily_df[f'Pos_{t}'] = holdings[t] if t in holdings.columns else 0.0
        daily_df['Cash'] = cash_series
        
        # 5. Calculate Portfolio Value
        # Value = Cash + Sum(Holdings * Price * Multiplier / FX)
        # Split into Long and Short MV for analytics
        long_mv = pd.Series(0.0, index=daily_df.index)
        short_mv = pd.Series(0.0, index=daily_df.index)
        
        # FX Data (ensure no NaNs)
        usdkrw = price_df[fx_ticker].ffill().fillna(1200.0) if fx_ticker in price_df.columns else pd.Series(1.0, index=daily_df.index)
        
        for t in tickers:
            if t in price_df.columns:
                # Determine instrument properties
                mult = 1.0
                div_fx = pd.Series(1.0, index=daily_df.index)
                
                # Check for KM (KOSPI 200 Futures)
                # Matches: KM1 Index, KMA Index, KM...
                if t.startswith('KM'):
                    mult = 250000.0
                    # User requested no FX conversion for KM (Data seems to be pre-adjusted or using specific scale)
                    div_fx = pd.Series(1.0, index=daily_df.index)
                
                # Check for TWT (SGX Taiwan Futures - USD based)
                # User specified: TWT
                elif t.startswith('TWT') or t.startswith('TW'):
                     mult = 40.0
                     # USD based, no conversion needed (div_fx = 1.0)
                
                # Calculate Position Value
                # Value = Qty * Price * Multiplier / FX
                raw_price = price_df[t].fillna(0.0)
                qty = holdings[t]
                
                # Handle division by zero for FX just in case
                fx_safe = div_fx.replace(0.0, 1.0) 
                
                pos_val = (qty * raw_price * mult) / fx_safe
                pos_val = pos_val.fillna(0.0)
                
                # Update Aggregate Series
                long_mv = long_mv.add(pos_val.where(pos_val > 0, 0.0), fill_value=0.0)
                short_mv = short_mv.add(pos_val.where(pos_val < 0, 0.0), fill_value=0.0)
        
        market_value = long_mv + short_mv
        total_equity = cash_series + market_value
        
        # Gross and Net Exposure
        gross_exposure = long_mv + short_mv.abs()
        net_exposure = long_mv + short_mv
        
        # Avoid division by zero for leverage
        valid_equity = total_equity.replace(0, np.nan) 
        gross_leverage = gross_exposure / valid_equity
        net_leverage = net_exposure / valid_equity
        
        # 6. Benchmark Comparison
        # Normalize to 100 or just % returns
        # Ensure benchmark is also filled
        
        # Calculate daily returns (Total Equity / NAV)
        daily_returns = total_equity.pct_change().fillna(0)
        
        # FIX: Correct Day 0 return based on Initial Capital
        # pct_change() defaults first element to NaN -> 0. 
        # But we need (End_Day_0 / Initial_Capital) - 1
        if not daily_returns.empty and initial_capital > 0:
            daily_returns.iloc[0] = (total_equity.iloc[0] / initial_capital) - 1.0

        if benchmark_ticker in price_df.columns:
            bench_price = price_df[benchmark_ticker].ffill()
            bench_returns = bench_price.pct_change().fillna(0)
        else:
            bench_returns = pd.Series(0, index=daily_df.index)
            
        # 7. Metrics
        # 7. Metrics
        start_nav = initial_capital
        end_nav = total_equity.iloc[-1]
        days_count = (end_date - start_date).days
        
        metrics = self._calculate_metrics(daily_returns, bench_returns, start_nav, end_nav, days_count)
        metrics['Slippage_BPS'] = slippage_bps
        metrics['Total_Slippage_Cost'] = trades_df['SlippageCost'].sum()
        metrics['Total_Commission'] = trades_df['CommissionCost'].sum()
        metrics['Total_Tax'] = trades_df['TaxCost'].sum()
        metrics['Total_Transaction_Costs'] = metrics['Total_Slippage_Cost'] + metrics['Total_Commission'] + metrics['Total_Tax']
        
        # Turnover Calculation (Annualized)
        # Turnover ~ (Total Traded Value / 2) / Average Equity * (252 / Days)
        total_traded_val = trades_df['TradeValue'].sum()
        avg_equity = total_equity.mean()
        days_count = (end_date - start_date).days
        if days_count > 0 and avg_equity > 0:
            turnover_ann = (total_traded_val / 2.0) / avg_equity * (365.0 / days_count)
        else:
            turnover_ann = 0.0
        
        metrics['Turnover_Annualized'] = turnover_ann
        metrics['Avg_Gross_Leverage'] = gross_leverage.mean()
        metrics['Avg_Net_Leverage'] = net_leverage.mean()
        metrics['End_NAV'] = total_equity.iloc[-1]
        
        
        # Assemble Result
        result_df = pd.DataFrame({
            'Total_Equity': total_equity,
            'Cash': cash_series,
            'Market_Value': market_value,
            'Long_MV': long_mv,
            'Short_MV': short_mv,
            'Gross_Leverage': gross_leverage,
            'Net_Leverage': net_leverage,
            'Portfolio_Return': daily_returns,
            'Benchmark_Return': bench_returns,
            'Benchmark_Price': bench_price if benchmark_ticker in price_df.columns else 0
        })
        
        # Add Position Columns
        pos_cols = [c for c in daily_df.columns if c.startswith('Pos_')]
        if pos_cols:
            result_df = result_df.join(daily_df[pos_cols])
        
        # --- 6. Calculate Detailed Holdings (Weights) ---
        last_row_daily_metrics = result_df.iloc[-1]
        end_nav_for_holdings = last_row_daily_metrics['Total_Equity']
        
        holdings_list = []
        for col in result_df.columns: # Iterate through result_df for position columns
            if col.startswith('Pos_'):
                ticker = col.replace('Pos_', '')
                shares = last_row_daily_metrics[col]
                if abs(shares) > 0.0001:
                    # Get Price and FX
                    price = price_df.loc[result_df.index[-1], ticker] if ticker in price_df.columns else 0.0
                    
                    # Determine instrument properties for accurate MV calculation
                    mult = 1.0
                    div_fx_val = 1.0
                    
                    if ticker.startswith('KM'):
                        mult = 250000.0
                        div_fx_val = 1.0 # No FX conversion for KM
                    elif ticker.startswith('TWT') or ticker.startswith('TW'):
                        mult = 40.0
                        div_fx_val = 1.0 # USD based, no conversion needed
                    
                    # Calculate Market Value for the position
                    pos_mv = (shares * price * mult) / div_fx_val
                    
                    weight = (pos_mv / end_nav_for_holdings) * 100 if end_nav_for_holdings != 0 else 0.0
                    
                    holdings_list.append({
                        'Ticker': ticker,
                        'Shares': shares,
                        'Price': price,
                        'Market Value': pos_mv,
                        'Weight %': weight
                    })
        
        holdings_df = pd.DataFrame(holdings_list)
        if not holdings_df.empty:
             holdings_df = holdings_df.sort_values('Weight %', ascending=False)
        
        # --- NEW: Aggregate Asset Returns for Correlation Engine ---
        # We need a DataFrame of historical returns for all current holdings
        # Use price_df which is already loaded (contains all tickers)
        asset_returns_data = {}
        if not holdings_df.empty:
            for ticker in holdings_df['Ticker'].values:
                if ticker in price_df.columns:
                    prices = price_df[ticker]
                    # Align with portfolio date range
                    prices = prices[prices.index >= result_df.index[0]]
                    asset_returns_data[ticker] = prices.pct_change()
        
        asset_returns_df = pd.DataFrame(asset_returns_data).dropna(how='all')
        # Fill missing with 0 for now or forward fill (better) to keep correlation structure
        asset_returns_df = asset_returns_df.fillna(0.0)

        
        # --- 7. Calculate Rolling Beta (60 days) ---
        rolling_cov = result_df['Portfolio_Return'].rolling(window=60).cov(result_df['Benchmark_Return'])
        rolling_var = result_df['Benchmark_Return'].rolling(window=60).var()
        result_df['Rolling_Beta'] = rolling_cov / rolling_var
        
        # --- 8. Risk Engine Analytics (New) ---
        from src.risk_engine import RiskEngine
        risk_engine = RiskEngine(result_df['Portfolio_Return'])
        
        # Calculate VaR stats (95%)
        metrics['VaR_95_Hist'] = risk_engine.calculate_var_historical(0.95)
        metrics['VaR_95_Param'] = risk_engine.calculate_var_parametric(0.95)
        metrics['CVaR_95'] = risk_engine.calculate_expected_shortfall(0.95)
        
        # Run Stress Test
        current_beta = metrics['Beta']
        stress_df_beta = risk_engine.run_stress_test(current_beta, end_nav)
        
        # Run Sensitivity Test (Idiosyncratic)
        sens_df = risk_engine.run_sensitivity_test(holdings_df, end_nav)
        
        # Combine
        stress_df = pd.concat([stress_df_beta, sens_df], ignore_index=True)
        
        # Run Monte Carlo (Future VaR)
        mc_results = risk_engine.run_monte_carlo_simulation(end_nav, n_sims=10000, days=20)
        
        # --- NEW: Correlation-based Multivariate Stress Test ---
        multivar_stress_results = {}
        if not asset_returns_df.empty and asset_returns_df.shape[1] >= 2:
            # Get current weights aligned with asset_returns_df columns
            weights_dict = {}
            for _, row in holdings_df.iterrows():
                weights_dict[row['Ticker']] = row['Weight %'] / 100.0
            
            # Align weights array to asset_returns_df columns
            weights_arr = np.array([weights_dict.get(t, 0.0) for t in asset_returns_df.columns])
            # Renormalize if sum != 1
            if weights_arr.sum() > 0:
                weights_arr = weights_arr / weights_arr.sum()
            
            # Compute Correlation Matrix for visualization
            _, corr_matrix = RiskEngine.compute_covariance_matrix(asset_returns_df)
            
            # Run Normal Regime
            normal_stress = RiskEngine.run_multivariate_stress_test(
                asset_returns_df, weights_arr, end_nav, n_sims=5000, days=20, correlation_shock=0.0
            )
            
            # Run Panic Regime (High Correlation)
            panic_stress = RiskEngine.run_multivariate_stress_test(
                asset_returns_df, weights_arr, end_nav, n_sims=5000, days=20, correlation_shock=0.9
            )
            
            multivar_stress_results = {
                'normal': normal_stress,
                'panic': panic_stress,
                'corr_matrix': corr_matrix
            }
        
        # A. Rolling Sharpe (3-Month ~ 60 days) - Adjusted for short history
        # Sharpe = Mean / Std * sqrt(252)
        rolling_mean = result_df['Portfolio_Return'].rolling(60).mean()
        rolling_std = result_df['Portfolio_Return'].rolling(60).std()
        result_df['Rolling_Sharpe'] = (rolling_mean / rolling_std * np.sqrt(252)).fillna(0)
        
        # B. Rolling VaR (Rolling 1-Year Window)
        # 95% Var = 5th percentile
        # Use rolling window 252 (1 Year), min_periods=60 to show early evolution
        result_df['Rolling_VaR_95'] = result_df['Portfolio_Return'].rolling(window=252, min_periods=60).apply(lambda x: np.percentile(x, 5)).abs().fillna(0)
        
        # C. Volatility Cone Data
        vol_windows = [21, 63, 126, 252] # 1M, 3M, 6M, 1Y
        vol_cone_data = {}
        for w in vol_windows:
            vol_series = result_df['Portfolio_Return'].rolling(w).std() * np.sqrt(252)
            vol_cone_data[w] = vol_series
            
        # --- 10. Attribution Analysis (Beta vs Alpha) ---
        # Market Timing (Systematic) = Beta * Benchmark Return
        # Stock Selection (Idiosyncratic) = Portfolio Return - Systematic
        
        # Ensure we have Benchmark Returns aligned
        if 'Benchmark_Return' in result_df.columns:
            # Use 60-day Rolling Beta for dynamic attribution
            # (We already calculated Rolling_Beta above, using 60d window)
            beta_series = result_df['Rolling_Beta'].fillna(1.0) # Default to 1 if NaN
            
            result_df['Attr_Systematic'] = beta_series * result_df['Benchmark_Return']
            result_df['Attr_Selection'] = result_df['Portfolio_Return'] - result_df['Attr_Systematic']
            
            # Cumulative Attribution (Growth of $100)
            result_df['Cum_Systematic'] = (1 + result_df['Attr_Systematic']).cumprod() * 100
            result_df['Cum_Selection'] = (1 + result_df['Attr_Selection']).cumprod() * 100
            
        return {
            'daily_metrics': result_df,
            'summary_stats': metrics,
            'holdings_df': holdings_df,
            'stress_test_df': stress_df,
            'monte_carlo_results': mc_results,
            'vol_cone_data': vol_cone_data,
            'multivar_stress': multivar_stress_results
        }

    def _calculate_metrics(self, returns: pd.Series, benchmark_returns: pd.Series, start_nav: float = None, end_nav: float = None, days: int = 0) -> dict:
        """Calculates generic risk/return metrics."""
        # Risk free rate assumption 0
        rf = 0.0
        
        # Annualization factor
        ann_factor = 252
        
        # 1. Total Return (NAV-based is primary truth for closed portfolio)
        if start_nav and end_nav and start_nav > 0:
            total_return = (end_nav / start_nav) - 1
            
            # CAGR
            if days > 0:
                # (End/Start)^(365/Days) - 1
                cagr = (end_nav / start_nav) ** (365.0 / days) - 1
            else:
                cagr = 0.0
        else:
            # Fallback to geometric series
            total_return = (1 + returns).prod() - 1
            cagr = 0.0 # Cannot compute properly without time
            
        # 2. Volatility (Annualized)
        ann_factor = 252
        
        
        # Cumulative Return (Legacy/Check)
        # total_return = (1 + returns).prod() - 1
        
        # Volatility (Annualized)
        vol = returns.std() * np.sqrt(ann_factor)
        
        # Sharpe
        sharpe = (returns.mean() * ann_factor) / (vol if vol > 0 else 1)
        
        # Max Drawdown
        cum_ret_curve = (1 + returns).cumprod()
        peak = cum_ret_curve.cummax()
        drawdown = (cum_ret_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Alpha / Beta
        # Need covariance
        # Align series
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 10: # meaningful data
            cov = np.cov(aligned.iloc[:,0], aligned.iloc[:,1])
            beta = cov[0,1] / (cov[1,1] if cov[1,1] > 0 else 1)
            
            # Alpha (Jensen's) -> R_p = alpha + beta * R_m
            # Ann Alpha ~ (Mean_p - Beta * Mean_m) * 252
            mean_p = aligned.iloc[:,0].mean() * ann_factor
            mean_m = aligned.iloc[:,1].mean() * ann_factor
            alpha = mean_p - (beta * mean_m)
        else:
            beta = 0.0
            alpha = 0.0
            
        # Sortino Ratio
        # (Mean Return - RF) / Downside Deviation
        # Downside Deviation = std(returns where returns < 0)
        negative_returns = returns[returns < 0]
        if not negative_returns.empty:
            downside_std = negative_returns.std() * np.sqrt(ann_factor)
            sortino = (returns.mean() * ann_factor) / (downside_std if downside_std > 0 else 1)
        else:
            sortino = 0.0 # No negative returns implies infinite sortino, but report 0 or cap it
            if total_return > 0:
                sortino = 10.0 # High number to indicate perfection
        
        return {
            'Total_Return': total_return,
            'CAGR': cagr,
            'Volatility': vol,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_drawdown,
            'Alpha': alpha,
            'Beta': beta
        }
