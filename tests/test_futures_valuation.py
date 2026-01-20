import pandas as pd
import numpy as np
from datetime import datetime
from src.portfolio_analyzer import PortfolioAnalyzer
from unittest.mock import MagicMock

def test_futures_valuation():
    print("--- Starting Futures Valuation Test ---")
    
    # Setup Data
    initial_capital = 1_000_000.0 # USD
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=3, freq='D')
    
    # Mock Market Data Manager
    mock_mdm = MagicMock()
    
    # Tickers: KM1 Index, TWT Index, USDKRW Curncy
    # KM Price: 0.3 (Reflecting observed data scale ~0.283)
    # TWT Price: 400.0 (Index Points)
    # USDKRW: 1200.0
    
    prices = pd.DataFrame({
        'KM1 Index': [0.30, 0.31, 0.31],
        'TWT Index': [400.0, 410.0, 410.0],
        'USDKRW Curncy': [1200.0, 1000.0, 1200.0] # Irrelevant for KM now
    }, index=dates)
    
    mock_mdm.fetch_history.return_value = prices
    
    # Portfolio:
    # Day 0: Buy 1 KM1 Index. Price 0.3.
    # Value = 1 * 0.3 * 250,000 = 75,000 USD. (No FX division)
    
    # Day 0: Buy 1 TWT Index. Price 400.
    # Value = 1 * 400 * 40 / 1 = 16,000 USD.
    
    # Total Market Value Day 0 = 75,000 + 16,000 = 91,000 USD.
    
    trades = pd.DataFrame([
        {'Date': start_date, 'Ticker': 'KM1 Index', 'Shares': 1, 'Traded_Total_Value': -75000},
        {'Date': start_date, 'Ticker': 'TWT Index', 'Shares': 1, 'Traded_Total_Value': -16000}
    ])
    
    analyzer = PortfolioAnalyzer(mock_mdm)
    results = analyzer.process_portfolio(trades, initial_capital=initial_capital)
    
    daily_df = results['daily_metrics']
    long_mv = daily_df['Long_MV']
    
    print("\n--- Market Value Check ---")
    print(daily_df[['Long_MV', 'Cash', 'Total_Equity']])
    
    # Check Day 0
    # KM: 0.3 * 250k = 75,000
    # TWT: 400 * 40 = 16,000
    # Total = 91,000
    expected_d0 = 91000.0
    actual_d0 = long_mv.iloc[0]
    
    print(f"\nDay 0 Expected MV: {expected_d0}")
    print(f"Day 0 Actual MV:   {actual_d0}")
    
    if abs(actual_d0 - expected_d0) < 1.0:
        print("PASS: Day 0 MV correct.")
    else:
        print("FAIL: Day 0 MV mismatch.")
        
    # Check Day 1
    # KM: Price 310. FX 1000.
    # Value = 1 * 310 * 250,000 / 1000 = 77,500.
    # TWT: Price 410.
    # Value = 1 * 410 * 40 = 16,400.
    # Total = 77,500 + 16,400 = 93,900.
    
    expected_d1 = 93900.0
    actual_d1 = long_mv.iloc[1]
    
    print(f"\nDay 1 Expected MV: {expected_d1}")
    print(f"Day 1 Actual MV:   {actual_d1}")
    
    if abs(actual_d1 - expected_d1) < 1.0:
        print("PASS: Day 1 MV correct.")
    else:
        print("FAIL: Day 1 MV mismatch.")

if __name__ == "__main__":
    test_futures_valuation()
