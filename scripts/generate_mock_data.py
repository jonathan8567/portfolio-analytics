import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_mock_trades(file_path):
    """
    Generates a realistic mock trade history Excel file.
    tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] (US Tech focus)
    """
    
    np.random.seed(42)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Start date 1 year ago
    start_date = datetime.now() - timedelta(days=365)
    
    trades = []
    
    # Simulate an initial buy for each ticker
    for ticker in tickers:
        price = np.random.uniform(100, 300)
        shares = np.random.choice([10, 20, 50, 100])
        trades.append({
            'Date': start_date + timedelta(days=np.random.randint(0, 30)),
            'Ticker': ticker,
            'Shares': shares,
            'Traded_Avg_Price': price
        })
        
    # Simulate random trading activity
    for _ in range(20):
        ticker = np.random.choice(tickers)
        action = np.random.choice(['BUY', 'SELL'])
        days_offset = np.random.randint(31, 360)
        trade_date = start_date + timedelta(days=days_offset)
        
        # Simple price drift simulation
        base_price = 150
        price = base_price * (1 + np.random.normal(0, 0.2))
        
        # Shares
        shares = np.random.choice([5, 10, 15, 20])
        
        if action == 'SELL':
            shares = -shares # Sell is negative
            
        trades.append({
            'Date': trade_date,
            'Ticker': ticker,
            'Shares': shares,
            'Traded_Avg_Price': round(price, 2)
        })
        
    df = pd.DataFrame(trades)
    
    # Ensure date is sorted
    df = df.sort_values('Date')
    
    # Save
    df.to_excel(file_path, index=False)
    print(f"Mock trade data saved to {file_path}")
    print(df.head())

if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'mock_trades.xlsx')
    generate_mock_trades(os.path.abspath(output_path))
