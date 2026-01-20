import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

class MarketDataManager:
    """
    Handles fetching and processing of market data.
    """
    
    def __init__(self):
        pass

    def fetch_history(self, tickers: list, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """
        Fetches historical adjusted close prices for the given tickers.
        
        Args:
            tickers: List of ticker symbols (e.g., ['AAPL', 'MSFT']).
            start_date: Start date for data fetching.
            end_date: End date for data fetching (default: today).
            
        Returns:
            pd.DataFrame: DataFrame of Adjusted Close prices. Index is Date, Columns are Tickers.
        """
        if end_date is None:
            end_date = datetime.now()
            
        if not tickers:
            return pd.DataFrame()
        
        print(f"Fetching data for {len(tickers)} tickers from {start_date.date()} to {end_date.date()}...")
        
        try:
            # yfinance expects str dates YYYY-MM-DD
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = (end_date + timedelta(days=1)).strftime('%Y-%m-%d') # Add 1 day to include end_date
            
            data = yf.download(
                tickers, 
                start=start_str, 
                end=end_str, 
                auto_adjust=True, # Returns adjusted data (splits/dividends)
                progress=False,
                threads=True
            )
            
            # If multiple tickers, 'Close' is a MultiIndex or just the columns. 
            # yfinance structure varies by version. 
            # We want just the Close/Adj Close price.
            # With auto_adjust=True, the 'Close' column IS the adjusted close.
            
            if 'Close' in data.columns:
                df_close = data['Close']
            else:
                # If only one ticker, it might be just columns like Open, High, Low, Close...
                # Or if multiple, it's hierarchical.
                # Let's handle generic case.
                # If single index (one ticker), Close is a Series/Column.
                # If multi index (multiple tickers), Close is a DataFrame.
                 df_close = data['Close'] if 'Close' in data else data
            
            # If single ticker result, it might be a Series with name 'Close'??
            # yf.download for single ticker returns DataFrame with columns Open, High...
            if len(tickers) == 1:
                # Ensure it's a DataFrame with the ticker name as column
                ticker = tickers[0]
                if isinstance(df_close, pd.Series):
                    df_close = df_close.to_frame(name=ticker)
                elif isinstance(df_close, pd.DataFrame):
                    # If it's the full OHLC DF
                    if 'Close' in df_close.columns and ticker not in df_close.columns:
                         df_close = df_close[['Close']].rename(columns={'Close': ticker})
            
            # forward fill missing data (weekend/holiday handling logic can be complex, ffill is simple start)
            df_close = df_close.ffill()
            
            return df_close
            
        except Exception as e:
            print(f"Error fetching data: {e}. Returning empty DataFrame.")
            return pd.DataFrame()

    def get_benchmark_data(self, benchmark_ticker: str, start_date: datetime, end_date: datetime = None) -> pd.Series:
        """Fetches benchmark data."""
        df = self.fetch_history([benchmark_ticker], start_date, end_date)
        if not df.empty and benchmark_ticker in df.columns:
            return df[benchmark_ticker]
        return pd.Series()
