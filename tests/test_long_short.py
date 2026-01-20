
import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.portfolio_analyzer import PortfolioAnalyzer

class MockMarketData:
    def fetch_history(self, tickers, start_date, end_date):
        # Create a date range
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Mock Prices
        # AAPL: 150 -> 160
        # MSFT: 300 -> 310
        # QQQ: 100 -> 101
        
        data = {
            'AAPL': [150.0] * len(dates),
            'MSFT': [300.0] * len(dates),
            'QQQ': [100.0] * len(dates)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Change prices on the second day if exists
        if len(dates) > 1:
            df.loc[dates[1]:, 'AAPL'] = 160.0
            df.loc[dates[1]:, 'MSFT'] = 310.0
            
        return df

class TestLongShort(unittest.TestCase):
    def test_metrics(self):
        mdm = MockMarketData()
        analyzer = PortfolioAnalyzer(mdm)
        
        # Setup Trades
        trades_data = [
            {'Date': datetime(2023, 1, 1), 'Ticker': 'AAPL', 'Shares': 100, 'Traded_Total_Value': -15000.0}, # Long Buy -> Cash Outflow
            {'Date': datetime(2023, 1, 1), 'Ticker': 'MSFT', 'Shares': -100, 'Traded_Total_Value': 30000.0} # Short Sell -> Cash Inflow
        ]
        trades_df = pd.DataFrame(trades_data)
        
        # Run Process
        initial_cap = 1_000_000.0
        # We need a longer date range to see annualized things, but let's just check the daily values logic first
        # Hack analysis date in analyzer?? 
        # Analyzer uses datetime.now() as end_date. 
        # We should probably mock datetime or just ignore the trailing days.
        # However, analyzer fetches history from trade min to now(). 
        # Our mock MDM will return data for that range.
        
        # Let's just trust logic for now and assertions.
        # But wait, analyzer uses datetime.now(). If I run this, it will fetch huge range.
        # I should probably update MockMDM to return small range?
        # Actually `fetch_history` arguments will determine it. 
        # But `process_portfolio` calculates `end_date = datetime.now()`.
        # This is a bit annoying for deterministic testing. 
        # I'll rely on the returned dataframe index.
        
        results = analyzer.process_portfolio(trades_df, benchmark_ticker='QQQ', initial_capital=initial_cap)
        daily = results['daily_metrics']
        stats = results['summary_stats']
        
        # Check Day 1 (2023-01-01)
        # Note: Timezone might be an issue, normalizing dates.
        d1 = daily.iloc[0]
        
        # Cash: 1M - 15k + 30k = 1,015,000
        self.assertAlmostEqual(d1['Cash'], 1_015_000.0)
        
        # Positions: AAPL 100, MSFT -100
        # Prices: 150, 300
        # Long MV: 15000
        # Short MV: -30000
        self.assertAlmostEqual(d1['Long_MV'], 15000.0)
        self.assertAlmostEqual(d1['Short_MV'], -30000.0)
        
        # Equity: 1,015,000 + 15,000 - 30,000 = 1,000,000
        self.assertAlmostEqual(d1['Total_Equity'], 1_000_000.0)
        
        # Gross Exposure: 15k + 30k = 45k
        self.assertAlmostEqual(daily.iloc[0]['Long_MV'] + abs(daily.iloc[0]['Short_MV']), 45000.0)
        
        # Check Day 2 (if exists in mock data returned by fetch_history)
        # Since end_date is now(), it will exist.
        # Prices: 160, 310
        # LMV: 16000
        # SMV: -31000
        # Equity: 1,015,000 + 16,000 - 31,000 = 1,000,000
        d2 = daily.iloc[1]
        self.assertAlmostEqual(d2['Long_MV'], 16000.0)
        self.assertAlmostEqual(d2['Short_MV'], -31000.0)
        self.assertAlmostEqual(d2['Total_Equity'], 1_000_000.0)
        
        # Check Returns (NAV based)
        # NAV started 1000000, ended 1000000. Return should be 0.
        self.assertAlmostEqual(stats['Total_Return'], 0.0)
        
        # Test CAGR if days > 0 (Mock data is 2023-01-01 to ... depends on current date)
        # Since we use datetime.now() for end date, days >> 365
        # If valid equity > 0
        if stats['Total_Return'] == 0:
             self.assertAlmostEqual(stats['CAGR'], 0.0)
        
        print("Test Passed: Long/Short Logic Valid")
        print("Total Return:", stats['Total_Return'])
        print("CAGR:", stats['CAGR'])
        print("Turnover Annualized:", stats['Turnover_Annualized'])
        print("Avg Gross Leverage:", stats['Avg_Gross_Leverage'])

if __name__ == '__main__':
    unittest.main()
