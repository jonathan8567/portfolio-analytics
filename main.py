import argparse
import os
import pandas as pd
from src.data_loader import DataLoader
from src.bloomberg_data import BloombergDataManager
from src.portfolio_analyzer import PortfolioAnalyzer
from src.visualizer import PortfolioVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Portfolio Analytics Engine")
    parser.add_argument('--file', type=str, required=False, help="Path to trade log Excel file")
    parser.add_argument('--capital', type=float, default=1000000.0, help="Initial Capital (NAV), default 1,000,000")
    parser.add_argument('--slippage', type=float, default=0.0, help="Slippage in basis points (e.g. 5.0)")
    parser.add_argument('--benchmark', type=str, default='TWSE Index', help="Benchmark ticker (Bloomberg format, e.g. TWSE Index)")
    parser.add_argument('--output', type=str, default='index.html', help='Output HTML file')
    
    # Google Sheets Args
    parser.add_argument('--gsheet_id', type=str, help='Google Sheet ID (replaces files)')
    parser.add_argument('--creds', type=str, default='service_account.json', help='Path to Google Service Account JSON')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Auto-generate timestamped filename if default is used
    if args.output == 'index.html':
        # Default now FIXED FILENAME
        output_file = 'index.html'
    else:
        output_file = args.output
        
    # Default file path
    file_path = args.file
    if file_path is None:
        file_path = r"C:\Users\jonathanku\Documents\Jonathan Ku\Code\data.xlsx"
        
    trades_df = pd.DataFrame()
    
    # 1. Load Data
    if args.gsheet_id:
        print(f"Loading trades from Google Sheet: {args.gsheet_id}")
        from src.data_loader import GoogleSheetLoader
        if not os.path.exists(args.creds):
             print(f"Error: Credential file '{args.creds}' not found. Please provide valid path.")
             return
             
        loader = GoogleSheetLoader(args.creds)
        trades_df = loader.load_trades(args.gsheet_id)
        if trades_df.empty:
             print("No trades loaded from Sheet. Exiting.")
             return
    else:
        if not os.path.exists(file_path):
            print(f"Error: Input file not found at {file_path}")
            print("Please provide a valid file path using --file or ensure the default file exists.")
            return
        
        print(f"Starting analysis on {file_path}")
        loader = DataLoader()
        try:
            trades_df = loader.load_trades(file_path)
            print(f"Loaded {len(trades_df)} trades.")
        except Exception as e:
            print(f"Error loading data: {e}")
            return

    print(f"Slippage: {args.slippage} bps")
    print(f"Benchmark: {args.benchmark}")

    # 2. Market Data
    # Use Bloomberg Data Manager
    mdm = BloombergDataManager()
    
    # 3. Analyze
    analyzer = PortfolioAnalyzer(mdm)
    print("Fetching market data and calculating metrics...")
    try:
        results = analyzer.process_portfolio(trades_df, args.benchmark, args.slippage, args.capital)
    except ValueError as ve:
        # Check if it was due to empty market data (Stop condition from BloombergManager)
        if "No market data fetched" in str(ve):
            print("\n*** Analysis Stopped ***")
            print("Please follow the instructions above to generate/update the data file.")
            return
        else:
            print(f"Error processing portfolio: {ve}")
            return
    except Exception as e:
        print(f"Error processing portfolio: {e}")
        import traceback
        traceback.print_exc()
        return

    if not results or results['daily_metrics'].empty:
        print("Analysis failed or produced no data.")
        return

    daily_df = results['daily_metrics']
    stats = results['summary_stats']
    holdings_df = results.get('holdings_df')
    stress_df = results.get('stress_test_df')
    mc_results = results.get('monte_carlo_results')
    vol_cone_data = results.get('vol_cone_data')
    
    # Explicitly set Benchmark in stats for display
    stats['Benchmark'] = args.benchmark
    
    # Print Stats
    print("\n--- Performance Metrics ---")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    # 4. Visualize
    viz = PortfolioVisualizer()
    print("\nGenerating professional dashboard...")
    
    # Generate full HTML string
    html_report = viz.create_dashboard_html(daily_df, stats, holdings_df, trades_df, stress_df, mc_results, vol_cone_data)
    
    # Save to HTML
    # Ensure it saves in the script directory if no path provided
    if not os.path.isabs(output_file):
        output_file = os.path.join(os.path.dirname(__file__), output_file)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_report)
        
    print(f"\nReport saved to {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
