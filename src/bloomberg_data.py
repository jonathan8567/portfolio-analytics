import pandas as pd
import os
from datetime import datetime
import openpyxl
from openpyxl import Workbook
from openpyxl.styles import Font

class BloombergDataManager:
    """
    Handles fetching and processing of market data using local Excel files 
    populated with Bloomberg formulas.
    """
    
    DEFAULT_PATH = r"Z:\03 MidOffice\JonathanKu\data\price_history.xlsx"

    def __init__(self, file_path: str = None):
        self.file_path = file_path if file_path else self.DEFAULT_PATH

    def fetch_history(self, tickers: list, start_date: datetime, end_date: datetime = None) -> pd.DataFrame:
        """
        Main entry point. 
        1. Checks if file exists and has data for tickers.
        2. If not, generates file and prompts user.
        3. If yes, reads and returns DataFrame.
        """
        if end_date is None:
            end_date = datetime.now()

        # Check if we need to regenerate
        if self._needs_generation(tickers):
            print(f"Data file missing or outdated for requested tickers.")
            self._generate_request_file(tickers, start_date, end_date)
            print("="*60)
            print(f"ACTION REQUIRED:")
            print(f"1. A new file has been generated at:\n   {self.file_path}")
            print(f"2. Open this file in Excel on a terminal with Bloomberg.")
            print(f"3. Wait for 'Requesting Data...' to change to values.")
            print(f"4. Save the file (Ctrl+S) and Close it. (Ensure values are saved).")
            print(f"5. Run this script again.")
            print("="*60)
            return pd.DataFrame() # Return empty to signal stop

        return self._load_data(tickers)

    def get_benchmark_data(self, benchmark_ticker: str, start_date: datetime, end_date: datetime = None) -> pd.Series:
        """Fetches benchmark data."""
        # Treat benchmark as just another ticker
        df = self.fetch_history([benchmark_ticker], start_date, end_date)
        if not df.empty and benchmark_ticker in df.columns:
            return df[benchmark_ticker]
        return pd.Series()

    def _needs_generation(self, tickers: list) -> bool:
        """
        Checks if the file exists and contains columns for all tickers.
        This is a simple check. It doesn't verify date ranges deeply, 
        assuming if ticker exists, it's fine.
        """
        if not os.path.exists(self.file_path):
            return True
        
        try:
            # Read just the header to check tickers
            df = pd.read_excel(self.file_path, header=0, nrows=0)
            existing_cols = [c for c in df.columns if "Date" not in c and "Unnamed" not in c]
            
            # If any ticker is missing, we need to regenerate (or append, but regenerate is safer for now)
            # Normalization: File headers might be "2330 TT Equity".
            missing = [t for t in tickers if t not in existing_cols]
            
            if missing:
                print(f"Missing data for: {missing}")
                return True
                
            return False
        except Exception as e:
            print(f"Error checking file: {e}. Will regenerate.")
            return True

    def _generate_request_file(self, tickers: list, start_date: datetime, end_date: datetime):
        """
        Generates an Excel file with BDH formulas.
        Layout:
        Row 1: Ticker Name (A1, C1, E1...)
        Row 2: =BDH(...)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Price History"
        
        start_str = start_date.strftime('%m/%d/%Y')
        end_str = end_date.strftime('%m/%d/%Y')
        
        # BDH(Ticker, Field, Start, End)
        # We put Ticker in Row 1, Formula in Row 2
        # Columns: A=Date, B=Close, C=Date, D=Close... (Gap of 2)
        
        for i, ticker in enumerate(tickers):
            col_idx = 1 + (i * 2) # 1, 3, 5... (A, C, E)
            
            # Header (Ticker)
            cell_ticker = ws.cell(row=1, column=col_idx + 1) # B1, D1... Put ticker above value column
            cell_ticker.value = ticker
            cell_ticker.font = Font(bold=True)
            
            # Formula
            # =BDH("2330 TT Equity", "PX_LAST", "01/01/2023", "12/31/2023", "Dir=V", "Dts=S", "Sort=A", "Quote=C", "QtTyp=Y", "Days=T", "Per=cd", "DtFmt=D", "Fill=P", "UseDPDF=Y")
            # Using simpler defaults: BDH(ticker, "PX_LAST", start, end)
            
            # Formula Cell (Top-Left of the array) -> A2, C2, E2...
            # Actually BDH returns 2 columns (Date, Value). So we put formula in A2. It fills A2:B...
            cell_formula = ws.cell(row=2, column=col_idx)
            # Add Curr=USD to force conversion
            formula = f'=BDH("{ticker}", "PX_LAST", "{start_str}", "{end_str}", "Curr=USD")'
            cell_formula.value = formula
            
        wb.save(self.file_path)
        print(f"Generated Bloomberg request file at {self.file_path}")

    def _load_data(self, tickers: list) -> pd.DataFrame:
        """
        Reads the filled Excel file and consolidates into a single DataFrame.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            # We need to read the whole sheet. 
            # The structure is loose columns.
            df_raw = pd.read_excel(self.file_path, header=None)
            
            # Row 0 (1-based Row 1) has Tickers at indices 1, 3, 5... (Cols B, D, F...)
            # Data starts at Row 1 (Row 2).
            # But wait, if user saved it, the formula results are there.
            
            # Reconstruct DataFrame: Index=Date, Columns=Tickers
            combined_df = pd.DataFrame()
            
            # Iterate through the chunks
            # We know the logic: Every 2 columns is a ticker block.
            num_cols = df_raw.shape[1]
            
            for i in range(0, num_cols, 2):
                if i+1 >= num_cols:
                    break
                    
                # Ticker name is in Row 0, Col i+1 (Value column)
                ticker_name = df_raw.iloc[0, i+1]
                
                if pd.isna(ticker_name):
                    continue
                    
                # Data chunk (Date, Price)
                # Skip row 0 (header) and row 1 (formula itself if visible, usually value overrides or spills)
                # If values are pasted, row 1 might be start of data.
                # Let's assume standard array spill: Row 1 (index) might be the first date?
                # Actually, if formula is in A2 (index 1), data starts there.
                
                chunk = df_raw.iloc[1:, i:i+2].copy()
                chunk.columns = ['Date', ticker_name]
                
                # Clean up
                chunk = chunk.dropna(subset=['Date'])
                chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
                chunk = chunk.dropna(subset=['Date'])
                chunk = chunk.set_index('Date')
                
                # Coerce price to numeric
                chunk[ticker_name] = pd.to_numeric(chunk[ticker_name], errors='coerce')
                
                # Merge
                # Use outer join to keep all dates
                if combined_df.empty:
                    combined_df = chunk
                else:
                    combined_df = combined_df.join(chunk, how='outer')
            
            return combined_df.sort_index()
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
