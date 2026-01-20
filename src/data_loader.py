import pandas as pd
import os
from datetime import datetime

class DataLoader:
    """
    Handles loading and processing of trade data from Excel files.
    """
    
    REQUIRED_COLUMNS = ['Date', 'Ticker', 'Shares', 'Traded_Total_Value']

    def __init__(self):
        pass

    def load_trades(self, file_path: str) -> pd.DataFrame:
        """
        Loads trade history from an Excel file.
        
        Args:
            file_path: Absolute path to the Excel file.
            
        Returns:
            pd.DataFrame: Processed trade DataFrame.
            
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If required columns are missing.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trade file not found at: {file_path}")
            
        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read Excel file: {e}")
            
        # Validate columns
        # Normalize column names?
        # Let's be strict for now
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Ensure Date is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure Ticker is uppercase and stripped
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Ensure Traded_Total_Value is numeric (keep sign)
        if 'Traded_Total_Value' in df.columns:
            df['Traded_Total_Value'] = pd.to_numeric(df['Traded_Total_Value'], errors='coerce').fillna(0.0)
            
        # Derive Implied Price if missing
        if 'Price' not in df.columns and 'Traded_Total_Value' in df.columns and 'Shares' in df.columns:
             # Price = Total Value / Shares. Handle division by zero.
             # Traded_Total_Value is usually signed (Negative for Buy? Or Positive?)
             # Usually: Buy $1000 of stock. Cash -1000. 
             # Let's assume Price is abs(Value / Shares) to be safe.
             
             # Avoid div by zero
             safe_shares = df['Shares'].replace(0, 1) 
             df['Price'] = (df['Traded_Total_Value'] / safe_shares).abs()
        
        return df

    def get_unique_tickers(self, df: pd.DataFrame) -> list:
        """Returns a list of unique tickers from the trade log."""
        if 'Ticker' not in df.columns:
            return []
        return df['Ticker'].unique().tolist()

class GoogleSheetLoader:
    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        
    def load_trades(self, sheet_id: str) -> pd.DataFrame:
        """Loads trades from a Google Sheet."""
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError:
            raise ImportError("Please install gspread: pip install gspread google-auth")
            
        scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        
        creds = Credentials.from_service_account_file(self.credentials_path, scopes=scopes)
        client = gspread.authorize(creds)
        
        # Open Sheet
        try:
            sheet = client.open_by_key(sheet_id)
            worksheet = sheet.get_worksheet(0) # Assume first tab
            data = worksheet.get_all_records()
            
            df = pd.DataFrame(data)
            
            # Normalize Columns
            # Convert Date
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Ensure numeric
            df['Shares'] = pd.to_numeric(df['Shares'])
            df['Price'] = pd.to_numeric(df['Price'])
            if 'Commission' in df.columns:
                df['Commission'] = pd.to_numeric(df['Commission']).fillna(0)
            else:
                df['Commission'] = 0.0
                
            # Parse Ticker (ensure string)
            df['Ticker'] = df['Ticker'].astype(str).str.strip()
            
            # Handle Buy/Sell Action if simple signs not used
            if 'Action' in df.columns:
                # If "Buy"/"Sell" text is used, convert to sign
                mask_sell = df['Action'].str.lower().str.contains('sell', na=False)
                df.loc[mask_sell, 'Shares'] = -np.abs(df.loc[mask_sell, 'Shares'])
                df.loc[~mask_sell, 'Shares'] = np.abs(df.loc[~mask_sell, 'Shares'])
            
            print(f"Loaded {len(df)} trades from Google Sheet.")
            return df
            
        except Exception as e:
            print(f"Error loading Google Sheet: {e}")
            return pd.DataFrame()
