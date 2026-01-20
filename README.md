# Portfolio Analytics Engine

**Institutional-grade quantitative portfolio analysis framework.**

This engine provides automated PnL tracking, advanced risk attribution (Monte Carlo, Volatility Cones), and professional-grade visualization for multi-asset strategies (Equities, Futures).

**[View Live Dashboard](https://jonathan8567.github.io/portfolio-analytics/)**

## Key Features

### 1. Robust Core Engine
-   **NAV-Based Accounting**: Accurately tracks daily Net Asset Value (NAV), handling cash flows, execution costs, and market value changes.
-   **Multi-Asset Support**: Built-in support for:
    -   Global Equities (via Bloomberg/Yahoo Finance).
    -   Futures (KOSPI 200, SGX Taiwan) with accurate contract multiplier and FX logic.
-   **Data Sources**: Supports local Excel files or **Live Google Sheets** integration.

### 2. Institutional Risk Analytics
-   **Monte Carlo Simulation**: Projects 10,000 future price paths to estimate Forward VaR.
-   **Volatility Cone**: Analyzes the term structure of realized volatility (30d, 60d, 90d).
-   **Attribution Analysis**: Decomposes returns into **Market Timing (Beta)** vs **Stock Selection (Alpha)**.
-   **Dynamic Risk**: Rolling Sharpe Ratio (60d) and Expanding Window VaR.

### 3. Professional Visualization
-   Generates a standalone **`index.html`** Dashboard (using Plotly).
-   Interactive Equity Curve, Drawdown Analysis, and Monthly Heatmaps.
-   **Risk Insights** section with specialized quantitative charts.

## Technology Stack
-   **Python 3.10+**: Core logic.
-   **Pandas/NumPy**: Vectorized time-series analysis.
-   **Plotly**: Interactive visualization.
-   **GSpread**: Google Sheets API integration.

## Getting Started

### Prerequisites
-   Python 3.10 or higher.
-   Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### 1. Standard (Excel)
Place your trade log (`data.xlsx`) in the root directory and run:

```bash
python main.py --benchmark "SPX Index" --slippage 10
```
This generates `index.html` in the same folder.

#### 2. Advanced (Google Sheets)
Load trades directly from a Google Sheet (requires `service_account.json`):

```bash
python main.py --gsheet_id "YOUR_SHEET_ID_HERE"
```

## Directory Structure
-   `src/`: Core analytics logic.
    -   `risk_engine.py`: VaR, CVaR, Monte Carlo logic.
    -   `portfolio_analyzer.py`: Metric calculation and attribution.
    -   `visualizer.py`: Plotly dashboard generation.
    -   `data_loader.py`: Excel and Google Sheets adapters.
-   `tests/`: Unit tests for risk models and valuation.

## License
MIT License
