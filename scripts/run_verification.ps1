# Run Unit Tests
Write-Host "Running Unit Tests..."
& .venv\Scripts\python -m pytest tests/

# Run Main Analysis
Write-Host "Running Portfolio Analytics..."
& .venv\Scripts\python main.py --file data/mock_trades.xlsx --slippage 5 --output report.html

Write-Host "Done. Check report.html"
