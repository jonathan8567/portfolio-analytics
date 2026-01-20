# Portfolio Performance Engine - Design & Best Practices

## 1. Core Concept: Unit Consistency
The most common error in portfolio analytics is **Unit Mismatch**.
- **Trades** are often recorded in USD (or Base Currency).
- **Market Data** (Bloomberg/Yahoo) often defaults to **Local Currency** (TWD, JPY, HKD).

**Impact**: If you hold 1,000 shares of TSMC (2330 TT):
- Price (TWD) ~ 1,000.
- Price (USD) ~ 30.
- If system uses TWD Price: Market Value = 1,000 * 1,000 = 1,000,000.
- Real Market Value (USD): 1,000 * 30 = 30,000.
- **Result**: Your assets are inflated 33x. Your leverage looks massive. Your returns are wrong.

**Solution**:
1. **Force Market Data to Base Currency**: Always request `Curr=USD` from Bloomberg.
2. **Explicit Data Schema**: Define columns clearly.

---

## 2. Recommended Data Structure

To support a robust **Long/Short Portfolio** with **Capital Projections** (Deposits/Withdrawals), we recommend the following Schema for your Trade Log:

### Columns
| Column Name | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| **Date** | Date | Yes | Trade date (YYYY-MM-DD) |
| **Ticker** | String | Yes | Bloomberg formatted (e.g., `2330 TT Equity`) |
| **Shares** | Float | Yes | (+) for Buy/Cover, (-) for Short/Sell. |
| **Traded_Total_Price** | Float | Yes | **Total value in USD**. Magnitude only (always positive). |
| **Action** | String | Optional | `BUY`, `SELL`, `SS` (Short Sell), `BTC` (Buy to Cover), `DEPOSIT`, `WITHDRAW`. |
| **FX_Rate** | Float | Optional | Exchange rate used for conversion (for audit). |
| **Comm_USD** | Float | Optional | Commission paid in USD. |

*Note: If `Action` is DEPOSIT, Ticker can be `CASH` or empty, and `Traded_Total_Price` is the amount.*

### Handling Cash Flows (Deposits/Withdrawals)
Currently, the system assumes a fixed `Initial Capital`.
To calculate **True Returns (Time-Weighted or Money-Weighted)** with deposits:
1.  **Deposits** increase Cash & Equity but should **NOT** count as Profit.
2.  **Withdrawals** decrease Cash & Equity but should **NOT** count as Loss.

**Formula**:
`Rt = (NAV_end - NAV_start - NetFlows) / (NAV_start + WeightedFlows)`

---

## 3. Implementation Plan (Corrective Actions)

### Step 1: Fix Currency Mismatch (Immediate)
Modify `BloombergDataManager` to append `Curr=USD` to all BDH requests. This ensures Market Value aligns with your USD Trade Log.

### Step 2: Refactor Cash Logic
Update `process_portfolio` to handle `Initial Capital` correctly and potentially infer Cash Flows if we add an `Action` column.

### Step 3: Return Calculation
Implement **Modified Dietz** method or simple **Daily Returns** adjusted for flows if flows occur.

---
