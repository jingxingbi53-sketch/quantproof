# QuantProof

QuantProof answers one practical research question: **Can this backtest be trusted?**

Upload daily strategy returns to receive a transparent audit score, data-quality
checks, cost-adjusted performance metrics, equity and drawdown charts, and a
transaction-cost sensitivity test.

## CSV contract

| Column | Required | Meaning |
| --- | --- | --- |
| `date` | Yes | Trading date in a parseable date format |
| `strategy_return` | Yes | Daily decimal return; use `0.01` for 1% |
| `turnover` | No | Daily traded fraction; use `0.40` for 40% |

If turnover is absent, QuantProof labels all performance as gross of transaction
costs instead of pretending that costs were applied.

## Audit checks

- Required columns and non-empty data
- Invalid or missing values
- Duplicate dates and chronological order
- Impossible or unusually large daily returns
- Minimum preferred sample length of 252 observations
- Availability of turnover for transaction-cost adjustment

The score is an explainable checklist, not a guarantee that a strategy will work.
Return data alone cannot prove the absence of look-ahead bias, survivorship bias,
data snooping, or overfitting.

## Run locally

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

## Run tests

```powershell
python -m unittest discover -p "test_*.py" -v
```

## Metric conventions

- 252 trading days per year
- 0% risk-free rate for the displayed Sharpe ratio
- Net return = gross return - turnover x transaction cost
- Equity curve = cumulative product of 1 + net return

For research and educational use only. This app does not provide investment advice.
