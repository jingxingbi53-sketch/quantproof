from dataclasses import dataclass
from math import sqrt

import pandas as pd


TRADING_DAYS = 252
REQUIRED_COLUMNS = {"date", "strategy_return"}


@dataclass
class Check:
    name: str
    status: str
    detail: str
    points: int
    possible_points: int


@dataclass
class AuditResult:
    data: pd.DataFrame
    checks: list[Check]
    blocking_errors: list[str]
    problem_rows: pd.DataFrame

    @property
    def score(self) -> int:
        possible = sum(check.possible_points for check in self.checks)
        earned = sum(check.points for check in self.checks)
        return round(100 * earned / possible) if possible else 0

    @property
    def verdict(self) -> str:
        if self.blocking_errors or self.score < 50:
            return "Fail"
        if self.score < 80:
            return "Caution"
        return "Pass"


def _check(name: str, passed: bool, detail: str, points: int) -> Check:
    return Check(name, "Pass" if passed else "Warning", detail, points if passed else 0, points)


def prepare_backtest(data: pd.DataFrame, transaction_cost_bps: int) -> AuditResult:
    """Validate a daily-return backtest and build an auditable net return series."""
    cleaned = data.copy()
    checks: list[Check] = []
    errors: list[str] = []

    missing = REQUIRED_COLUMNS - set(cleaned.columns)
    if missing:
        text = ", ".join(sorted(missing))
        checks.append(Check("Required columns", "Fail", f"Missing: {text}", 0, 20))
        return AuditResult(cleaned, checks, [f"Missing required columns: {text}"], pd.DataFrame())
    checks.append(Check("Required columns", "Pass", "date and strategy_return found", 20, 20))

    if cleaned.empty:
        checks.append(Check("Valid values", "Fail", "The CSV has no data rows", 0, 20))
        return AuditResult(
            cleaned,
            checks,
            ["The CSV contains column headers but no data rows."],
            pd.DataFrame(),
        )

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned["strategy_return"] = pd.to_numeric(cleaned["strategy_return"], errors="coerce")
    checked_columns = ["date", "strategy_return"]

    has_turnover = "turnover" in cleaned.columns
    if has_turnover:
        cleaned["turnover"] = pd.to_numeric(cleaned["turnover"], errors="coerce")
        checked_columns.append("turnover")

    invalid = cleaned[checked_columns].isna().any(axis=1)
    if invalid.any():
        count = int(invalid.sum())
        checks.append(Check("Valid values", "Fail", f"{count} invalid or missing row(s)", 0, 20))
        errors.append(f"Found {count} row(s) with invalid or missing values.")
        return AuditResult(cleaned, checks, errors, cleaned.loc[invalid])
    checks.append(Check("Valid values", "Pass", "No invalid or missing values", 20, 20))

    duplicates = cleaned["date"].duplicated(keep=False)
    if duplicates.any():
        count = int(duplicates.sum())
        checks.append(Check("Unique dates", "Fail", f"{count} rows share duplicate dates", 0, 15))
        errors.append(f"Found {count} row(s) with duplicate dates.")
        return AuditResult(cleaned, checks, errors, cleaned.loc[duplicates])
    checks.append(Check("Unique dates", "Pass", "One observation per date", 15, 15))

    chronological = cleaned["date"].is_monotonic_increasing
    checks.append(_check("Chronological order", chronological, "Dates are ordered oldest to newest" if chronological else "Dates are out of order", 10))
    if not chronological:
        errors.append("Dates are not in chronological order.")
        return AuditResult(cleaned, checks, errors, cleaned)

    impossible = cleaned["strategy_return"] <= -1
    if impossible.any():
        checks.append(Check("Return bounds", "Fail", "Return at or below -100% found", 0, 10))
        errors.append("Strategy returns must be greater than -100%.")
        return AuditResult(cleaned, checks, errors, cleaned.loc[impossible])

    extreme_count = int((cleaned["strategy_return"].abs() > 0.25).sum())
    checks.append(_check("Return plausibility", extreme_count == 0, "No daily return exceeds 25%" if extreme_count == 0 else f"{extreme_count} unusually large daily return(s)", 10))

    enough_history = len(cleaned) >= TRADING_DAYS
    checks.append(_check("Sample length", enough_history, f"{len(cleaned)} daily observations; 252+ preferred", 10))

    if has_turnover:
        negative_turnover = cleaned["turnover"] < 0
        if negative_turnover.any():
            checks.append(Check("Turnover", "Fail", "Negative turnover found", 0, 15))
            errors.append("Turnover cannot be negative.")
            return AuditResult(cleaned, checks, errors, cleaned.loc[negative_turnover])
        cleaned["net_return"] = cleaned["strategy_return"] - cleaned["turnover"] * transaction_cost_bps / 10000
        checks.append(Check("Transaction costs", "Pass", f"Applied {transaction_cost_bps} bps to daily turnover", 15, 15))
    else:
        cleaned["turnover"] = pd.NA
        cleaned["net_return"] = cleaned["strategy_return"]
        checks.append(Check("Transaction costs", "Warning", "No turnover column; results remain gross of costs", 0, 15))

    cleaned["equity_curve"] = (1 + cleaned["net_return"]).cumprod()
    cleaned["drawdown"] = cleaned["equity_curve"] / cleaned["equity_curve"].cummax() - 1
    return AuditResult(cleaned, checks, errors, pd.DataFrame())


def calculate_metrics(returns: pd.Series) -> dict[str, float]:
    """Calculate descriptive metrics from decimal daily returns."""
    n = len(returns)
    total_return = float((1 + returns).prod() - 1)
    annualized_return = float((1 + total_return) ** (TRADING_DAYS / n) - 1)
    daily_volatility = float(returns.std(ddof=1)) if n > 1 else 0.0
    annualized_volatility = daily_volatility * sqrt(TRADING_DAYS)
    sharpe = float(returns.mean()) / daily_volatility * sqrt(TRADING_DAYS) if daily_volatility > 0 else 0.0
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
    }


def cost_sensitivity(data: pd.DataFrame, costs: list[int]) -> pd.DataFrame:
    """Recalculate terminal return across cost assumptions."""
    rows = []
    for cost in costs:
        net = data["strategy_return"] - data["turnover"] * cost / 10000
        rows.append({"Cost (bps)": cost, "Total return": float((1 + net).prod() - 1)})
    return pd.DataFrame(rows)
