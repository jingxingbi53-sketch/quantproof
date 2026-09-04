"""Standard performance statistics for a periodic return series.

Everything here is descriptive: it says what the backtest
claims. Whether those claims survive scrutiny is the job of
:mod:`qp.diagnostics`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Performance:
    """What the backtest claims about itself."""

    n: int
    start: pd.Timestamp
    end: pd.Timestamp
    years: float
    periods_per_year: int

    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float

    hit_rate: float
    skew: float
    excess_kurtosis: float
    best_period: float
    worst_period: float
    var_95: float
    cvar_95: float

    longest_drawdown_days: int
    time_underwater: float

    def as_dict(self) -> dict:
        return asdict(self)


def per_period_rate(annual_rate: float, periods_per_year: int) -> float:
    """Convert an annual rate (e.g. a 4% risk-free rate) to one period."""
    if periods_per_year <= 0:
        return 0.0
    return float((1.0 + annual_rate) ** (1.0 / periods_per_year) - 1.0)


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound returns into a wealth index."""
    curve = initial * (1.0 + returns).cumprod()
    curve.name = "equity"
    return curve


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak, as a negative number."""
    curve = equity_curve(returns)
    peak = curve.cummax()
    dd = curve / peak - 1.0
    dd.name = "drawdown"
    return dd


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if dd.size else 0.0


def underwater_stats(returns: pd.Series) -> tuple[int, float]:
    """Longest stretch under the prior peak, and the share of time there."""
    dd = drawdown_series(returns)
    if dd.empty:
        return 0, 0.0

    underwater = dd < -1e-12
    share = float(underwater.mean())

    longest, current_start = 0, None
    for stamp, is_under in underwater.items():
        if is_under and current_start is None:
            current_start = stamp
        elif not is_under and current_start is not None:
            longest = max(longest, int((stamp - current_start).days))
            current_start = None
    if current_start is not None:
        longest = max(longest, int((dd.index[-1] - current_start).days))
    return longest, share


def sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int,
    rf_annual: float = 0.0,
) -> float:
    """Annualised Sharpe ratio using the naive sqrt-of-time scaling."""
    excess = returns - per_period_rate(rf_annual, periods_per_year)
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def sharpe_per_period(
    returns: pd.Series,
    periods_per_year: int,
    rf_annual: float = 0.0,
) -> float:
    """Sharpe in the data's own frequency, the unit the tests expect."""
    excess = returns - per_period_rate(rf_annual, periods_per_year)
    sd = float(excess.std(ddof=1))
    if not np.isfinite(sd) or sd == 0.0:
        return 0.0
    return float(excess.mean() / sd)


def sortino_ratio(
    returns: pd.Series,
    periods_per_year: int,
    rf_annual: float = 0.0,
) -> float:
    """Annualised return per unit of downside deviation."""
    excess = returns - per_period_rate(rf_annual, periods_per_year)
    downside = excess.clip(upper=0.0)
    dd = float(np.sqrt((downside ** 2).mean()))
    if not np.isfinite(dd) or dd == 0.0:
        return float("inf") if float(excess.mean()) > 0 else 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def year_fraction(
    index: pd.DatetimeIndex,
    n: int,
    periods_per_year: int,
) -> float:
    """Sample length in years, taken from the calendar when it is usable."""
    if index.size >= 2:
        span_days = float((index[-1] - index[0]).days)
        if span_days > 0:
            return span_days / 365.25
    return float(n) / float(periods_per_year)


def compute_performance(
    returns: pd.Series,
    periods_per_year: int,
    rf_annual: float = 0.0,
) -> Performance:
    """Compute the full descriptive picture of a return series."""
    n = int(returns.size)
    index = pd.DatetimeIndex(returns.index)
    years = year_fraction(index, n, periods_per_year)

    total_return = float((1.0 + returns).prod() - 1.0)
    can_compound = years > 0 and total_return > -1.0
    cagr = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if can_compound
        else float("nan")
    )
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))

    mdd = max_drawdown(returns)
    calmar = (
        float(cagr / abs(mdd))
        if mdd < 0 and np.isfinite(cagr)
        else float("nan")
    )
    longest_dd, underwater = underwater_stats(returns)

    values = returns.to_numpy(dtype=float)
    var_95 = float(np.percentile(values, 5)) if n else 0.0
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if tail.size else var_95

    return Performance(
        n=n,
        start=index[0] if n else pd.NaT,
        end=index[-1] if n else pd.NaT,
        years=years,
        periods_per_year=int(periods_per_year),
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe_ratio(returns, periods_per_year, rf_annual),
        sortino=sortino_ratio(returns, periods_per_year, rf_annual),
        max_drawdown=mdd,
        calmar=calmar,
        hit_rate=float((returns > 0).mean()) if n else 0.0,
        skew=float(returns.skew()) if n > 2 else 0.0,
        excess_kurtosis=float(returns.kurt()) if n > 3 else 0.0,
        best_period=float(returns.max()) if n else 0.0,
        worst_period=float(returns.min()) if n else 0.0,
        var_95=var_95,
        cvar_95=cvar_95,
        longest_drawdown_days=longest_dd,
        time_underwater=underwater,
    )


def rolling_sharpe(
    returns: pd.Series,
    periods_per_year: int,
    window: int | None = None,
) -> pd.Series:
    """Trailing annualised Sharpe; the window defaults to one year."""
    win = int(window or periods_per_year)
    win = max(4, min(win, max(4, returns.size)))
    mean = returns.rolling(win).mean()
    sd = returns.rolling(win).std(ddof=1)
    out = (mean / sd.replace(0.0, np.nan)) * np.sqrt(periods_per_year)
    out.name = "rolling_sharpe"
    return out


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """The worst drawdowns with their peak, trough and recovery dates.

    A single maximum-drawdown number hides the thing that actually decides
    whether a strategy survives contact with an investor: whether the loss was
    one bad week or a three-year grind back to the previous high.
    """
    if returns.empty:
        return pd.DataFrame()

    dd = drawdown_series(returns)
    underwater = (dd < -1e-12).to_numpy()
    index = pd.DatetimeIndex(dd.index)

    episodes = []
    start = None
    for position, is_under in enumerate(underwater):
        if is_under and start is None:
            start = position
        elif not is_under and start is not None:
            episodes.append((start, position - 1, True))
            start = None
    if start is not None:
        episodes.append((start, len(underwater) - 1, False))

    rows = []
    for begin, end, recovered in episodes:
        window = dd.iloc[begin:end + 1]
        trough = int(window.to_numpy().argmin())
        peak_date = index[begin - 1] if begin > 0 else index[begin]
        rows.append(
            {
                "depth": float(window.iloc[trough]),
                "peak": peak_date,
                "trough": index[begin + trough],
                "recovered": index[end] if recovered else pd.NaT,
                "length_days": int((index[end] - peak_date).days),
                "recovery_days": (
                    int((index[end] - index[begin + trough]).days)
                    if recovered
                    else -1
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.nsmallest(top, "depth").reset_index(drop=True)


def monthly_return_grid(returns: pd.Series) -> pd.DataFrame:
    """Returns compounded into a year-by-month grid, the classic tearsheet."""
    if returns.empty:
        return pd.DataFrame()
    index = pd.DatetimeIndex(returns.index)
    # Name the groupers explicitly: an unnamed grouper inherits the index's
    # own name and then collides with it on reset_index.
    years = pd.Index(index.year, name="year")
    months = pd.Index(index.month, name="month")
    growth = pd.Series(
        (1.0 + returns).to_numpy(dtype=float),
        index=pd.MultiIndex.from_arrays([years, months]),
    )
    monthly = growth.groupby(level=["year", "month"]).prod() - 1.0
    monthly.name = "return"
    return monthly.reset_index()


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    """Compounded return for each calendar year in the sample."""
    if returns.empty:
        return pd.Series(dtype=float)
    years = pd.DatetimeIndex(returns.index).year
    grouped = (1.0 + returns).groupby(years).prod() - 1.0
    grouped.index.name = "year"
    grouped.name = "return"
    return grouped
