"""Separate alpha from beta.

The tests in :mod:`qp.diagnostics` ask whether a Sharpe ratio is real. They
cannot ask whether it is *yours*: a 1.2 Sharpe that is 0.8 beta to the equity
market is leverage, not a strategy, and every statistical test in the package
will happily call it significant.

This module regresses the strategy on a benchmark with Newey-West standard
errors, and hands back the hedged residual stream so the whole battery can be
re-run on what is left once the benchmark is taken away.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics


@dataclass
class Attribution:
    """What survives once the benchmark is regressed out."""

    n: int
    alpha_period: float
    alpha_annual: float
    alpha_t: float
    alpha_p: float
    beta: float
    beta_t: float
    r_squared: float
    correlation: float
    nw_lags: int
    hedged: pd.Series          # strategy minus beta times benchmark
    benchmark_sharpe: float
    strategy_sharpe: float
    hedged_sharpe: float


def newey_west_bandwidth(n: int) -> int:
    """Newey and West's (1994) automatic lag choice, 4(T/100)^(2/9)."""
    if n < 8:
        return 0
    return int(max(1, np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def _hac_covariance(x: np.ndarray, resid: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance of the OLS coefficients.

    Bartlett weights keep the estimate positive semi-definite, which a raw
    truncated sum does not guarantee.
    """
    n, k = x.shape
    xtx_inv = np.linalg.pinv(x.T @ x)
    u = x * resid[:, None]

    omega = u.T @ u
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        omega += weight * (gamma + gamma.T)

    cov = xtx_inv @ omega @ xtx_inv
    # Guard against tiny negative diagonals from round-off.
    return cov + np.eye(k) * 1e-300


def align(
    strategy: pd.Series,
    benchmark: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Intersect two return series on their dates."""
    frame = pd.DataFrame(
        {"strategy": strategy, "benchmark": benchmark}
    ).dropna()
    return frame["strategy"], frame["benchmark"]


def regress(
    strategy: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int,
    rf_annual: float = 0.0,
) -> Attribution | None:
    """Regress strategy on benchmark excess returns with HAC standard errors.

    Returns ``None`` when the two series barely overlap, since a regression on
    a handful of shared dates says nothing.
    """
    y, x = align(strategy, benchmark)
    n = int(y.size)
    if n < 20:
        return None

    rf = metrics.per_period_rate(rf_annual, periods_per_year)
    y_ex = y.to_numpy(dtype=float) - rf
    x_ex = x.to_numpy(dtype=float) - rf

    design = np.column_stack([np.ones(n), x_ex])
    coefficients, *_ = np.linalg.lstsq(design, y_ex, rcond=None)
    alpha, beta = float(coefficients[0]), float(coefficients[1])

    fitted = design @ coefficients
    resid = y_ex - fitted

    lags = newey_west_bandwidth(n)
    cov = _hac_covariance(design, resid, lags)
    se = np.sqrt(np.abs(np.diag(cov)))

    alpha_t = float(alpha / se[0]) if se[0] > 0 else float("nan")
    beta_t = float(beta / se[1]) if se[1] > 0 else float("nan")
    # Two-sided, on the t distribution rather than the normal: with a few
    # hundred observations the difference is small but free.
    alpha_p = (
        float(2.0 * stats.t.sf(abs(alpha_t), df=max(1, n - 2)))
        if np.isfinite(alpha_t)
        else float("nan")
    )

    total_var = float(np.var(y_ex, ddof=1))
    r_squared = (
        float(1.0 - np.var(resid, ddof=1) / total_var) if total_var > 0 else 0.0
    )

    # The hedged stream is what a desk would actually hold after shorting the
    # benchmark against the position, so alpha stays in and beta comes out.
    hedged = pd.Series(
        y.to_numpy(dtype=float) - beta * x.to_numpy(dtype=float),
        index=y.index,
        name="hedged",
    )

    return Attribution(
        n=n,
        alpha_period=alpha,
        alpha_annual=float(alpha * periods_per_year),
        alpha_t=alpha_t,
        alpha_p=alpha_p,
        beta=beta,
        beta_t=beta_t,
        r_squared=r_squared,
        correlation=float(np.corrcoef(y_ex, x_ex)[0, 1]),
        nw_lags=lags,
        hedged=hedged,
        benchmark_sharpe=metrics.sharpe_ratio(x, periods_per_year, rf_annual),
        strategy_sharpe=metrics.sharpe_ratio(y, periods_per_year, rf_annual),
        hedged_sharpe=metrics.sharpe_ratio(hedged, periods_per_year, rf_annual),
    )
