"""Statistical tests that ask whether a track record means anything.

Two families live here.

*Inference* asks whether the observed Sharpe could plausibly have come from a
strategy with no edge: the standard error of the Sharpe ratio under non-normal
returns, the Probabilistic Sharpe Ratio, the Deflated Sharpe Ratio (which prices
in how many strategies were tried before this one was picked), and the minimum
track record length needed for the number to clear a significance bar.

*Fragility* asks whether the number survives contact with reality:
autocorrelation that inflates the Sharpe, dependence on a handful of
lucky periods, transaction costs, a suspiciously smooth equity path,
and stability across sub-periods.

References
----------
Lo (2002), "The Statistics of Sharpe Ratios."
Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier."
Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio."
Getmansky, Lo & Makarov (2004), "An Econometric Model of Serial Correlation and
Illiquidity in Hedge Fund Returns."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import metrics

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Sharpe inference
# ---------------------------------------------------------------------------

def sharpe_variance_factor(
    sr_period: float,
    skew: float,
    kurtosis: float,
) -> float:
    """The bracketed term in Lo's (2002) variance of the Sharpe estimator.

    ``kurtosis`` is the raw fourth moment ratio (3.0 for a normal), not excess.
    The factor exceeds 1 when returns are negatively skewed or fat-tailed, which
    is exactly when a headline Sharpe is least trustworthy.
    """
    factor = 1.0 - skew * sr_period + ((kurtosis - 1.0) / 4.0) * sr_period ** 2
    # Sample moments can push this negative in tiny samples; keep it usable.
    return float(max(factor, 1e-9))


def sharpe_standard_error(
    sr_period: float,
    n: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Standard error of the per-period Sharpe estimate."""
    if n < 3:
        return float("nan")
    factor = sharpe_variance_factor(sr_period, skew, kurtosis)
    return float(np.sqrt(factor / (n - 1)))


def probabilistic_sharpe_ratio(
    sr_period: float,
    n: int,
    skew: float,
    kurtosis: float,
    sr_benchmark_period: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark), given skew, fat tails and sample size."""
    se = sharpe_standard_error(sr_period, n, skew, kurtosis)
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr_period - sr_benchmark_period) / se))


def min_track_record_length(
    sr_period: float,
    skew: float,
    kurtosis: float,
    sr_benchmark_period: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for the Sharpe to clear the benchmark.

    Returns ``inf`` when the observed Sharpe is at or below the benchmark, since
    no amount of extra data makes a non-edge significant.
    """
    edge = sr_period - sr_benchmark_period
    if edge <= 0:
        return float("inf")
    z = float(stats.norm.ppf(confidence))
    factor = sharpe_variance_factor(sr_period, skew, kurtosis)
    return float(1.0 + factor * (z / edge) ** 2)


def expected_max_sharpe(n_trials: int, trial_sharpe_std: float) -> float:
    """Expected best Sharpe from ``n_trials`` genuinely worthless strategies.

    This is the multiple-testing correction: search hard enough and something
    will look brilliant by chance. Bailey & Lopez de Prado approximate the
    expected maximum of ``n_trials`` independent draws with the Gumbel limit.
    """
    if n_trials <= 1 or trial_sharpe_std <= 0:
        return 0.0
    n = float(n_trials)
    a = float(stats.norm.ppf(1.0 - 1.0 / n))
    b = float(stats.norm.ppf(1.0 - 1.0 / (n * np.e)))
    return float(trial_sharpe_std * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(
    sr_period: float,
    n: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    trial_sharpe_std_period: float,
) -> tuple[float, float]:
    """PSR measured against the Sharpe that pure search would have produced.

    Returns ``(dsr, benchmark_sharpe_period)``.
    """
    benchmark = expected_max_sharpe(n_trials, trial_sharpe_std_period)
    dsr = probabilistic_sharpe_ratio(
        sr_period, n, skew, kurtosis, benchmark
    )
    return dsr, benchmark


def trial_sharpe_dispersion(
    returns: pd.Series,
    periods_per_year: int,
) -> tuple[float, str]:
    """Estimate the spread of Sharpe ratios across the trials that were run.

    The Deflated Sharpe Ratio needs the variance of the trial Sharpes, which no
    upload can contain. The best available proxy is how much this strategy's own
    Sharpe moves between non-overlapping years; if there are too few years, fall
    back to the estimator's own standard error, which is the conservative floor.
    Returns the annualised dispersion and a label describing where it came from.
    """
    arr = returns.to_numpy(dtype=float)
    n, ppy = arr.size, int(periods_per_year)
    blocks = [arr[i:i + ppy] for i in range(0, n, ppy)]
    usable = [b for b in blocks if b.size >= max(6, ppy // 4)]

    sharpes = []
    for block in usable:
        sd = float(np.std(block, ddof=1))
        if sd > 0:
            sharpes.append(float(np.mean(block) / sd * np.sqrt(ppy)))

    if len(sharpes) >= 3:
        dispersion = float(np.std(np.asarray(sharpes), ddof=1))
        if dispersion > 0:
            return dispersion, (
                "spread of this strategy's own year-by-year"
                " Sharpe ratios"
            )

    sd = float(np.std(arr, ddof=1))
    sr_p = float(np.mean(arr) / sd) if sd > 0 else 0.0
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = (
        float(stats.kurtosis(arr, fisher=False, bias=False))
        if n > 3
        else 3.0
    )
    se = sharpe_standard_error(sr_p, n, skew, kurt)
    fallback = float(se * np.sqrt(ppy)) if np.isfinite(se) else 0.5
    return max(fallback, 1e-6), (
        "standard error of this Sharpe estimate (too few years to"
        " measure it directly)"
    )


# ---------------------------------------------------------------------------
# Autocorrelation and smoothing
# ---------------------------------------------------------------------------

def autocorrelations(returns: pd.Series, max_lag: int = 10) -> np.ndarray:
    """Sample autocorrelations for lags 1..max_lag."""
    arr = returns.to_numpy(dtype=float)
    n = arr.size
    lags = int(max(1, min(max_lag, n // 4)))
    centred = arr - arr.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 0:
        return np.zeros(lags)
    return np.array([
        float(np.dot(centred[k:], centred[:-k]) / denom)
        for k in range(1, lags + 1)
    ])


def ljung_box(
    returns: pd.Series,
    max_lag: int = 10,
) -> tuple[float, float, int]:
    """Ljung-Box test for serial correlation: (Q, p-value, lags)."""
    rho = autocorrelations(returns, max_lag)
    n = int(returns.size)
    h = int(rho.size)
    if h == 0 or n <= h + 1:
        return float("nan"), float("nan"), 0
    ks = np.arange(1, h + 1)
    q = float(n * (n + 2) * np.sum(rho ** 2 / (n - ks)))
    p = float(stats.chi2.sf(q, df=h))
    return q, p, h


def lo_annualisation_factor(
    returns: pd.Series,
    periods_per_year: int,
    max_lag: int = 10,
) -> float:
    """Lo's (2002) correct scaling factor, which replaces the naive sqrt(q).

    Positively autocorrelated returns compound their risk faster than the
    square-root rule assumes, so the naive annualised Sharpe is overstated.
    Autocorrelations beyond ``max_lag`` are treated as zero.
    """
    q = float(periods_per_year)
    if q <= 1:
        return 1.0
    rho = autocorrelations(returns, min(max_lag, int(q) - 1))
    ks = np.arange(1, rho.size + 1)
    denom = q + 2.0 * float(np.sum((q - ks) * rho))
    if denom <= 0:
        return float("nan")
    return float(q / np.sqrt(denom))


def smoothing_profile(returns: pd.Series, periods_per_year: int) -> dict:
    """Autocorrelation summary plus the Sharpe it would survive at."""
    rho = autocorrelations(returns, 10)
    q_stat, p_value, lags = ljung_box(returns, 10)
    factor = lo_annualisation_factor(returns, periods_per_year)
    sr_p = metrics.sharpe_per_period(returns, periods_per_year)
    naive = float(sr_p * np.sqrt(periods_per_year))
    adjusted = float(sr_p * factor) if np.isfinite(factor) else float("nan")
    return {
        "rho": rho,
        "rho1": float(rho[0]) if rho.size else 0.0,
        "ljung_box_q": q_stat,
        "ljung_box_p": p_value,
        "ljung_box_lags": lags,
        "naive_sharpe": naive,
        "adjusted_sharpe": adjusted,
        "sharpe_inflation": (
            float(naive - adjusted)
            if np.isfinite(adjusted)
            else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def _chunk_sizes(total: int, chunk: int) -> list[int]:
    full, rest = divmod(total, chunk)
    return [chunk] * full + ([rest] if rest else [])


def bootstrap_sharpe(
    returns: pd.Series,
    periods_per_year: int,
    n_boot: int = 2000,
    block: int | None = None,
    seed: int = 7,
) -> dict:
    """Circular block bootstrap confidence interval for the annualised Sharpe.

    Blocks preserve short-range dependence, so the interval does not assume the
    returns are independent -- which for most strategies they are not.
    """
    arr = returns.to_numpy(dtype=float)
    n = arr.size
    if n < 10:
        return {"samples": np.array([]), "block": 0, "n_boot": 0}

    size = int(block or max(2, round(n ** (1.0 / 3.0))))
    size = max(1, min(size, n))
    n_blocks = int(np.ceil(n / size))
    rng = np.random.default_rng(seed)
    scale = np.sqrt(periods_per_year)
    offsets = np.arange(size)

    out = []
    for batch in _chunk_sizes(n_boot, 250):
        starts = rng.integers(0, n, size=(batch, n_blocks))
        raw = starts[:, :, None] + offsets[None, None, :]
        idx = raw.reshape(batch, -1) % n
        sample = arr[idx[:, :n]]
        sd = sample.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out.append(np.where(sd > 0, sample.mean(axis=1) / sd * scale, 0.0))

    samples = np.concatenate(out) if out else np.array([])
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return {"samples": samples, "block": size, "n_boot": 0}

    return {
        "samples": samples,
        "block": size,
        "n_boot": int(samples.size),
        "p05": float(np.percentile(samples, 5)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "prob_positive": float(np.mean(samples > 0)),
    }


def permutation_drawdown(
    returns: pd.Series,
    n_sim: int = 2000,
    seed: int = 11,
) -> dict:
    """Where the observed max drawdown sits among reshuffles.

    Shuffling keeps every return but destroys their order. If the real path is
    far smoother than almost any reordering of it, the sequence itself is doing
    suspicious work -- the signature of look-ahead bias or a fitted curve.
    """
    arr = returns.to_numpy(dtype=float)
    n = arr.size
    observed = metrics.max_drawdown(returns)
    if n < 20:
        return {
            "observed": observed,
            "samples": np.array([]),
            "percentile": float("nan"),
        }

    log_r = np.log1p(np.clip(arr, -0.9999, None))
    rng = np.random.default_rng(seed)

    out = []
    for batch in _chunk_sizes(n_sim, 250):
        tiled = np.tile(log_r, (batch, 1))
        shuffled = np.apply_along_axis(rng.permutation, 1, tiled)
        cum = np.cumsum(shuffled, axis=1)
        peak = np.maximum.accumulate(cum, axis=1)
        out.append(np.expm1(np.min(cum - peak, axis=1)))

    samples = np.concatenate(out) if out else np.array([])
    percentile = (
        float(np.mean(samples > observed))
        if samples.size
        else float("nan")
    )
    return {
        "observed": observed,
        "samples": samples,
        "median": float(np.median(samples)) if samples.size else float("nan"),
        "p05": (
            float(np.percentile(samples, 5))
            if samples.size
            else float("nan")
        ),
        # Share of reshuffles whose drawdown was shallower than the real path.
        # Near zero means the real ordering was implausibly kind.
        "percentile": percentile,
    }


# ---------------------------------------------------------------------------
# Fragility
# ---------------------------------------------------------------------------

def concentration(returns: pd.Series, top_k: int = 5) -> dict:
    """How much of the total profit came from a handful of periods."""
    log_r = np.log1p(np.clip(returns.to_numpy(dtype=float), -0.9999, None))
    total = float(log_r.sum())
    k = int(min(top_k, log_r.size))
    if k == 0:
        return {
            "top_k": 0,
            "top_k_share": float("nan"),
            "total_positive": total > 0,
        }

    best = np.sort(log_r)[-k:]
    share = float(best.sum() / total) if total > 0 else float("nan")
    return {
        "top_k": k,
        "top_k_share": share,
        "top_k_dates": list(returns.nlargest(k).index),
        "total_positive": total > 0,
    }


def sharpe_without_best(
    returns: pd.Series,
    periods_per_year: int,
    k: int = 5,
) -> float:
    """Annualised Sharpe after deleting the k best periods."""
    if returns.size <= k + 5:
        return float("nan")
    trimmed = returns.drop(returns.nlargest(k).index)
    return metrics.sharpe_ratio(trimmed, periods_per_year)


def cost_sensitivity(
    returns: pd.Series,
    periods_per_year: int,
    levels_bps: tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0),
) -> dict:
    """Sharpe after a flat per-period drag, and the drag that kills the edge.

    A per-period cost is a crude stand-in for spread, slippage and commission,
    but it answers the question that matters: how much friction does the claimed
    edge survive?
    """
    mean = float(returns.mean())
    breakeven_bps = float(mean * 10_000.0)
    curve = {
        float(bps): metrics.sharpe_ratio(
            returns - bps / 10_000.0, periods_per_year
        )
        for bps in levels_bps
    }
    return {
        "breakeven_bps": breakeven_bps,
        "curve": curve,
        "gross_sharpe": curve.get(0.0, float("nan")),
    }


def stability(returns: pd.Series, periods_per_year: int) -> dict:
    """Does the edge show up in both halves of the sample, and year to year?"""
    n = returns.size
    half = n // 2
    first = returns.iloc[:half]
    second = returns.iloc[half:]

    sr_first = (
        metrics.sharpe_ratio(first, periods_per_year)
        if first.size > 5
        else float("nan")
    )
    sr_second = (
        metrics.sharpe_ratio(second, periods_per_year)
        if second.size > 5
        else float("nan")
    )

    roll = metrics.rolling_sharpe(returns, periods_per_year).dropna()
    positive_share = float((roll > 0).mean()) if roll.size else float("nan")

    yearly = metrics.calendar_year_returns(returns)
    positive_years = int((yearly > 0).sum())

    decay = float("nan")
    if np.isfinite(sr_first) and np.isfinite(sr_second):
        decay = float(sr_second - sr_first)

    return {
        "sharpe_first_half": sr_first,
        "sharpe_second_half": sr_second,
        "decay": decay,
        "rolling_sharpe": roll,
        "rolling_positive_share": positive_share,
        "yearly_returns": yearly,
        "positive_years": positive_years,
        "total_years": int(yearly.size),
    }


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """User-tunable assumptions behind the tests."""

    rf_annual: float = 0.0
    n_trials: int = 1
    benchmark_sharpe: float = 0.0     # annualised hurdle for PSR
    confidence: float = 0.95
    n_boot: int = 2000
    cost_bps: float = 1.0


def run_all(
    returns: pd.Series,
    periods_per_year: int,
    settings: Settings,
) -> dict:
    """Run every diagnostic once and hand back a single dictionary."""
    ppy = int(periods_per_year)
    n = int(returns.size)
    arr = returns.to_numpy(dtype=float)

    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = (
        float(stats.kurtosis(arr, fisher=False, bias=False))
        if n > 3
        else 3.0
    )

    sr_p = metrics.sharpe_per_period(returns, ppy, settings.rf_annual)
    sr_ann = float(sr_p * np.sqrt(ppy))
    benchmark_p = float(settings.benchmark_sharpe / np.sqrt(ppy))

    se_p = sharpe_standard_error(sr_p, n, skew, kurt)
    psr = probabilistic_sharpe_ratio(sr_p, n, skew, kurt, benchmark_p)
    min_n = min_track_record_length(
        sr_p, skew, kurt, benchmark_p, settings.confidence
    )

    dispersion_ann, dispersion_source = trial_sharpe_dispersion(returns, ppy)
    dsr, dsr_benchmark_p = deflated_sharpe_ratio(
        sr_p, n, skew, kurt, settings.n_trials, dispersion_ann / np.sqrt(ppy)
    )

    return {
        "n": n,
        "periods_per_year": ppy,
        "skew": skew,
        "kurtosis": kurt,
        "excess_kurtosis": kurt - 3.0,
        "sharpe_period": sr_p,
        "sharpe_annual": sr_ann,
        "sharpe_se_period": se_p,
        "sharpe_se_annual": (
            float(se_p * np.sqrt(ppy))
            if np.isfinite(se_p)
            else float("nan")
        ),
        "sharpe_ci95": (
            float(sr_ann - 1.96 * se_p * np.sqrt(ppy)),
            float(sr_ann + 1.96 * se_p * np.sqrt(ppy)),
        ) if np.isfinite(se_p) else (float("nan"), float("nan")),
        "psr": psr,
        "psr_benchmark_annual": settings.benchmark_sharpe,
        "min_track_record_periods": min_n,
        "min_track_record_years": (
            float(min_n / ppy)
            if np.isfinite(min_n)
            else float("inf")
        ),
        "dsr": dsr,
        "dsr_benchmark_annual": float(dsr_benchmark_p * np.sqrt(ppy)),
        "trial_dispersion_annual": dispersion_ann,
        "trial_dispersion_source": dispersion_source,
        "n_trials": int(settings.n_trials),
        "smoothing": smoothing_profile(returns, ppy),
        "bootstrap": bootstrap_sharpe(returns, ppy, settings.n_boot),
        "permutation_dd": permutation_drawdown(returns),
        "concentration": concentration(returns),
        "sharpe_without_best5": sharpe_without_best(returns, ppy, 5),
        "costs": cost_sensitivity(
            returns,
            ppy,
            tuple(sorted(
                {0.0, 1.0, 2.0, 5.0, 10.0, float(settings.cost_bps)}
            )),
        ),
        "stability": stability(returns, ppy),
    }
