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

from . import benchmark as bm
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


def breakeven_trials(
    sr_period: float,
    n: int,
    skew: float,
    kurtosis: float,
    trial_sharpe_std_period: float,
    threshold: float = 0.95,
    ceiling: int = 1_000_000,
) -> int:
    """The largest number of trials this Sharpe would still survive.

    Asking a researcher how many variants they tried invites the flattering
    answer, and the deflated Sharpe ratio is only as honest as that number.
    Inverting the question removes the incentive: the deflated Sharpe falls
    monotonically in the trial count, so there is a unique point where it
    crosses the threshold. Reporting *that* leaves the user to judge whether
    their own search was larger, which is a question they can answer without
    having to grade themselves.

    Returns 0 when the result fails even as a single untested hypothesis.
    """
    def survives(trials: int) -> bool:
        dsr, _ = deflated_sharpe_ratio(
            sr_period, n, skew, kurtosis, trials, trial_sharpe_std_period
        )
        return bool(np.isfinite(dsr) and dsr >= threshold)

    if not survives(1):
        return 0
    if survives(ceiling):
        return ceiling

    low, high = 1, ceiling
    while high - low > 1:
        middle = (low + high) // 2
        if survives(middle):
            low = middle
        else:
            high = middle
    return int(low)


def dsr_curve(
    sr_period: float,
    n: int,
    skew: float,
    kurtosis: float,
    trial_sharpe_std_period: float,
    max_trials: int = 10_000,
) -> pd.DataFrame:
    """Deflated Sharpe as a function of the trial count, for plotting."""
    grid = np.unique(
        np.round(np.geomspace(1, max(2, max_trials), 90)).astype(int)
    )
    rows = [
        {
            "trials": int(trials),
            "dsr": deflated_sharpe_ratio(
                sr_period, n, skew, kurtosis, int(trials),
                trial_sharpe_std_period,
            )[0],
        }
        for trials in grid
    ]
    return pd.DataFrame(rows)


def haircut_sharpe(
    sr_annual: float,
    n: int,
    periods_per_year: int,
    n_trials: int,
) -> dict:
    """Harvey and Liu's (2015) multiple-testing haircut on a Sharpe ratio.

    The observed Sharpe implies a t-statistic; that t-statistic implies a
    p-value; the p-value is corrected for having been the best of ``n_trials``
    attempts, and the corrected p-value is turned back into a Sharpe. The
    result is the Sharpe that carries the same evidential weight *after*
    admitting how hard the researcher looked.

    A Bonferroni correction is used, which is the conservative end of Harvey
    and Liu's three adjustments: it assumes the trials were independent, and
    correlated trials would be penalised less. This is an adjustment for
    selection, not a forecast of future performance.
    """
    if n < 3 or not np.isfinite(sr_annual):
        return {"haircut_sharpe": float("nan"), "haircut": float("nan")}

    years = n / float(periods_per_year)
    t_stat = float(sr_annual * np.sqrt(years))
    if t_stat <= 0:
        return {
            "haircut_sharpe": float(min(sr_annual, 0.0)),
            "haircut": 1.0,
            "t_stat": t_stat,
            "p_value": 1.0,
            "p_adjusted": 1.0,
        }

    p_value = float(2.0 * stats.t.sf(t_stat, df=max(1, n - 1)))
    p_adjusted = float(min(1.0, p_value * max(1, n_trials)))

    # Invert back to a t-statistic, then to a Sharpe on the same horizon.
    t_adjusted = float(stats.t.isf(p_adjusted / 2.0, df=max(1, n - 1)))
    t_adjusted = max(t_adjusted, 0.0)
    adjusted = float(t_adjusted / np.sqrt(years))

    return {
        "haircut_sharpe": adjusted,
        "haircut": float(1.0 - adjusted / sr_annual) if sr_annual > 0 else 1.0,
        "t_stat": t_stat,
        "p_value": p_value,
        "p_adjusted": p_adjusted,
    }


def window_sensitivity(
    returns: pd.Series,
    periods_per_year: int,
    grid: int = 26,
) -> pd.DataFrame:
    """Annualised Sharpe over every start and end date on a coarse grid.

    Nothing else in the battery catches a sample window chosen with hindsight.
    The permutation test reshuffles returns inside a fixed window, which is a
    different question entirely. Recomputing the Sharpe over every sub-window
    shows immediately whether the headline number depends on where the sample
    happens to begin and end.

    Prefix sums make each window O(1), so the whole grid is cheap.
    """
    arr = returns.to_numpy(dtype=float)
    n = arr.size
    index = pd.DatetimeIndex(returns.index)
    minimum = max(20, min(periods_per_year, n // 5))
    if n < minimum * 2:
        return pd.DataFrame(
            columns=["start", "end", "sharpe", "n", "start_i", "end_i"]
        )

    cumulative = np.concatenate([[0.0], np.cumsum(arr)])
    squared = np.concatenate([[0.0], np.cumsum(arr ** 2)])
    scale = np.sqrt(periods_per_year)

    starts = np.unique(np.linspace(0, n - minimum, grid).astype(int))
    ends = np.unique(np.linspace(minimum, n, grid).astype(int))

    rows = []
    for start in starts:
        for end in ends:
            length = int(end - start)
            if length < minimum:
                continue
            total = cumulative[end] - cumulative[start]
            total_sq = squared[end] - squared[start]
            mean = total / length
            variance = (total_sq - length * mean ** 2) / (length - 1)
            sharpe = (
                float(mean / np.sqrt(variance) * scale)
                if variance > 0
                else float("nan")
            )
            rows.append(
                {
                    "start": index[int(start)],
                    "end": index[int(end) - 1],
                    "sharpe": sharpe,
                    "n": length,
                    "start_i": int(start),
                    "end_i": int(end),
                }
            )

    frame = pd.DataFrame(rows)
    return frame.dropna(subset=["sharpe"])


def window_summary(frame: pd.DataFrame, observed: float) -> dict:
    """Condense the window grid into the few numbers worth reporting."""
    if frame.empty:
        return {
            "n_windows": 0,
            "share_positive": float("nan"),
            "worst": float("nan"),
            "best": float("nan"),
            "median": float("nan"),
            "share_above_half": float("nan"),
        }
    values = frame["sharpe"].to_numpy(dtype=float)
    reference = observed / 2.0 if np.isfinite(observed) else 0.0
    return {
        "n_windows": int(values.size),
        "share_positive": float(np.mean(values > 0)),
        "worst": float(np.min(values)),
        "best": float(np.max(values)),
        "median": float(np.median(values)),
        "share_above_half": float(np.mean(values >= reference)),
    }


def trial_sharpe_dispersion(
    returns: pd.Series,
    periods_per_year: int,
) -> tuple[float, str, float]:
    """Estimate the spread of Sharpe ratios across the trials that were run.

    The Deflated Sharpe Ratio needs the variance of the trial Sharpes, which a
    single uploaded series cannot contain. Under the null the correction is
    built on -- every trial worthless -- the trial Sharpes differ only by
    estimation error, so their standard deviation *is* the standard error of a
    full-sample Sharpe estimate. That is the quantity used here.

    An earlier version used the spread of this strategy's own year-by-year
    Sharpe ratios, which is wrong in a way worth recording: yearly estimates
    carry roughly sqrt(years) times the noise of a full-sample estimate, so it
    overstated the dispersion badly and deflated even honest records into
    nothing. The year-by-year figure is still returned, because a spread far
    wider than estimation error hints at a strategy whose behaviour changes
    regime to regime.

    Real trials differ in quality as well as in luck, so genuine dispersion is
    wider than this and the deflation here is the gentler end of the range.
    """
    arr = returns.to_numpy(dtype=float)
    n, ppy = arr.size, int(periods_per_year)

    sd = float(np.std(arr, ddof=1))
    sr_p = float(np.mean(arr) / sd) if sd > 0 else 0.0
    skew = float(stats.skew(arr, bias=False)) if n > 2 else 0.0
    kurt = (
        float(stats.kurtosis(arr, fisher=False, bias=False))
        if n > 3
        else 3.0
    )
    se = sharpe_standard_error(sr_p, n, skew, kurt)
    estimate = float(se * np.sqrt(ppy)) if np.isfinite(se) else 0.5

    blocks = [arr[i:i + ppy] for i in range(0, n, ppy)]
    usable = [b for b in blocks if b.size >= max(6, ppy // 4)]
    yearly = []
    for block in usable:
        block_sd = float(np.std(block, ddof=1))
        if block_sd > 0:
            yearly.append(float(np.mean(block) / block_sd * np.sqrt(ppy)))
    spread = (
        float(np.std(np.asarray(yearly), ddof=1))
        if len(yearly) >= 3
        else float("nan")
    )

    label = (
        "standard error of a full-sample Sharpe estimate, which is how far"
        " apart worthless trials land by luck alone"
    )
    return max(estimate, 1e-6), label, spread


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


def hac_bandwidth(n: int) -> int:
    """Newey and West's (1994) automatic lag choice, 4(T/100)^(2/9)."""
    if n < 8:
        return 1
    return int(max(1, np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def lo_annualisation_factor(
    returns: pd.Series,
    periods_per_year: int,
    max_lag: int | None = None,
) -> tuple[float, int]:
    """Lo's (2002) correct scaling factor, which replaces the naive sqrt(q).

    Positively autocorrelated returns compound their risk faster than the
    square-root rule assumes, so the naive annualised Sharpe is overstated.

    Lo's formula sums over all q-1 lags. For daily data that is 251 sample
    autocorrelations estimated from the same series, most of them noise, so
    the sum is truncated at a Newey-West bandwidth and the remaining lags are
    treated as zero. Bartlett weights taper the retained lags, which keeps the
    variance estimate positive.

    The truncation is deliberately conservative: a series with dependence
    running past the bandwidth is corrected *less* than it should be, so the
    adjusted Sharpe reported here is an upper bound on the honest one. The lag
    count is returned so the app can say what it used.
    """
    q = float(periods_per_year)
    if q <= 1:
        return 1.0, 0

    n = int(returns.size)
    lags = int(max_lag) if max_lag else hac_bandwidth(n)
    lags = int(max(1, min(lags, int(q) - 1, max(1, n // 4))))

    rho = autocorrelations(returns, lags)
    ks = np.arange(1, rho.size + 1)
    taper = 1.0 - ks / (rho.size + 1.0)
    denom = q + 2.0 * float(np.sum(taper * (q - ks) * rho))
    if denom <= 0:
        return float("nan"), lags
    return float(q / np.sqrt(denom)), lags


def smoothing_profile(returns: pd.Series, periods_per_year: int) -> dict:
    """Autocorrelation summary plus the Sharpe it would survive at."""
    rho = autocorrelations(returns, 10)
    q_stat, p_value, lags = ljung_box(returns, 10)
    factor, lo_lags = lo_annualisation_factor(returns, periods_per_year)
    sr_p = metrics.sharpe_per_period(returns, periods_per_year)
    naive = float(sr_p * np.sqrt(periods_per_year))
    adjusted = float(sr_p * factor) if np.isfinite(factor) else float("nan")
    return {
        "rho": rho,
        "rho1": float(rho[0]) if rho.size else 0.0,
        "ljung_box_q": q_stat,
        "ljung_box_p": p_value,
        "ljung_box_lags": lags,
        "lo_lags": lo_lags,
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


def dependence_block_length(returns: pd.Series) -> int:
    """Block length scaled to how far this series' dependence actually runs.

    A fixed n**(1/3) block is right only for a series with no memory. Blocks
    have to be long enough to span the dependence they are meant to preserve,
    so the base rate is multiplied by the integrated autocorrelation time,
    tau = 1 + 2 * sum(rho_k), which is the factor by which serial correlation
    inflates the variance of a sample mean.

    This is a documented heuristic rather than the Politis-White plug-in rule;
    it is transparent, it moves in the right direction with the data, and the
    app lets the user override it.
    """
    n = int(returns.size)
    if n < 20:
        return 1
    rho = autocorrelations(returns, hac_bandwidth(n))
    ks = np.arange(1, rho.size + 1)
    taper = 1.0 - ks / (rho.size + 1.0)
    tau = 1.0 + 2.0 * float(np.sum(taper * rho))
    tau = float(np.clip(tau, 1.0, 25.0))
    base = n ** (1.0 / 3.0)
    return int(max(2, min(round(tau * base), n // 4)))


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

    size = int(block) if block else dependence_block_length(returns)
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

    # A plain percentile interval is biased whenever the bootstrap
    # distribution is not centred on the observed statistic, which for the
    # Sharpe ratio it usually is not. Bias correction shifts the quantiles by
    # the median bias the resamples reveal (Efron's BC interval; the
    # acceleration term of BCa would need a jackknife over every observation,
    # which is not worth the cost here).
    observed = metrics.sharpe_ratio(returns, periods_per_year)
    below = float(np.mean(samples < observed))
    lo_pct, hi_pct = 5.0, 95.0
    z0 = float("nan")
    if 0.0 < below < 1.0:
        z0 = float(stats.norm.ppf(below))
        for target, name in ((0.05, "lo"), (0.95, "hi")):
            z = float(stats.norm.ppf(target))
            corrected = float(stats.norm.cdf(2.0 * z0 + z)) * 100.0
            corrected = float(np.clip(corrected, 0.1, 99.9))
            if name == "lo":
                lo_pct = corrected
            else:
                hi_pct = corrected

    return {
        "samples": samples,
        "block": size,
        "n_boot": int(samples.size),
        "observed": observed,
        "bias_z0": z0,
        "lo_pct": lo_pct,
        "hi_pct": hi_pct,
        "p05": float(np.percentile(samples, lo_pct)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, hi_pct)),
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
    turnover: pd.Series | None = None,
) -> dict:
    """Sharpe after trading friction, and the cost level that kills the edge.

    With a turnover column the cost is charged where it is actually incurred:
    ``net = gross - turnover * cost``. A strategy that trades 5% of the book a
    day and one that trades it twice a day face wildly different bills for the
    same spread, and a flat per-period charge cannot tell them apart.

    Without turnover the cost is a flat drag on every period, which is a crude
    proxy that silently assumes constant trading. The model actually used is
    reported so the number is never read as more precise than it is.
    """
    mean = float(returns.mean())

    if turnover is not None and turnover.size:
        aligned = pd.DataFrame(
            {"r": returns, "turnover": turnover}
        ).dropna()
        rates = aligned["turnover"].clip(lower=0.0)
        stream = aligned["r"]
        mean_turnover = float(rates.mean())
        breakeven = (
            float(stream.mean() / mean_turnover * 10_000.0)
            if mean_turnover > 0
            else float("inf")
        )
        curve = {
            float(bps): metrics.sharpe_ratio(
                stream - rates * bps / 10_000.0, periods_per_year
            )
            for bps in levels_bps
        }
        return {
            "model": "turnover",
            "breakeven_bps": breakeven,
            "curve": curve,
            "gross_sharpe": curve.get(0.0, float("nan")),
            "mean_turnover": mean_turnover,
            "annual_turnover": float(mean_turnover * periods_per_year),
            "n_priced": int(len(aligned)),
        }

    curve = {
        float(bps): metrics.sharpe_ratio(
            returns - bps / 10_000.0, periods_per_year
        )
        for bps in levels_bps
    }
    return {
        "model": "flat",
        "breakeven_bps": float(mean * 10_000.0),
        "curve": curve,
        "gross_sharpe": curve.get(0.0, float("nan")),
        "mean_turnover": float("nan"),
        "annual_turnover": float("nan"),
        "n_priced": int(returns.size),
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
    max_plausible_sharpe: float = 2.5   # set by the asset-class context
    block: int | None = None            # bootstrap block length override


def run_all(
    returns: pd.Series,
    periods_per_year: int,
    settings: Settings,
    turnover: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
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

    dispersion_ann, dispersion_source, yearly_spread = (
        trial_sharpe_dispersion(returns, ppy)
    )
    dispersion_p = dispersion_ann / np.sqrt(ppy)
    dsr, dsr_benchmark_p = deflated_sharpe_ratio(
        sr_p, n, skew, kurt, settings.n_trials, dispersion_p
    )
    survives_to = breakeven_trials(
        sr_p, n, skew, kurt, dispersion_p, settings.confidence
    )
    windows = window_sensitivity(returns, ppy)

    return {
        "attribution": (
            bm.regress(
                returns, benchmark_returns, ppy, settings.rf_annual
            )
            if benchmark_returns is not None
            else None
        ),
        "breakeven_trials": survives_to,
        "dsr_curve": dsr_curve(sr_p, n, skew, kurt, dispersion_p),
        "haircut": haircut_sharpe(sr_ann, n, ppy, settings.n_trials),
        "windows": windows,
        "window_summary": window_summary(windows, sr_ann),
        "max_plausible_sharpe": float(settings.max_plausible_sharpe),
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
        "trial_dispersion_yearly": yearly_spread,
        "n_trials": int(settings.n_trials),
        "smoothing": smoothing_profile(returns, ppy),
        "bootstrap": bootstrap_sharpe(
            returns, ppy, settings.n_boot, settings.block
        ),
        "permutation_dd": permutation_drawdown(returns),
        "concentration": concentration(returns),
        "sharpe_without_best5": sharpe_without_best(returns, ppy, 5),
        "costs": cost_sensitivity(
            returns,
            ppy,
            tuple(sorted(
                {0.0, 1.0, 2.0, 5.0, 10.0, float(settings.cost_bps)}
            )),
            turnover=turnover,
        ),
        "stability": stability(returns, ppy),
    }
