"""Worked examples, so the app can be judged without uploading anything.

Each sample is generated from a fixed seed and delivered in a different file
shape — decimal returns, percentage points, an equity curve — so the loader's
detection is exercised as well as the diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Sample:
    key: str
    name: str
    tagline: str
    expectation: str
    suggested_trials: int
    frame: pd.DataFrame


def _bdays(n: int, start: str = "2016-01-04") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _month_ends(n: int, start: str = "2016-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


def trend_follower() -> Sample:
    """A believable systematic strategy: modest Sharpe, real drawdowns."""
    rng = np.random.default_rng(20240517)
    n = 2016  # eight years of business days

    # Volatility clusters, the way real markets do.
    vol = np.empty(n)
    vol[0] = 0.008
    shocks = rng.normal(0.0, 1.0, n)
    for t in range(1, n):
        vol[t] = np.sqrt(
            0.000004
            + 0.05 * (vol[t - 1] * shocks[t - 1]) ** 2
            + 0.93 * vol[t - 1] ** 2
        )

    # Student-t innovations give fat tails without an implausible skew.
    innovation = rng.standard_t(df=8, size=n) / np.sqrt(8 / 6)
    noise = vol * innovation
    noise = (noise - noise.mean()) / noise.std(ddof=1) * (0.12 / np.sqrt(252))

    # Add exactly enough drift to land on an annualised Sharpe of 0.9.
    mu = 0.9 * noise.std(ddof=1) / np.sqrt(252)
    returns = mu + noise

    frame = pd.DataFrame(
        {"date": _bdays(n), "strategy_return": np.round(returns, 6)}
    )
    return Sample(
        key="trend",
        name="Diversified trend follower",
        tagline="Eight years, Sharpe near 0.9, drawdowns you would actually "
                "have to sit through.",
        expectation="Should pass. This is what a credible track record looks "
                    "like.",
        suggested_trials=1,
        frame=frame,
    )


def overfit_search() -> Sample:
    """The winner of a 500-strategy parameter sweep over pure noise."""
    rng = np.random.default_rng(31337)
    n_trials, n = 500, 504  # two years of business days

    candidates = rng.normal(0.0, 0.009, size=(n_trials, n))
    sharpes = candidates.mean(axis=1) / candidates.std(axis=1, ddof=1)
    winner = candidates[int(np.argmax(sharpes))]

    # Delivered as percentage points, the way most spreadsheets export.
    frame = pd.DataFrame({
        "Date": _bdays(n, "2022-01-03"),
        "Return (%)": np.round(winner * 100, 4),
    })
    return Sample(
        key="overfit",
        name="Best of 500 parameter sets",
        tagline="Two years, high Sharpe, and no edge whatsoever — the winner "
                "of a sweep over noise.",
        expectation=(
            "Should fail once the trial count is honest. Every return here is "
            "random; the only thing that made it look good was being picked "
            "from 500 attempts."
        ),
        suggested_trials=500,
        frame=frame,
    )


def smoothed_fund() -> Sample:
    """Monthly marks on illiquid positions: a real edge, wildly overstated."""
    rng = np.random.default_rng(90210)
    n = 96  # eight years of monthly marks

    true_sigma = 0.035
    noise = rng.normal(0.0, 1.0, n + 2)
    noise = (noise - noise.mean()) / noise.std(ddof=1) * true_sigma
    # A genuine but unremarkable edge: a true annualised Sharpe of 1.0.
    true_returns = noise + 1.0 * true_sigma / np.sqrt(12)

    # Reported returns average the last three months' economics: the classic
    # signature of marking illiquid positions to a model.
    weights = np.array([0.60, 0.25, 0.15])
    reported = np.convolve(true_returns, weights, mode="valid")[:n]

    frame = pd.DataFrame({
        "month": _month_ends(n),
        "net_return": np.round(reported, 6),
    })
    return Sample(
        key="smoothed",
        name="Illiquid credit fund",
        tagline="Monthly marks smoothed across three months, inflating the "
                "Sharpe by roughly half.",
        expectation=(
            "Should be flagged for serial correlation. The underlying "
            "strategy is fine; the reported volatility is not."
        ),
        suggested_trials=1,
        frame=frame,
    )


def lookahead_bug() -> Sample:
    """A signal that quietly reads the bar it is supposed to predict."""
    rng = np.random.default_rng(4242)
    n = 1260  # five years of business days

    market = rng.normal(0.0002, 0.011, n)
    # The position is right 85% of the time because it saw the answer.
    correct = rng.random(n) < 0.85
    strategy = np.where(correct, np.abs(market), -np.abs(market)) * 0.55

    equity = 100_000.0 * np.cumprod(1.0 + strategy)
    frame = pd.DataFrame({
        "date": _bdays(n, "2019-01-02"),
        "portfolio_value": np.round(equity, 2),
    })
    return Sample(
        key="lookahead",
        name="Signal with a look-ahead bug",
        tagline="An equity curve so clean it could only have been produced by "
                "knowing the answer.",
        expectation=(
            "Should be rejected on plausibility even though it is "
            "overwhelmingly significant — the point being that significance "
            "and trustworthiness are different questions."
        ),
        suggested_trials=1,
        frame=frame,
    )


BUILDERS = (trend_follower, overfit_search, smoothed_fund, lookahead_bug)


def all_samples() -> list[Sample]:
    return [build() for build in BUILDERS]


def get_sample(key: str) -> Sample:
    for build in BUILDERS:
        sample = build()
        if sample.key == key:
            return sample
    raise KeyError(key)


def example_format() -> pd.DataFrame:
    """The two-column shape the app is looking for, for the empty state."""
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "return": [0.0031, -0.0018, 0.0007, 0.0042],
        }
    )
