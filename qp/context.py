"""Asset-class context, so a fast book is not failed for being fast.

A Sharpe of 6 is a red flag in a daily equity backtest and unremarkable on a
market-making desk. Judging every upload against one ceiling produces confident
false positives against exactly the strategies most likely to be real, so the
ceiling is a setting the user picks rather than a constant baked into a check.

Costs are given as round-trip friction *per unit of turnover*, paired with how
much of the book a strategy of this kind typically trades in a period. The two
multiply into the per-period drag, which keeps the numbers honest: 5 bps of
spread is not 5 bps a day unless the whole book turns over daily. When the
uploaded file carries a turnover column the assumption is dropped and the real
figures are used instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Context:
    key: str
    label: str
    max_plausible_sharpe: float
    round_trip_bps: float      # cost per unit of turnover
    assumed_turnover: float    # fraction of the book traded per period
    note: str

    @property
    def flat_cost_bps(self) -> float:
        """Per-period drag implied by the two assumptions above."""
        return float(self.round_trip_bps * self.assumed_turnover)


CONTEXTS: tuple[Context, ...] = (
    Context(
        key="equity_daily",
        label="Equity, daily or slower",
        max_plausible_sharpe=2.5,
        round_trip_bps=5.0,
        assumed_turnover=0.1,
        note=(
            "Long-horizon equity strategies. Published factors sit near "
            "0.3–0.6 and good systematic books run 0.8–1.5."
        ),
    ),
    Context(
        key="futures_macro",
        label="Futures or macro",
        max_plausible_sharpe=2.0,
        round_trip_bps=2.0,
        assumed_turnover=0.15,
        note=(
            "Liquid futures are cheap to trade, so the cost bar is low and "
            "the Sharpe bar is too: trend followers live near 0.5–1.0."
        ),
    ),
    Context(
        key="intraday",
        label="Intraday or stat arb",
        max_plausible_sharpe=4.0,
        round_trip_bps=3.0,
        assumed_turnover=1.0,
        note=(
            "Shorter holding periods support higher Sharpes on far smaller "
            "capacity. Costs dominate: check turnover before anything else."
        ),
    ),
    Context(
        key="hft",
        label="High frequency or market making",
        max_plausible_sharpe=10.0,
        round_trip_bps=1.0,
        assumed_turnover=5.0,
        note=(
            "Sharpes in the high single digits are normal here and are not "
            "evidence of a bug. Capacity and queue position are the binding "
            "constraints, and neither is visible in a return series."
        ),
    ),
    Context(
        key="crypto",
        label="Crypto",
        max_plausible_sharpe=3.0,
        round_trip_bps=15.0,
        assumed_turnover=0.2,
        note=(
            "Wide spreads, thin books and exchange risk. Backtests here are "
            "unusually prone to survivorship bias from delisted pairs."
        ),
    ),
    Context(
        key="fund",
        label="Fund or monthly track record",
        max_plausible_sharpe=2.0,
        round_trip_bps=0.0,
        assumed_turnover=0.0,
        note=(
            "Reported net of fees already, so the cost test is turned down. "
            "Watch the smoothing check instead: monthly marks on illiquid "
            "positions are the usual reason a Sharpe here looks too good."
        ),
    ),
)

BY_KEY = {c.key: c for c in CONTEXTS}
DEFAULT = CONTEXTS[0]


def get(key: str) -> Context:
    return BY_KEY.get(key, DEFAULT)
