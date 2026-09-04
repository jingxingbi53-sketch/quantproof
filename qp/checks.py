"""Turn raw diagnostics into a readable verdict.

Each check answers one question about the backtest, returns pass / caution /
fail, and explains what the number means and what to do about it. The scoring
weights say how much each question matters to the headline trust score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diagnostics import Settings
from .loading import LoadedSeries
from .metrics import Performance

PASS, WARN, FAIL, SKIP = "pass", "warn", "fail", "skip"

CATEGORIES = (
    "Statistical significance",
    "Overfitting risk",
    "Fragility",
    "Data quality",
)

PERIOD_NOUN = {
    "Daily": "day",
    "Weekly": "week",
    "Monthly": "month",
    "Quarterly": "quarter",
    "Annual": "year",
}

CATEGORY_ICONS = {
    "Statistical significance": ":material/functions:",
    "Overfitting risk": ":material/travel_explore:",
    "Fragility": ":material/compress:",
    "Data quality": ":material/rule:",
}


@dataclass
class Check:
    key: str
    name: str
    category: str
    status: str
    score: float          # 0..1, how much credit this check gives
    weight: float
    headline: str         # the number, stated plainly
    detail: str           # what the number means for this backtest
    advice: str           # what to do about it

    @property
    def passed(self) -> bool:
        return self.status == PASS


def _ramp(value: float, low: float, high: float) -> float:
    """Linear 0..1 credit between a failing level and a passing one."""
    if not np.isfinite(value):
        return 0.5
    if high == low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _pct(x: float, digits: int = 1) -> str:
    return "n/a" if not np.isfinite(x) else f"{x * 100:.{digits}f}%"


def _num(x: float, digits: int = 2) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:.{digits}f}"


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def check_sample_size(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    n, years = perf.n, perf.years
    if n < 30 or years < 1.0:
        status, score = FAIL, _ramp(min(n / 30.0, years / 1.0), 0.0, 1.0) * 0.4
    elif n < 120 or years < 3.0:
        reached = min(n / 120.0, years / 3.0)
        status, score = WARN, 0.4 + 0.4 * _ramp(reached, 0.3, 1.0)
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(years, 3.0, 10.0)

    return Check(
        key="sample_size",
        name="Sample size",
        category="Statistical significance",
        status=status,
        score=float(score),
        weight=1.0,
        headline=f"{n:,} {loaded.frequency_label.lower()} observations over "
                 f"{years:.1f} years",
        detail=(
            "Every statistic below is estimated from this sample, and short "
            "samples produce wild estimates. A Sharpe ratio measured over one "
            "year has a standard error near 1.0, which means a 'Sharpe 1.0' "
            "strategy and a coin flip are indistinguishable."
        ),
        advice=(
            "Extend the backtest before drawing conclusions."
            if status != PASS
            else "The sample is long enough for the tests below to carry "
                 "weight."
        ),
    )


def check_psr(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    psr = diag["psr"]
    hurdle = cfg.benchmark_sharpe
    lo, hi = diag["sharpe_ci95"]

    if not np.isfinite(psr):
        status, score = SKIP, 0.5
    elif psr >= 0.95:
        status, score = PASS, 1.0
    elif psr >= 0.80:
        status, score = WARN, 0.5 + 0.5 * _ramp(psr, 0.80, 0.95)
    else:
        status, score = FAIL, 0.5 * _ramp(psr, 0.40, 0.80)

    hurdle_text = (
        "beats zero" if hurdle == 0
        else f"beats a Sharpe of {hurdle:.2f}"
    )
    return Check(
        key="psr",
        name="Probabilistic Sharpe ratio",
        category="Statistical significance",
        status=status,
        score=float(score),
        weight=1.5,
        headline=f"PSR = {_num(psr)} that the true Sharpe {hurdle_text}",
        detail=(
            f"The observed annualised Sharpe is {_num(perf.sharpe)}, with a "
            f"95% interval of [{_num(lo)}, {_num(hi)}]. PSR converts that "
            f"into a probability, correcting for skew ({_num(diag['skew'])}) "
            f"and fat tails (excess kurtosis "
            f"{_num(diag['excess_kurtosis'])}), both of which make a Sharpe "
            f"less reliable than the textbook formula assumes."
        ),
        advice=(
            "The edge is statistically credible on its own terms; the "
            "remaining risk is selection, not noise."
            if status == PASS
            else "Treat the headline Sharpe as one draw from a wide "
                 "distribution, not a fact."
        ),
    )


def check_min_trl(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    need_periods = diag["min_track_record_periods"]
    need_years = diag["min_track_record_years"]
    ppy = diag["periods_per_year"]

    if not np.isfinite(need_periods):
        return Check(
            key="min_trl",
            name="Minimum track record length",
            category="Statistical significance",
            status=FAIL,
            score=0.0,
            weight=1.0,
            headline="No track record length would make this significant",
            detail=(
                "The observed Sharpe does not exceed the hurdle, so more "
                "history cannot rescue it. Statistical significance requires "
                "an edge to be there in the first place."
            ),
            advice="Revisit the strategy itself rather than the length of the "
                   "test.",
        )

    ratio = perf.n / need_periods
    if ratio >= 1.0:
        status, score = PASS, 1.0
    elif ratio >= 0.5:
        status, score = WARN, 0.4 + 0.6 * _ramp(ratio, 0.5, 1.0)
    else:
        status, score = FAIL, 0.4 * _ramp(ratio, 0.0, 0.5)

    return Check(
        key="min_trl",
        name="Minimum track record length",
        category="Statistical significance",
        status=status,
        score=float(score),
        weight=1.0,
        headline=(
            f"Needs {need_years:.1f} years at this Sharpe; you have "
            f"{perf.years:.1f}"
        ),
        detail=(
            f"At a Sharpe of {_num(perf.sharpe)} with this return "
            f"distribution, {need_periods:,.0f} "
            f"{loaded.frequency_label.lower()} observations ({need_years:.1f} "
            f"years) are required before the result clears "
            f"{cfg.confidence:.0%} confidence. The sample holds {perf.n:,} "
            f"({ppy} per year)."
        ),
        advice=(
            "The record is long enough to support the claim."
            if status == PASS
            else f"Roughly {max(0.0, need_years - perf.years):.1f} more years "
                 f"of the same performance would be needed."
        ),
    )


# ---------------------------------------------------------------------------
# Overfitting risk
# ---------------------------------------------------------------------------

def check_dsr(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    dsr = diag["dsr"]
    n_trials = diag["n_trials"]
    hurdle = diag["dsr_benchmark_annual"]
    survives = diag["breakeven_trials"]

    survives_text = (
        "This record would stop being distinguishable from search at about "
        f"{survives:,} trials, so the question is simply whether your own "
        "search was wider than that."
        if survives > 0
        else "This record does not clear the bar even as a single untested "
             "hypothesis, so the trial count is not what is wrong with it."
    )

    if not np.isfinite(dsr):
        status, score = SKIP, 0.5
    elif dsr >= 0.95:
        status, score = PASS, 1.0
    elif dsr >= 0.60:
        status, score = WARN, 0.45 + 0.55 * _ramp(dsr, 0.60, 0.95)
    else:
        status, score = FAIL, 0.45 * _ramp(dsr, 0.10, 0.60)

    trials_text = (
        "You reported a single trial, so no deflation was applied — if you "
        "tested more variants than that, raise the trial count and re-read "
        "this number."
        if n_trials <= 1
        else f"Across {n_trials:,} trials, luck alone would be expected to "
             f"produce a best Sharpe of about {_num(hurdle)}."
    )

    return Check(
        key="dsr",
        name="Deflated Sharpe ratio",
        category="Overfitting risk",
        status=status,
        score=float(score),
        weight=2.0,
        headline=f"Survives up to {survives:,} trials; you reported "
                 f"{n_trials:,}",
        detail=(
            f"This is the single most important number on the page. Search "
            f"enough parameter combinations and one of them will look "
            f"excellent by chance. The Deflated Sharpe Ratio raises the bar "
            f"to the Sharpe that pure search would have produced, then asks "
            f"whether yours still clears it. {trials_text} {survives_text} "
            f"Trial Sharpe dispersion was taken as "
            f"{_num(diag['trial_dispersion_annual'])}: the "
            f"{diag['trial_dispersion_source']}."
        ),
        advice=(
            "The result survives the multiple-testing correction."
            if status == PASS
            else "Count every variant you tried — every parameter grid, every "
                 "universe, every stop-loss level — and enter the honest "
                 "number. Then hold out data you have never looked at."
        ),
    )


def check_sharpe_prior(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    sr = perf.sharpe
    ceiling = float(cfg.max_plausible_sharpe)
    hard = ceiling * 1.6

    if not np.isfinite(sr):
        status, score = SKIP, 0.5
    elif sr > hard:
        status, score = FAIL, 0.0
    elif sr > ceiling:
        status, score = WARN, 0.35 + 0.3 * _ramp(hard - sr, 0.0, hard - ceiling)
    elif sr < 0:
        status, score = FAIL, 0.0
    else:
        status, score = PASS, 1.0

    return Check(
        key="sharpe_prior",
        name="Sharpe plausibility",
        category="Overfitting risk",
        status=status,
        score=float(score),
        weight=1.5,
        headline=f"Annualised Sharpe of {_num(sr)}",
        detail=(
            f"Judged against a ceiling of {_num(ceiling)} for this asset "
            f"class and horizon. Published equity factors sit near 0.3–0.6 "
            f"and good systematic strategies run 0.8–1.5, but a "
            f"capacity-constrained market-making book legitimately runs far "
            f"higher, which is why this ceiling is a setting rather than a "
            f"constant. Set the context in the sidebar if it is wrong: a "
            f"real edge should not be failed for being fast."
        ),
        advice=(
            "The magnitude is in a believable range."
            if status == PASS
            else "Audit for look-ahead bias first: signals using same-bar "
                 "closes, survivorship-free universes, and fills at prices "
                 "you could not actually have traded."
        ),
    )


def check_calmar(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    calmar = perf.calmar

    if not np.isfinite(calmar) or perf.max_drawdown >= 0:
        status, score = SKIP, 0.5
    elif calmar > 10.0:
        status, score = FAIL, 0.0
    elif calmar > 4.0:
        status, score = FAIL, 0.15 * _ramp(10.0 - calmar, 0.0, 6.0)
    elif calmar > 2.0:
        status, score = WARN, 0.45 + 0.35 * _ramp(4.0 - calmar, 0.0, 2.0)
    elif calmar <= 0:
        status, score = FAIL, 0.0
    else:
        status, score = PASS, 1.0

    return Check(
        key="calmar",
        name="Return versus drawdown",
        category="Overfitting risk",
        status=status,
        score=float(score),
        weight=1.0,
        headline=(
            f"Calmar {_num(calmar)}: {_pct(perf.cagr)} a year against a "
            f"{_pct(perf.max_drawdown)} worst loss"
        ),
        detail=(
            "Calmar is annual return divided by the worst peak-to-trough "
            "loss. Long-running systematic funds live between 0.3 and 1.0; "
            "above 3 belongs to a small number of capacity-limited desks. "
            "Very high values almost always mean the backtest avoided the "
            "losses rather than survived them, which is what a look-ahead bug "
            "looks like from the outside."
        ),
        advice=(
            "The reward-to-pain ratio is in a range real capital achieves."
            if status == PASS
            else "Check the exact timestamp your signal is computed on versus "
                 "the price it trades at. An off-by-one bar is the single "
                 "most common cause of a curve this clean."
        ),
    )


def check_window_sensitivity(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    summary = diag["window_summary"]
    share = summary["share_above_half"]
    worst, best = summary["worst"], summary["best"]

    if not summary["n_windows"] or not np.isfinite(share):
        status, score = SKIP, 0.5
    elif share < 0.45:
        status, score = FAIL, 0.3 * _ramp(share, 0.0, 0.45)
    elif share < 0.70:
        status, score = WARN, 0.3 + 0.5 * _ramp(share, 0.45, 0.70)
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(share, 0.70, 0.95)

    return Check(
        key="window_sensitivity",
        name="Sample window sensitivity",
        category="Overfitting risk",
        status=status,
        score=float(score),
        weight=1.25,
        headline=(
            f"Across {summary['n_windows']:,} sub-windows the Sharpe ranges "
            f"from {_num(worst)} to {_num(best)}"
        ),
        detail=(
            f"Recomputing the Sharpe over every start and end date shows "
            f"whether the headline number depends on where the sample happens "
            f"to begin. {_pct(share, 0)} of sub-windows keep at least half of "
            f"the reported {_num(perf.sharpe)}, and the median sub-window "
            f"gives {_num(summary['median'])}. Nothing else in this battery "
            f"catches a date range chosen with hindsight: the reshuffle test "
            f"varies the order of returns inside a fixed window, which is a "
            f"different question."
        ),
        advice=(
            "The result does not depend on where the sample starts or ends."
            if status == PASS
            else "Check whether the start date was chosen after seeing the "
                 "data. A result that needs one particular window is a "
                 "statement about that window, not about the strategy."
        ),
    )


def check_alpha(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    attribution = diag.get("attribution")

    if attribution is None:
        return Check(
            key="alpha",
            name="Alpha versus the benchmark",
            category="Overfitting risk",
            status=SKIP,
            score=0.5,
            weight=1.75,
            headline="No benchmark supplied",
            detail=(
                "Without a benchmark this battery cannot tell an edge from "
                "leverage. A Sharpe of 1.2 that is 0.8 beta to the equity "
                "market will pass every other test on this page, because "
                "every other test only looks at the return series in "
                "isolation. This is the largest blind spot in the analysis."
            ),
            advice=(
                "Add a benchmark column, or upload a second file with the "
                "benchmark's returns, and this check will separate the two."
            ),
        )

    t_stat = attribution.alpha_t
    beta, r2 = attribution.beta, attribution.r_squared

    if not np.isfinite(t_stat):
        status, score = SKIP, 0.5
    elif t_stat >= 2.5:
        status, score = PASS, 0.85 + 0.15 * _ramp(t_stat, 2.5, 4.0)
    elif t_stat >= 1.65:
        status, score = WARN, 0.4 + 0.45 * _ramp(t_stat, 1.65, 2.5)
    else:
        status, score = FAIL, 0.4 * _ramp(t_stat, 0.0, 1.65)

    return Check(
        key="alpha",
        name="Alpha versus the benchmark",
        category="Overfitting risk",
        status=status,
        score=float(score),
        weight=1.75,
        headline=(
            f"Alpha {_pct(attribution.alpha_annual)} a year, t = "
            f"{_num(t_stat)}, beta {_num(beta)}"
        ),
        detail=(
            f"Regressing the strategy on the benchmark leaves "
            f"{_pct(attribution.alpha_annual)} a year of alpha with a "
            f"Newey-West t-statistic of {_num(t_stat)} at "
            f"{attribution.nw_lags} lags, which corrects for the "
            f"autocorrelation that would otherwise overstate significance. "
            f"The benchmark explains {_pct(r2, 0)} of the variance. Hedging "
            f"it out takes the Sharpe from "
            f"{_num(attribution.strategy_sharpe)} to "
            f"{_num(attribution.hedged_sharpe)} — that hedged number is the "
            f"one worth paying for."
        ),
        advice=(
            "The edge survives once the benchmark is taken away."
            if status == PASS
            else "Most of this return is the benchmark. Compare the hedged "
                 "Sharpe against simply holding the benchmark at the same "
                 "volatility, which is cheaper and needs no research."
        ),
    )


def check_drawdown_plausibility(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    perm = diag["permutation_dd"]
    pct = perm.get("percentile", float("nan"))
    observed = perm.get("observed", float("nan"))
    median = perm.get("median", float("nan"))

    if not np.isfinite(pct):
        status, score = SKIP, 0.5
    elif pct < 0.02:
        status, score = FAIL, 0.2 * _ramp(pct, 0.0, 0.02)
    elif pct < 0.10:
        status, score = WARN, 0.2 + 0.5 * _ramp(pct, 0.02, 0.10)
    else:
        status, score = PASS, 0.7 + 0.3 * _ramp(pct, 0.10, 0.35)

    return Check(
        key="drawdown_plausibility",
        name="Drawdown plausibility",
        category="Overfitting risk",
        status=status,
        score=float(score),
        weight=1.0,
        headline=(
            f"Max drawdown {_pct(observed)} vs {_pct(median)} for a typical "
            f"reshuffle"
        ),
        detail=(
            f"Reshuffling the same returns into a random order keeps every "
            f"gain and loss but destroys their sequence. Comparing the real "
            f"path to those reshuffles isolates how much of the smooth ride "
            f"came from the ordering. Only {_pct(pct)} of reshuffles produced "
            f"a drawdown this shallow."
        ),
        advice=(
            "The path looks like a plausible ordering of its own returns."
            if status == PASS
            else "An equity curve far smoother than any reordering of it "
                 "usually means losses were placed where they could not hurt "
                 "— the classic fingerprint of a look-ahead bug or a curve "
                 "fitted to the sample."
        ),
    )


# ---------------------------------------------------------------------------
# Fragility
# ---------------------------------------------------------------------------

def check_autocorrelation(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    sm = diag["smoothing"]
    rho1, p_value = sm["rho1"], sm["ljung_box_p"]
    naive, adjusted = sm["naive_sharpe"], sm["adjusted_sharpe"]
    serial = np.isfinite(p_value) and p_value < 0.05

    if rho1 > 0.35 or (serial and rho1 > 0.20):
        status, score = FAIL, 0.3 * _ramp(0.5 - rho1, 0.0, 0.3)
    elif rho1 > 0.15 or serial:
        status, score = WARN, 0.3 + 0.5 * _ramp(0.35 - rho1, 0.0, 0.20)
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(0.15 - abs(rho1), 0.0, 0.15)

    return Check(
        key="autocorrelation",
        name="Return smoothing",
        category="Fragility",
        status=status,
        score=float(score),
        weight=1.5,
        headline=f"Lag-1 autocorrelation {_num(rho1)} (Ljung-Box p = "
                 f"{_num(p_value, 3)})",
        detail=(
            f"Independent returns are the assumption behind the "
            f"square-root-of-time scaling that produced the headline Sharpe "
            f"of {_num(naive)}. Correcting for the serial correlation "
            f"actually present gives {_num(adjusted)}. Positive "
            f"autocorrelation typically comes from stale or model-marked "
            f"prices, overlapping positions, or returns that were smoothed "
            f"before they reached the file."
        ),
        advice=(
            "Serial dependence is mild enough that the standard annualisation "
            "holds."
            if status == PASS
            else "Mark positions to genuinely tradeable prices and re-run. "
                 "Report the corrected Sharpe, not the naive one."
        ),
    )


def check_concentration(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    conc = diag["concentration"]
    share, k = conc["top_k_share"], conc["top_k"]
    trimmed = diag["sharpe_without_best5"]

    if not np.isfinite(share):
        status, score = SKIP, 0.5
    elif share > 0.60:
        status, score = FAIL, 0.25 * _ramp(1.0 - share, 0.0, 0.4)
    elif share > 0.30:
        status, score = WARN, 0.25 + 0.55 * _ramp(0.60 - share, 0.0, 0.30)
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(0.30 - share, 0.0, 0.25)

    return Check(
        key="concentration",
        name="Profit concentration",
        category="Fragility",
        status=status,
        score=float(score),
        weight=1.25,
        headline=f"Top {k} periods produced {_pct(share)} of the total profit",
        detail=(
            f"Removing just those {k} periods takes the annualised Sharpe "
            f"from {_num(perf.sharpe)} to {_num(trimmed)}. When a handful of "
            f"observations carry the result, the strategy is a bet on rare "
            f"events being repeatable, and the sample contains almost no "
            f"evidence that they are."
        ),
        advice=(
            "Profit is spread across the sample rather than resting on a few "
            "periods."
            if status == PASS
            else "Check whether those periods are real, tradeable moves or "
                 "data artefacts — bad ticks, unadjusted splits, and "
                 "delisting prices all show up here."
        ),
    )


def check_tails(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    skew, ex_kurt = diag["skew"], diag["excess_kurtosis"]

    if skew < -1.0 and ex_kurt > 5.0:
        status, score = FAIL, 0.2
    elif skew < -0.5 or ex_kurt > 3.0:
        status, score = WARN, 0.45 + 0.35 * _ramp(-skew, 1.0, 0.0)
    else:
        status, score = PASS, 1.0

    return Check(
        key="tails",
        name="Tail shape",
        category="Fragility",
        status=status,
        score=float(score),
        weight=0.75,
        headline=f"Skew {_num(skew)}, excess kurtosis {_num(ex_kurt)}",
        detail=(
            f"Worst single period {_pct(perf.worst_period)}; the average of "
            f"the worst 5% of periods is {_pct(perf.cvar_95)}. Negative skew "
            f"with fat tails is the signature of strategies that earn a small "
            f"premium most of the time and give it back at once — short "
            f"volatility, carry, and mean reversion into falling prices all "
            f"share it. The Sharpe ratio, which only sees the first two "
            f"moments, cannot express that risk."
        ),
        advice=(
            "The return distribution holds no unusual tail risk."
            if status == PASS
            else "Size the strategy on tail loss rather than volatility, and "
                 "check the backtest covers at least one genuine stress "
                 "period."
        ),
    )


def check_costs(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    costs = diag["costs"]
    gross = costs["gross_sharpe"]
    assumed = float(cfg.cost_bps)
    # run_all always prices the configured level, so it is in the curve.
    net = costs["curve"][assumed]

    if costs["model"] == "turnover":
        model_text = (
            f"Costs are charged against the turnover column, so a period that "
            f"traded nothing pays nothing. Average turnover is "
            f"{_pct(costs['mean_turnover'])} per "
            f"{PERIOD_NOUN.get(loaded.frequency_label, 'period')}, or "
            f"{_num(costs['annual_turnover'])}x the book a year."
        )
    else:
        model_text = (
            "With no turnover column the cost is a flat drag on every period, "
            "which assumes constant trading and is only a rough proxy. Add a "
            "turnover column and this becomes a real estimate."
        )

    breakeven = costs["breakeven_bps"]
    usable = np.isfinite(gross) and gross > 0
    retained = float(net / gross) if usable else float("nan")

    if not np.isfinite(net) or net <= 0:
        status, score = FAIL, 0.0
    elif not np.isfinite(retained):
        status, score = SKIP, 0.5
    elif retained < 0.5:
        status, score = WARN, 0.35 + 0.45 * _ramp(retained, 0.0, 0.5)
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(retained, 0.5, 0.9)

    return Check(
        key="costs",
        name="Cost sensitivity",
        category="Fragility",
        status=status,
        score=float(score),
        weight=1.25,
        headline=f"At {assumed:.1f} bps per period, Sharpe {_num(gross)} "
                 f"becomes {_num(net)}",
        detail=(
            f"{model_text} The strategy breaks even at {breakeven:.2f} bps of "
            f"cost per {PERIOD_NOUN.get(loaded.frequency_label, 'period')} — "
            f"that is the entire budget available for spread, slippage, "
            f"commission, borrow and market impact. Most backtests are gross "
            f"of all of it."
        ),
        advice=(
            "The edge is large relative to plausible trading friction."
            if status == PASS
            else "Model costs inside the backtest rather than subtracting "
                 "them afterwards, and measure turnover so the assumption can "
                 "be checked."
        ),
    )


def check_stability(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    stab = diag["stability"]
    first, second = stab["sharpe_first_half"], stab["sharpe_second_half"]

    if not (np.isfinite(first) and np.isfinite(second)):
        status, score = SKIP, 0.5
    elif first > 0 and second <= 0:
        status, score = FAIL, 0.1
    elif first > 0 and second < 0.4 * first:
        status, score = WARN, 0.35 + 0.4 * _ramp(second / first, 0.0, 0.4)
    elif second <= 0 and first <= 0:
        status, score = FAIL, 0.0
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(min(second, first), 0.0, 1.0)

    return Check(
        key="stability",
        name="Sub-period stability",
        category="Fragility",
        status=status,
        score=float(score),
        weight=1.25,
        headline=f"Sharpe {_num(first)} in the first half, {_num(second)} in "
                 f"the second",
        detail=(
            "Splitting the sample in two is the cheapest out-of-sample test "
            "available. A real edge shows up in both halves with similar sign "
            "and rough magnitude. An edge that lives entirely in one half was "
            "probably fitted to that half, or was arbitraged away once enough "
            "people found it."
        ),
        advice=(
            "The edge is present in both halves of the sample."
            if status == PASS
            else "Re-fit on the first half only and evaluate on the second "
                 "without touching a single parameter afterwards."
        ),
    )


def check_rolling_consistency(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    stab = diag["stability"]
    share = stab["rolling_positive_share"]
    positive_years, total_years = stab["positive_years"], stab["total_years"]

    if not np.isfinite(share):
        status, score = SKIP, 0.5
    elif share < 0.45:
        status, score = FAIL, 0.3 * _ramp(share, 0.0, 0.45)
    elif share < 0.65:
        status, score = WARN, 0.3 + 0.5 * _ramp(share, 0.45, 0.65)
    else:
        status, score = PASS, 0.8 + 0.2 * _ramp(share, 0.65, 0.90)

    return Check(
        key="rolling_consistency",
        name="Rolling consistency",
        category="Fragility",
        status=status,
        score=float(score),
        weight=1.0,
        headline=f"{_pct(share, 0)} of rolling one-year windows had a "
                 f"positive Sharpe",
        detail=(
            f"{positive_years} of {total_years} calendar years were "
            f"profitable. The longest stretch below a prior peak lasted "
            f"{perf.longest_drawdown_days:,} days, and the strategy spent "
            f"{_pct(perf.time_underwater, 0)} of its life underwater. That is "
            f"the number an investor actually experiences, and the one that "
            f"decides whether a strategy gets shut down before it recovers."
        ),
        advice=(
            "Performance recurs across windows rather than arriving in one "
            "burst."
            if status == PASS
            else "Ask honestly whether you would keep funding this through "
                 "its worst stretch. If not, the backtest result is "
                 "unreachable in practice."
        ),
    )


# ---------------------------------------------------------------------------
# Data quality
# ---------------------------------------------------------------------------

def check_data_hygiene(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> Check:
    hy = loaded.hygiene
    flags: list[str] = []
    penalty = 0.0

    if hy.zero_fraction > 0.25:
        flags.append(f"{_pct(hy.zero_fraction, 0)} of periods are exactly zero")
        penalty += 0.30
    elif hy.zero_fraction > 0.10:
        flags.append(f"{_pct(hy.zero_fraction, 0)} of periods are exactly zero")
        penalty += 0.12

    if hy.longest_repeat_run >= 5:
        flags.append(f"{hy.longest_repeat_run} identical values in a row")
        penalty += 0.25
    elif hy.longest_repeat_run >= 3:
        penalty += 0.08

    if hy.unique_fraction < 0.5:
        flags.append(
            f"only {_pct(hy.unique_fraction, 0)} of values are distinct"
        )
        penalty += 0.20

    if hy.duplicate_dates:
        flags.append(f"{hy.duplicate_dates} duplicate timestamps")
        penalty += 0.15

    if hy.calendar_gaps:
        flags.append(f"{hy.calendar_gaps} unexplained gaps in the calendar")
        penalty += 0.10

    if hy.missing_values:
        flags.append(f"{hy.missing_values} missing values")
        penalty += 0.08

    if loaded.date_column is None:
        flags.append("no date column, so a synthetic calendar was used")
        penalty += 0.10

    score = float(np.clip(1.0 - penalty, 0.0, 1.0))
    status = PASS if score >= 0.85 else (WARN if score >= 0.55 else FAIL)

    return Check(
        key="data_hygiene",
        name="Data hygiene",
        category="Data quality",
        status=status,
        score=score,
        weight=1.0,
        headline=(
            "No structural problems in the file" if not flags
            else "; ".join(flags).capitalize()
        ),
        detail=(
            "Runs of identical values and clusters of exact zeros usually "
            "mean prices were carried forward rather than observed, which "
            "flatters volatility and inflates the Sharpe. Duplicate "
            "timestamps and calendar gaps mean the sample is not what its "
            "date range claims."
        ),
        advice=(
            "The file is structurally sound."
            if status == PASS
            else "Trace each flagged value back to the source data before "
                 "trusting anything computed from it."
        ),
    )


CHECK_FUNCS = (
    check_sample_size,
    check_psr,
    check_min_trl,
    check_dsr,
    check_sharpe_prior,
    check_calmar,
    check_alpha,
    check_window_sensitivity,
    check_drawdown_plausibility,
    check_autocorrelation,
    check_concentration,
    check_tails,
    check_costs,
    check_stability,
    check_rolling_consistency,
    check_data_hygiene,
)


def run_checks(
    perf: Performance,
    diag: dict,
    loaded: LoadedSeries,
    cfg: Settings,
) -> list[Check]:
    """Run the full battery in a fixed, reportable order."""
    return [fn(perf, diag, loaded, cfg) for fn in CHECK_FUNCS]
