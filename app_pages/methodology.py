"""What every test on the analysis page computes, and what it cannot."""

import pandas as pd
import streamlit as st

from qp import checks, diagnostics, loading, metrics, samples


@st.cache_data(show_spinner=False)
def check_catalog() -> pd.DataFrame:
    """Read the live battery so this page can never drift from the code."""
    sample = samples.trend_follower()
    loaded = loading.build_series(
        sample.frame, value_col="strategy_return", date_col="date"
    )
    cfg = diagnostics.Settings()
    perf = metrics.compute_performance(
        loaded.returns, loaded.periods_per_year, cfg.rf_annual
    )
    diag = diagnostics.run_all(loaded.returns, loaded.periods_per_year, cfg)
    rows = [
        {
            "Check": check.name,
            "Category": check.category,
            "Weight": check.weight,
        }
        for check in checks.run_checks(perf, diag, loaded, cfg)
    ]
    frame = pd.DataFrame(rows)
    frame["Share of score"] = frame["Weight"] / frame["Weight"].sum()
    return frame


st.title("Methodology")
st.markdown(
    "QuantProof answers one question: **how much of this track record is "
    "evidence, and how much is the shape noise takes when you look at it long "
    "enough?** Everything below is computed from the return series alone, "
    "with no access to the strategy, the universe, or the code that produced "
    "it. Where that matters, it is said plainly."
)

st.subheader("Notation", icon=":material/functions:")
st.markdown(
    r"""
Returns are $r_1 \dots r_n$ sampled $q$ times a year — 252 for daily data,
12 for monthly. $\hat{SR}$ is the Sharpe ratio **in the sampling frequency**,
$\hat{SR} = (\bar{r} - r_f) / \sigma$, and the annualised figure shown on the
analysis page is $\hat{SR}\sqrt{q}$. $\gamma_3$ is skewness and $\gamma_4$ is
kurtosis (3 for a normal distribution, not excess).
"""
)

st.subheader("The tests", icon=":material/science:")

with st.expander(
    "Sharpe ratio standard error", icon=":material/show_chart:", expanded=True
):
    st.markdown(
        r"""
A Sharpe ratio is an estimate, and estimates have error bars. Under normally
distributed returns the standard error is roughly $\sqrt{(1 + \hat{SR}^2/2)/n}$;
Lo (2002) shows that skew and fat tails make it worse:

$$
\widehat{\mathrm{SE}}(\hat{SR}) =
\sqrt{\frac{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4}\hat{SR}^2}{n - 1}}
$$

**How to read it.** One year of daily data gives a standard error near 1.0 on
the annualised Sharpe. A one-year backtest showing Sharpe 1.5 is entirely
consistent with a true Sharpe of zero. This single fact invalidates more
backtests than any other.
"""
    )

with st.expander(
    "Probabilistic Sharpe ratio (PSR)", icon=":material/percent:"
):
    st.markdown(
        r"""
Bailey and Lopez de Prado (2012) turn that standard error into the probability
that the true Sharpe exceeds a benchmark $SR^{*}$:

$$
\mathrm{PSR}(SR^{*}) = \Phi\!\left(
\frac{(\hat{SR} - SR^{*})\sqrt{n-1}}
{\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4}\hat{SR}^2}}
\right)
$$

**How to read it.** PSR above 0.95 means the edge is unlikely to be sampling
noise. It says nothing about whether you went looking for it — that is the
next test.

**Related: minimum track record length.** Inverting the same expression gives
the sample size at which the result would become significant:

$$
n^{*} = 1 + \left(1 - \gamma_3 \hat{SR} +
\frac{\gamma_4 - 1}{4}\hat{SR}^2\right)
\left(\frac{Z_\alpha}{\hat{SR} - SR^{*}}\right)^{2}
$$
"""
    )

with st.expander(
    "Deflated Sharpe ratio (DSR)", icon=":material/travel_explore:"
):
    st.markdown(
        r"""
Try $N$ strategies on the same data and the best one looks good whether or not
any of them work. Bailey and Lopez de Prado (2014) compute the Sharpe that
search alone would be expected to produce:

$$
SR^{*}_{0} = \sqrt{\hat{V}}\left[(1-\gamma)\,
\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,
\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right]
$$

where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant and $\hat{V}$ is
the variance of the Sharpe ratios across trials. The Deflated Sharpe Ratio is
then $\mathrm{PSR}(SR^{*}_{0})$ — the same probability, measured against a
much higher bar.

**The honest caveat.** $\hat{V}$ cannot be recovered from a single uploaded
series. QuantProof estimates it from the spread of the strategy's own
year-by-year Sharpe ratios when there are at least three years, and otherwise
falls back to the standard error of the Sharpe estimate, which is the
conservative floor. The trial count $N$ comes from you, and it is the input
people understate most. Counting only the runs you saved, rather than every
parameter you swept, defeats the entire correction.
"""
    )

with st.expander(
    "How much search a result survives", icon=":material/search_insights:"
):
    st.markdown(
        r"""
The deflated Sharpe ratio needs a trial count, and asking a researcher for one
invites the flattering answer. Since $\mathrm{DSR}$ falls monotonically as $N$
rises, the question can be inverted: there is a unique $N^{*}$ at which it
crosses the confidence threshold.

Reporting $N^{*}$ instead of demanding $N$ changes the incentive. Nobody has to
grade their own search; they only have to judge whether it was wider than the
number on screen, which is a question they can answer honestly.

**How to read it.** "Survives up to 40 trials" means that if you fitted more
than about forty variants on this data, the result is no longer distinguishable
from the best of that many random attempts.
"""
    )

with st.expander(
    "Multiple-testing haircut", icon=":material/content_cut:"
):
    st.markdown(
        r"""
Harvey and Liu (2015) run the same correction through the p-value rather than
the Sharpe. The observed Sharpe implies a t-statistic, that implies a p-value,
the p-value is adjusted for having been the best of $N$ attempts, and the
adjusted p-value is turned back into a Sharpe:

$$
t = \hat{SR}\sqrt{T}, \qquad
p_{\text{adj}} = \min(1,\, p N), \qquad
SR_{\text{haircut}} = \frac{t(p_{\text{adj}})}{\sqrt{T}}
$$

A Bonferroni adjustment is used, the most conservative of the three Harvey and
Liu describe: it assumes the trials were independent, and correlated trials
would be penalised less.

**How to read it.** This is an adjustment for selection, **not** a forecast. It
says what the evidence is worth after admitting how hard you looked, not what
the strategy will earn next year.
"""
    )

with st.expander("Alpha versus beta", icon=":material/call_split:"):
    st.markdown(
        r"""
Every other test on this page looks at the return series in isolation, and none
of them can tell an edge from leverage. A strategy that is 0.8 beta to the
equity market passes all of them, because holding the market really does
produce a positive Sharpe — just not one anybody should pay you for.

Given a benchmark, QuantProof runs

$$
r_t - r_f = \alpha + \beta\,(b_t - r_f) + \varepsilon_t
$$

with Newey-West standard errors at $\lfloor 4(T/100)^{2/9} \rfloor$ lags, so
autocorrelation cannot overstate the significance of $\alpha$. It then reports
the **hedged** stream $r_t - \beta b_t$ and its Sharpe, which is what a desk
would actually hold after shorting the benchmark against the position.

**How to read it.** The hedged Sharpe is the number worth paying for. If it is
near zero, the benchmark was the strategy.
"""
    )

with st.expander("Sample window sensitivity", icon=":material/crop_free:"):
    st.markdown(
        """
The Sharpe ratio is recomputed over every start and end date on a grid, using
prefix sums so each window costs almost nothing. The result is a heatmap of how
the headline number varies with the sample chosen.

**How to read it.** A real edge shows up over most windows. A result that lives
in one corner of the grid is a statement about that window, not about the
strategy. This is the only test here that catches a date range picked with
hindsight: the reshuffle test varies the *order* of returns inside a fixed
window, which is a different question.
"""
    )

with st.expander("Trading costs", icon=":material/receipt_long:"):
    st.markdown(
        """
With a turnover column, cost is charged where it is incurred:
`net = gross - turnover x cost`. A strategy that trades 5% of the book a day
and one that trades it twice a day face very different bills for the same
spread, and a flat per-period charge cannot tell them apart.

Without turnover, the cost is a flat drag on every period — the round-trip cost
multiplied by an assumed turnover taken from the asset-class setting. That is a
proxy, and it is labelled as one wherever it appears.

**How to read it.** The breakeven cost is the entire budget available for
spread, slippage, commission, borrow and market impact. Most backtests are
gross of all of it.
"""
    )

with st.expander(
    "Serial correlation and smoothing", icon=":material/waves:"
):
    st.markdown(
        r"""
Annualising by $\sqrt{q}$ assumes returns are independent. When they are not,
Lo (2002) gives the correct scaling:

$$
\hat{SR}_{\text{annual}} = \hat{SR} \cdot
\frac{q}{\sqrt{q + 2\sum_{k=1}^{q-1}(q-k)\rho_k}}
$$

Positive autocorrelation makes the denominator larger, so the true annualised
Sharpe is **lower** than the naive figure. QuantProof estimates $\rho_k$ for
the first ten lags and treats the rest as zero, and runs a Ljung-Box test:
$Q = n(n+2)\sum_{k=1}^{h} \rho_k^2/(n-k)$, compared against $\chi^2_h$.

**How to read it.** Lag-1 autocorrelation above about 0.2 in a liquid daily
strategy usually means stale prices, overlapping positions, or returns that
were smoothed before they reached the file. Getmansky, Lo and Makarov (2004)
document how much this inflates reported hedge fund Sharpe ratios.
"""
    )

with st.expander(
    "Block bootstrap confidence interval", icon=":material/casino:"
):
    st.markdown(
        """
The analytic standard error above assumes a lot. The bootstrap assumes less:
resample the return series in contiguous blocks of about $n^{1/3}$ periods,
recompute the Sharpe on each resample, and read the interval off the resulting
distribution. Blocks preserve short-run dependence that an
observation-by-observation resample would destroy.

**How to read it.** The width of that distribution is the honest uncertainty
around the headline number. If a meaningful share of resamples come out
negative, the backtest has not established that the strategy makes money.
"""
    )

with st.expander(
    "Reshuffled drawdowns", icon=":material/shuffle:"
):
    st.markdown(
        """
Shuffling the returns keeps every gain and loss the strategy produced and
changes only the order they arrived in. Doing that a few thousand times gives
the distribution of maximum drawdowns consistent with those returns, and shows
where the real path sits inside it.

**How to read it.** A real path lands somewhere unremarkable in that
distribution. A path far smoother than nearly every reordering of its own
returns means the sequence is doing work the returns alone cannot explain,
which is what a look-ahead bug looks like from the outside.
"""
    )

with st.expander(
    "Fragility: concentration, costs, stability",
    icon=":material/compress:",
):
    st.markdown(
        """
- **Concentration** — the share of total log profit contributed by the best
  five periods, and the Sharpe ratio recomputed without them. A result that
  disappears when five observations are removed is a bet on rare events, and
  the sample contains almost no evidence about whether they repeat.
- **Cost sensitivity** — a flat drag in basis points applied to every period,
  plus the breakeven cost at which the average return reaches zero. This is a
  crude proxy: it charges cost per period rather than per trade. A backtest
  that reports turnover can and should do better.
- **Sub-period stability** — Sharpe ratios of the first and second halves, and
  the share of rolling one-year windows with a positive Sharpe. Splitting the
  sample is the cheapest out-of-sample test that exists.
- **Return versus drawdown** — the Calmar ratio. Long-running systematic funds
  live between 0.3 and 1.0. Values above 10 are not achievements; they are
  symptoms.
"""
    )

with st.expander("Data hygiene", icon=":material/rule:"):
    st.markdown(
        """
Before any of the statistics run, the file is checked for the failure modes
that quietly flatter results: runs of identical values and clusters of exact
zeros (prices carried forward rather than observed, which understates
volatility), duplicate timestamps, gaps far longer than the typical spacing,
and a low count of distinct values (over-rounded data). Each of these makes
the sample something other than what its date range claims.
"""
    )

st.subheader("How the score is built", icon=":material/calculate:")
st.markdown(
    "Each check returns partial credit between 0 and 1. The score is their "
    "weighted average, and then failures cap it: a weighted average alone "
    "would let ten passes drown out one fatal problem, and a backtest is only "
    "as trustworthy as its weakest link. The cap falls with both the number "
    "of failures and how badly each one failed."
)

st.warning(
    "**The weights below are a judgement call, not an estimate.** They were "
    "chosen by hand and calibrated against nothing. The checks are also not "
    "independent: the probabilistic Sharpe, minimum track record length and "
    "deflated Sharpe are largely the same evidence counted three times, which "
    "quietly gives statistical significance more say than fragility. And any "
    "single number invites people to optimise against it, which is the exact "
    "pathology this app exists to detect. Read the individual checks; treat "
    "the score as a summary for people who will not.",
    icon=":material/warning:",
)

catalog = check_catalog()
st.dataframe(
    catalog,
    hide_index=True,
    column_config={
        "Weight": st.column_config.NumberColumn(format="%.2f"),
        "Share of score": st.column_config.ProgressColumn(
            format="percent", min_value=0.0, max_value=float(
                catalog["Share of score"].max()
            )
        ),
    },
)

st.markdown(
    """
| Score | Verdict |
| --- | --- |
| 80–100 | Credible |
| 60–79 | Plausible, with caveats |
| 40–59 | Weak evidence |
| 0–39 | Do not trust |
"""
)

st.subheader("What this cannot see", icon=":material/visibility_off:")
st.markdown(
    """
A return series is the output of a research process, and most of the ways a
backtest goes wrong happen upstream of it. No test on this page can detect:

- **Survivorship bias.** A universe assembled from today's index members has
  the failures already removed. The returns look clean because the losers were
  never in the file.
- **Look-ahead bias in the data itself.** Restated fundamentals, index
  additions known before they were announced, and prices adjusted with
  information from after the trade all leave a return series that looks
  ordinary. The plausibility checks catch only the crude cases.
- **Capacity and market impact.** Nothing in a return series says how much
  capital the strategy could carry before its own trading moved the price
  against it.
- **Trial counts you do not report.** The deflated Sharpe ratio is only as
  honest as the number you enter.
- **Regime dependence.** A strategy that worked in one interest-rate
  environment can pass every test here and still fail on the day that
  environment ends.

A high score means the record is internally consistent and statistically
supported. It is not a recommendation, and it is not investment advice.
"""
)

st.subheader("Sources", icon=":material/menu_book:")
st.markdown(
    """
- Lo, A. (2002). *The Statistics of Sharpe Ratios.* Financial Analysts
  Journal 58(4).
- Bailey, D. and Lopez de Prado, M. (2012). *The Sharpe Ratio Efficient
  Frontier.* Journal of Risk 15(2).
- Bailey, D. and Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality.*
  Journal of Portfolio Management 40(5).
- Getmansky, M., Lo, A. and Makarov, I. (2004). *An Econometric Model of
  Serial Correlation and Illiquidity in Hedge Fund Returns.* Journal of
  Financial Economics 74(3).
- Harvey, C. and Liu, Y. (2015). *Backtesting.* Journal of Portfolio
  Management 42(1).
- Newey, W. and West, K. (1994). *Automatic Lag Selection in Covariance
  Matrix Estimation.* Review of Economic Studies 61(4).
- Politis, D. and Romano, J. (1994). *The Stationary Bootstrap.* Journal of
  the American Statistical Association 89(428).
"""
)
