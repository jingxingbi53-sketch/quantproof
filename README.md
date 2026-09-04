# QuantProof

**Can this backtest be trusted?**

Upload a return series and QuantProof runs the tests a risk desk would run
before allocating to it: how much of the Sharpe ratio is sampling noise, how
much is survivorship of your own parameter search, and how much survives
trading costs, serial correlation and a bad year. It returns a 0–100 trust
score, a verdict, and — for every check — the number, what it means, and what
to do about it.

Live app: <https://quantproof-bobby.streamlit.app>

## What it tests

| Category | Checks |
| --- | --- |
| Statistical significance | Sample size, Probabilistic Sharpe Ratio, minimum track record length |
| Overfitting risk | Deflated Sharpe Ratio, alpha vs beta, sample-window sensitivity, Sharpe plausibility, return-vs-drawdown, reshuffled drawdowns |
| Fragility | Return smoothing (autocorrelation), profit concentration, tail shape, cost sensitivity, sub-period stability, rolling consistency |
| Data quality | Stale prices, exact zeros, duplicate timestamps, calendar gaps, over-rounding |

Three of these do work nothing else here does:

- **Breakeven trials.** The deflated Sharpe ratio depends entirely on how many
  variants you tried, and asking for that number invites the flattering answer.
  QuantProof inverts it and reports the trial count at which your Sharpe stops
  being distinguishable from the best of that many random attempts. You are
  never asked to grade your own search — only to judge whether it was wider
  than the number on screen.
- **Alpha versus beta.** Give it a benchmark and it regresses the strategy on
  it with Newey-West standard errors, then re-runs the whole battery on the
  hedged residual. Without this, a 1.2 Sharpe that is 0.8 beta to the equity
  market passes every other test on the page.
- **Sample-window sensitivity.** The Sharpe is recomputed over every start and
  end date on a grid. Nothing else catches a date range chosen with hindsight;
  the reshuffle test varies the order of returns inside a fixed window, which
  is a different question.

The statistics come from the literature rather than from folklore:

- Lo (2002), *The Statistics of Sharpe Ratios* — standard error of the Sharpe
  ratio under non-normal returns, and the correct annualisation factor when
  returns are serially correlated.
- Bailey & López de Prado (2012, 2014) — the Probabilistic and Deflated Sharpe
  Ratios, and minimum track record length.
- Getmansky, Lo & Makarov (2004) — return smoothing in illiquid portfolios.
- Harvey & Liu (2015) — the multiple-testing haircut on a Sharpe ratio.
- Politis & Romano (1994) — block bootstrap resampling.
- Newey & West (1994) — automatic lag selection for the HAC estimators.

The **Methodology** page in the app states every formula, and is explicit about
what a return series cannot reveal: survivorship bias, look-ahead bias baked
into the data, capacity limits, and trial counts you choose not to report.

## Input formats

One row per period, in CSV or Excel. QuantProof detects:

- which column holds dates (including `YYYYMMDD` integer stamps) and which
  holds the numbers;
- whether the numbers are period returns or a cumulative equity/NAV curve;
- whether returns are decimals (`0.0123`) or percentage points (`1.23`, `1.23%`);
- the sampling frequency (daily, weekly, monthly, quarterly, annual);
- an optional `turnover` column, which upgrades the cost model from a flat
  per-period drag to `net = gross - turnover x cost`;
- an optional benchmark column, or a second uploaded file.

It copes with `$1,234.56`, `(0.5)` for negatives, semicolon and tab delimiters,
unsorted rows, duplicate timestamps, and files with no date column at all.
Every inference is shown on the **Data** tab and can be overridden there.

## Running it locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Four synthetic examples are built in, so the app can be judged without
uploading anything:

| Example | What it is | Expected verdict |
| --- | --- | --- |
| Diversified trend follower | Eight years, Sharpe ≈ 0.9, real drawdowns | Credible (97) |
| Best of 500 parameter sets | The winner of a sweep over pure noise | Weak evidence (41) |
| Illiquid credit fund | A real edge, reported on smoothed monthly marks | Weak evidence (50) |
| Signal with a look-ahead bug | Sharpe 10.8, 3.4% max drawdown | Do not trust (16) |

The overfit sweep is the clearest demonstration: it survives up to **3 trials**
and was picked from 500.

The last one is the point of the whole app: it is overwhelmingly significant by
every classical test and still worthless.

## Tests

```bash
pip install pytest
pytest
```

The suite covers the loader's format detection, closed-form checks on the
statistics, the calibration of the four examples, and a headless run of the app
itself through `streamlit.testing.v1.AppTest`.

## Layout

```
streamlit_app.py       Entry point and navigation
ui.py                  Charts, theme-aware colours, cached pipeline
app_pages/
  analyze.py           Upload, verdict, findings, evidence
  methodology.py       Formulas, weights, and limitations
qp/                    Analytics, with no Streamlit dependency
  loading.py           File parsing and format detection
  metrics.py           Descriptive performance statistics
  diagnostics.py       Inference and fragility tests
  benchmark.py         Newey-West regression, alpha and the hedged stream
  context.py           Asset-class priors for plausibility and cost
  checks.py            Diagnostics turned into pass / caution / fail
  scoring.py           Trust score and verdict
  samples.py           The four worked examples
tests/                 pytest suite
```

`qp/` imports no Streamlit, so the analytics can be used from a notebook:

```python
from qp import checks, diagnostics, loading, metrics, scoring

loaded = loading.build_series(df, value_col="return", date_col="date")
cfg = diagnostics.Settings(n_trials=250)
perf = metrics.compute_performance(loaded.returns, loaded.periods_per_year)
diag = diagnostics.run_all(loaded.returns, loaded.periods_per_year, cfg)
verdict = scoring.score_checks(checks.run_checks(perf, diag, loaded, cfg))
print(verdict.score, verdict.label)
```

## Deploying to Streamlit Community Cloud

1. Push the contents of this folder to a public GitHub repository, with
   `streamlit_app.py` at the repository root.
2. At <https://share.streamlit.io>, choose **New app**, select the repository
   and branch, and set the main file to `streamlit_app.py`.
3. Set the custom subdomain to `quantproof-bobby`.
4. Under **Advanced settings**, pin Python to 3.12 or 3.13.

`requirements.txt` and `.streamlit/config.toml` are picked up automatically.
Uploaded files are held in memory for the session and never written to disk.

## A caveat worth repeating

A high score means a track record is internally consistent and statistically
supported. It is not a recommendation, and it is not investment advice. The
most expensive backtest errors happen before the return series exists.

The trust score itself deserves suspicion. Its weights were chosen by hand and
calibrated against nothing, the checks are not independent, and any single
number invites people to optimise against it — which is the exact pathology
this app exists to detect. Read the individual checks. The score is a summary
for people who will not.
