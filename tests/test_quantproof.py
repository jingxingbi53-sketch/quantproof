"""Tests for the loader, the statistics, the scoring, and the app itself."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qp import (  # noqa: E402
    checks, diagnostics, loading, metrics, samples, scoring,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def make_frame(n=300, seed=0, **columns):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    returns = rng.normal(0.0005, 0.01, n)
    frame = pd.DataFrame({"date": dates, "return": returns})
    for name, values in columns.items():
        frame[name] = values
    return frame


def test_detects_plain_returns():
    frame = make_frame()
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    assert loaded.source_kind == "returns"
    assert loaded.scale == "decimal"
    assert loaded.frequency_label == "Daily"
    assert loaded.periods_per_year == 252
    assert loaded.n == 300


def test_detects_percentage_points():
    frame = make_frame()
    frame["pct"] = frame["return"] * 100
    loaded = loading.build_series(frame, value_col="pct", date_col="date")
    assert loaded.scale == "percent"
    assert np.allclose(loaded.returns.to_numpy(), frame["return"].to_numpy())


def test_detects_percent_signs_in_text():
    frame = make_frame()
    frame["text"] = [f"{v * 100:.4f}%" for v in frame["return"]]
    loaded = loading.build_series(frame, value_col="text", date_col="date")
    assert loaded.scale == "percent"
    assert loaded.returns.abs().max() < 0.2


def test_converts_equity_curve_to_returns():
    frame = make_frame()
    frame["nav"] = 1_000_000 * (1 + frame["return"]).cumprod()
    loaded = loading.build_series(frame, value_col="nav", date_col="date")
    assert loaded.source_kind == "equity"
    assert loaded.n == len(frame) - 1
    assert np.allclose(
        loaded.returns.to_numpy(), frame["return"].to_numpy()[1:], atol=1e-12
    )


def test_handles_accounting_negatives_and_currency():
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=4),
            "pnl": ["$1,000.00", "(500.00)", "250", "$1,250.50"],
        }
    )
    values = loading.coerce_numeric(frame["pnl"])
    assert list(values) == [1000.0, -500.0, 250.0, 1250.5]


def test_sorts_and_deduplicates():
    frame = make_frame(n=50)
    shuffled = pd.concat([frame.iloc[::-1], frame.iloc[:3]], ignore_index=True)
    loaded = loading.build_series(shuffled, value_col="return", date_col="date")
    assert loaded.returns.index.is_monotonic_increasing
    assert loaded.n == 50
    assert loaded.hygiene.duplicate_dates == 3


def test_survives_missing_date_column():
    frame = make_frame().drop(columns=["date"])
    loaded = loading.build_series(frame, value_col="return", date_col=None)
    assert loaded.date_column is None
    assert loaded.n == 300


def test_infers_monthly_frequency():
    dates = pd.date_range("2015-01-31", periods=60, freq="ME")
    frame = pd.DataFrame({"d": dates, "r": np.zeros(60) + 0.01})
    loaded = loading.build_series(frame, value_col="r", date_col="d")
    assert loaded.frequency_label == "Monthly"
    assert loaded.periods_per_year == 12


def test_guesses_the_strategy_column_over_the_benchmark():
    frame = make_frame()
    frame = frame.rename(columns={"return": "strategy_return"})
    frame["benchmark"] = 0.001
    frame["row_id"] = range(len(frame))
    date_col = loading.guess_date_column(frame)
    value_col = loading.guess_value_column(frame, date_col)
    assert date_col == "date"
    assert value_col == "strategy_return"


def test_rejects_a_column_with_no_numbers():
    frame = pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=30)})
    assert loading.guess_value_column(frame, "date") is None


def test_flags_stale_prices():
    frame = make_frame(n=200)
    frame.loc[10:25, "return"] = 0.0
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    assert loaded.hygiene.longest_repeat_run >= 15
    assert loaded.hygiene.zero_fraction > 0.05


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_equity_and_drawdown_are_consistent():
    returns = pd.Series(
        [0.10, -0.20, 0.05], index=pd.bdate_range("2022-01-03", periods=3)
    )
    curve = metrics.equity_curve(returns)
    assert curve.iloc[-1] == pytest.approx(1.10 * 0.80 * 1.05)
    # The trough sits 20% below the 1.10 peak.
    assert metrics.max_drawdown(returns) == pytest.approx(-0.20)


def test_sharpe_matches_the_definition():
    rng = np.random.default_rng(1)
    values = rng.normal(0.001, 0.01, 1000)
    index = pd.bdate_range("2019-01-01", periods=1000)
    returns = pd.Series(values, index=index)
    expected = values.mean() / values.std(ddof=1) * np.sqrt(252)
    assert metrics.sharpe_ratio(returns, 252) == pytest.approx(expected)


def test_risk_free_rate_lowers_the_sharpe():
    frame = make_frame(n=1000, seed=5)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    gross = metrics.sharpe_ratio(loaded.returns, 252, 0.0)
    net = metrics.sharpe_ratio(loaded.returns, 252, 0.05)
    assert net < gross


def test_zero_volatility_does_not_explode():
    returns = pd.Series(
        np.zeros(50), index=pd.bdate_range("2022-01-03", periods=50)
    )
    assert metrics.sharpe_ratio(returns, 252) == 0.0
    assert metrics.max_drawdown(returns) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_psr_rises_with_sample_length():
    short = diagnostics.probabilistic_sharpe_ratio(0.05, 100, 0.0, 3.0)
    long = diagnostics.probabilistic_sharpe_ratio(0.05, 2000, 0.0, 3.0)
    assert 0.5 < short < long < 1.0


def test_psr_is_a_half_at_the_benchmark():
    psr = diagnostics.probabilistic_sharpe_ratio(0.05, 500, 0.0, 3.0, 0.05)
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_negative_skew_and_fat_tails_reduce_confidence():
    normal = diagnostics.probabilistic_sharpe_ratio(0.08, 500, 0.0, 3.0)
    ugly = diagnostics.probabilistic_sharpe_ratio(0.08, 500, -1.5, 9.0)
    assert ugly < normal


def test_expected_max_sharpe_grows_with_trials():
    one = diagnostics.expected_max_sharpe(1, 0.5)
    ten = diagnostics.expected_max_sharpe(10, 0.5)
    thousand = diagnostics.expected_max_sharpe(1000, 0.5)
    assert one == 0.0
    assert 0.0 < ten < thousand


def test_deflated_sharpe_falls_as_trials_rise():
    few, _ = diagnostics.deflated_sharpe_ratio(0.09, 500, 0.0, 3.0, 1, 0.05)
    many, hurdle = diagnostics.deflated_sharpe_ratio(
        0.09, 500, 0.0, 3.0, 5000, 0.05
    )
    assert many < few
    assert hurdle > 0


def test_min_track_record_length_is_unreachable_without_an_edge():
    assert diagnostics.min_track_record_length(0.0, 0.0, 3.0) == float("inf")
    assert diagnostics.min_track_record_length(-0.1, 0.0, 3.0) == float("inf")


def test_autocorrelation_detects_a_smoothed_series():
    rng = np.random.default_rng(3)
    raw = rng.normal(0, 0.01, 800)
    smoothed = np.convolve(raw, [0.6, 0.25, 0.15], mode="valid")
    series = pd.Series(
        smoothed, index=pd.bdate_range("2019-01-01", periods=len(smoothed))
    )
    rho = diagnostics.autocorrelations(series, 3)
    assert rho[0] > 0.25
    _, p_value, _ = diagnostics.ljung_box(series, 5)
    assert p_value < 0.01


def test_lo_factor_penalises_positive_autocorrelation():
    rng = np.random.default_rng(4)
    raw = rng.normal(0, 0.01, 800)
    smoothed = np.convolve(raw, [0.6, 0.25, 0.15], mode="valid")
    series = pd.Series(
        smoothed, index=pd.bdate_range("2019-01-01", periods=len(smoothed))
    )
    factor = diagnostics.lo_annualisation_factor(series, 252)
    assert factor < np.sqrt(252)


def test_bootstrap_brackets_the_observed_sharpe():
    frame = make_frame(n=800, seed=7)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    result = diagnostics.bootstrap_sharpe(loaded.returns, 252, n_boot=600)
    observed = metrics.sharpe_ratio(loaded.returns, 252)
    assert result["p05"] < observed < result["p95"]


def test_removing_the_best_days_lowers_the_sharpe():
    frame = make_frame(n=500, seed=9)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    full = metrics.sharpe_ratio(loaded.returns, 252)
    trimmed = diagnostics.sharpe_without_best(loaded.returns, 252, 5)
    assert trimmed < full


def test_costs_reduce_the_sharpe_monotonically():
    frame = make_frame(n=500, seed=11)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    curve = diagnostics.cost_sensitivity(loaded.returns, 252)["curve"]
    levels = sorted(curve)
    values = [curve[level] for level in levels]
    assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Checks and scoring
# ---------------------------------------------------------------------------

def evaluate(frame, value_col, date_col="date", **settings):
    loaded = loading.build_series(
        frame, value_col=value_col, date_col=date_col
    )
    cfg = diagnostics.Settings(**settings)
    perf = metrics.compute_performance(
        loaded.returns, loaded.periods_per_year, cfg.rf_annual
    )
    diag = diagnostics.run_all(loaded.returns, loaded.periods_per_year, cfg)
    results = checks.run_checks(perf, diag, loaded, cfg)
    return results, scoring.score_checks(results)


def evaluate_sample(sample, **settings):
    frame = sample.frame
    date_col = loading.guess_date_column(frame)
    value_col = loading.guess_value_column(frame, date_col)
    settings.setdefault("n_trials", sample.suggested_trials)
    return evaluate(frame, value_col, date_col, **settings)


def test_every_check_runs_and_is_well_formed():
    results, verdict = evaluate_sample(samples.trend_follower())
    assert len(results) == len(checks.CHECK_FUNCS)
    assert len({c.key for c in results}) == len(results)
    for check in results:
        assert check.status in (checks.PASS, checks.WARN, checks.FAIL,
                                checks.SKIP)
        assert 0.0 <= check.score <= 1.0
        assert check.category in checks.CATEGORIES
        assert check.headline and check.detail and check.advice
    assert 0 <= verdict.score <= 100


def test_a_credible_record_scores_well():
    _, verdict = evaluate_sample(samples.trend_follower())
    assert verdict.score >= 80
    assert verdict.label == "Credible"
    assert verdict.n_fail == 0


def test_the_overfit_sweep_is_caught_by_the_deflated_sharpe():
    results, verdict = evaluate_sample(samples.overfit_search())
    by_key = {c.key: c for c in results}
    # Classically significant...
    assert by_key["psr"].status == checks.PASS
    # ...but not once the 500 trials are priced in.
    assert by_key["dsr"].status == checks.FAIL
    assert verdict.score < 60


def test_the_trial_count_is_what_condemns_the_overfit_sweep():
    _, honest = evaluate_sample(samples.overfit_search(), n_trials=500)
    _, understated = evaluate_sample(samples.overfit_search(), n_trials=1)
    assert understated.score > honest.score + 30


def test_smoothed_returns_are_flagged():
    results, verdict = evaluate_sample(samples.smoothed_fund())
    by_key = {c.key: c for c in results}
    assert by_key["autocorrelation"].status == checks.FAIL
    assert verdict.score < 70


def test_a_look_ahead_bug_is_rejected_despite_significance():
    results, verdict = evaluate_sample(samples.lookahead_bug())
    by_key = {c.key: c for c in results}
    assert by_key["psr"].status == checks.PASS
    assert by_key["dsr"].status == checks.PASS
    assert by_key["sharpe_prior"].status == checks.FAIL
    assert by_key["calmar"].status == checks.FAIL
    assert verdict.label == "Do not trust"
    assert verdict.score < 30


def test_the_examples_are_ordered_as_intended():
    scored = {
        sample.key: evaluate_sample(sample)[1].score
        for sample in samples.all_samples()
    }
    assert scored["trend"] > scored["smoothed"] > scored["overfit"]
    assert scored["overfit"] > scored["lookahead"]


def test_failures_cap_the_score_below_the_weighted_average():
    results, verdict = evaluate_sample(samples.lookahead_bug())
    scored = [c for c in results if c.status != checks.SKIP]
    weight = sum(c.weight for c in scored)
    raw = 100.0 * sum(c.score * c.weight for c in scored) / weight
    assert verdict.score < raw


def test_categories_are_all_scored():
    _, verdict = evaluate_sample(samples.trend_follower())
    assert set(verdict.category_scores) == set(checks.CATEGORIES)


# ---------------------------------------------------------------------------
# The app itself
# ---------------------------------------------------------------------------

def test_app_starts_and_analyses_a_sample():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Can this backtest be trusted?"

    sample_buttons = [b for b in app.button if b.key == "sample_trend"]
    assert sample_buttons, "the empty state should offer the examples"
    sample_buttons[0].click().run()
    assert not app.exception

    scores = [m.value for m in app.metric if m.label == "Trust score"]
    assert scores and scores[0].endswith("/ 100")
    assert len(app.tabs) == 5


def test_app_reports_a_file_that_is_too_short():
    from streamlit.testing.v1 import AppTest

    frame = make_frame(n=8)
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    app.run()
    app.sidebar.file_uploader[0].set_value(
        ("short.csv", frame.to_csv(index=False).encode(), "text/csv")
    ).run()
    assert not app.exception
    assert any("20" in e.value for e in app.error)


def test_methodology_page_renders():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180)
    app.run()
    app.switch_page("app_pages/methodology.py")
    app.run()
    assert not app.exception
    assert app.title[0].value == "Methodology"
