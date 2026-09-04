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
    benchmark, checks, context, diagnostics, loading, metrics, samples,
    scoring,
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
    factor, lags = diagnostics.lo_annualisation_factor(series, 252)
    assert factor < np.sqrt(252)
    assert lags >= 1


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
    settings.setdefault("max_plausible_sharpe", 2.5)
    settings.setdefault("cost_bps", 0.5)
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
    assert len(app.tabs) == 6


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


# ---------------------------------------------------------------------------
# Breakeven trials and the haircut
# ---------------------------------------------------------------------------

def test_breakeven_trials_falls_as_the_edge_weakens():
    strong = diagnostics.breakeven_trials(0.09, 2000, 0.0, 3.0, 0.01)
    weak = diagnostics.breakeven_trials(0.03, 2000, 0.0, 3.0, 0.01)
    assert strong > weak


def test_breakeven_trials_is_zero_without_significance():
    assert diagnostics.breakeven_trials(0.001, 60, 0.0, 3.0, 0.05) == 0


def test_dsr_at_breakeven_sits_on_the_threshold():
    args = (0.08, 1500, 0.0, 3.0, 0.02)
    n_star = diagnostics.breakeven_trials(*args, threshold=0.95)
    assert n_star > 0
    at, _ = diagnostics.deflated_sharpe_ratio(*args[:4], n_star, args[4])
    beyond, _ = diagnostics.deflated_sharpe_ratio(
        *args[:4], n_star + 1, args[4]
    )
    assert at >= 0.95 > beyond


def test_dsr_curve_is_monotonically_decreasing():
    curve = diagnostics.dsr_curve(0.08, 1500, 0.0, 3.0, 0.02)
    assert curve["trials"].is_monotonic_increasing
    assert np.all(np.diff(curve["dsr"].to_numpy()) <= 1e-9)


def test_haircut_shrinks_with_more_trials():
    once = diagnostics.haircut_sharpe(1.5, 1000, 252, 1)
    many = diagnostics.haircut_sharpe(1.5, 1000, 252, 500)
    assert once["haircut_sharpe"] == pytest.approx(1.5, rel=0.15)
    assert many["haircut_sharpe"] < once["haircut_sharpe"]
    assert 0.0 <= many["haircut"] <= 1.0


def test_haircut_never_exceeds_the_reported_sharpe():
    result = diagnostics.haircut_sharpe(2.0, 800, 252, 50)
    assert result["haircut_sharpe"] <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# Window sensitivity
# ---------------------------------------------------------------------------

def test_window_sensitivity_covers_many_windows():
    frame = make_frame(n=1200, seed=21)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    grid = diagnostics.window_sensitivity(loaded.returns, 252)
    assert len(grid) > 50
    assert (grid["end"] > grid["start"]).all()
    assert grid["n"].min() >= 20


def test_window_sensitivity_flags_a_regime_change():
    """A strategy that dies halfway should look worse from a later start.

    ``share_above_half`` is measured against each series' own headline Sharpe,
    so it is not comparable between two different strategies. The property
    worth asserting is directional: windows that begin after the edge has gone
    should score materially lower than windows that begin before it.
    """
    rng = np.random.default_rng(31)
    values = np.concatenate(
        [rng.normal(0.0012, 0.008, 700), rng.normal(0.0, 0.008, 700)]
    )
    series = pd.Series(
        values, index=pd.bdate_range("2015-01-01", periods=values.size)
    )

    grid = diagnostics.window_sensitivity(series, 252)
    cutoff = grid["start_i"].median()
    early = grid[grid["start_i"] <= cutoff]["sharpe"].median()
    late = grid[grid["start_i"] > cutoff]["sharpe"].median()
    assert late < early - 0.5


# ---------------------------------------------------------------------------
# Benchmark attribution
# ---------------------------------------------------------------------------

def make_pair(beta=0.8, alpha_daily=0.0, n=1500, seed=41):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2017-01-02", periods=n)
    market = pd.Series(rng.normal(0.0004, 0.010, n), index=index)
    noise = rng.normal(alpha_daily, 0.004, n)
    strategy = pd.Series(beta * market.to_numpy() + noise, index=index)
    return strategy, market


def test_regression_recovers_a_known_beta():
    strategy, market = make_pair(beta=0.75)
    result = benchmark.regress(strategy, market, 252)
    assert result is not None
    assert result.beta == pytest.approx(0.75, abs=0.05)
    assert result.r_squared > 0.5


def test_pure_beta_leaves_no_alpha():
    strategy, market = make_pair(beta=0.9, alpha_daily=0.0)
    result = benchmark.regress(strategy, market, 252)
    assert abs(result.alpha_t) < 2.5


def test_real_alpha_is_detected():
    strategy, market = make_pair(beta=0.3, alpha_daily=0.0005)
    result = benchmark.regress(strategy, market, 252)
    assert result.alpha_t > 2.5
    assert result.alpha_annual > 0


def test_hedging_removes_the_benchmark():
    strategy, market = make_pair(beta=0.9, alpha_daily=0.0)
    result = benchmark.regress(strategy, market, 252)
    residual = benchmark.regress(result.hedged, market, 252)
    assert abs(residual.beta) < 0.05


def test_regression_needs_overlap():
    strategy, market = make_pair(n=1500)
    assert benchmark.regress(strategy.iloc[:5], market, 252) is None


def test_newey_west_bandwidth_grows_with_sample():
    small = benchmark.newey_west_bandwidth(100)
    large = benchmark.newey_west_bandwidth(100_000)
    assert small < large


def test_alpha_check_skips_without_a_benchmark():
    results, _ = evaluate_sample(samples.trend_follower())
    alpha = next(c for c in results if c.key == "alpha")
    assert alpha.status == checks.SKIP


def test_alpha_check_fails_on_pure_beta():
    strategy, market = make_pair(beta=0.9, alpha_daily=0.0, n=2000)
    frame = pd.DataFrame(
        {
            "date": strategy.index,
            "strategy_return": strategy.to_numpy(),
            "benchmark": market.to_numpy(),
        }
    )
    loaded = loading.build_series(
        frame, value_col="strategy_return", date_col="date"
    )
    cfg = diagnostics.Settings()
    perf = metrics.compute_performance(loaded.returns, 252)
    diag = diagnostics.run_all(
        loaded.returns, 252, cfg, benchmark_returns=market
    )
    alpha = next(
        c for c in checks.run_checks(perf, diag, loaded, cfg)
        if c.key == "alpha"
    )
    assert alpha.status == checks.FAIL


# ---------------------------------------------------------------------------
# Turnover and asset-class context
# ---------------------------------------------------------------------------

def test_turnover_column_is_detected_and_used():
    frame = make_frame(n=500, seed=51)
    frame["turnover"] = 0.4
    loaded = loading.build_series(
        frame, value_col="return", date_col="date", turnover_col="turnover"
    )
    assert loaded.turnover is not None

    priced = diagnostics.cost_sensitivity(
        loaded.returns, 252, (0.0, 10.0), turnover=loaded.turnover
    )
    flat = diagnostics.cost_sensitivity(loaded.returns, 252, (0.0, 10.0))
    assert priced["model"] == "turnover"
    assert priced["mean_turnover"] == pytest.approx(0.4)
    # 10 bps against 40% turnover costs 4 bps a period, not the full 10.
    assert priced["curve"][10.0] > flat["curve"][10.0]


def test_turnover_is_guessed_from_the_column_name():
    frame = make_frame(n=200)
    frame["turnover"] = 0.25
    guessed = loading.guess_turnover_column(frame, ("date", "return"))
    assert guessed == "turnover"


def test_zero_turnover_periods_pay_nothing():
    frame = make_frame(n=400, seed=52)
    frame["turnover"] = 0.0
    loaded = loading.build_series(
        frame, value_col="return", date_col="date", turnover_col="turnover"
    )
    costs = diagnostics.cost_sensitivity(
        loaded.returns, 252, (0.0, 50.0), turnover=loaded.turnover
    )
    assert costs["curve"][50.0] == pytest.approx(costs["curve"][0.0])


def test_context_raises_the_ceiling_for_fast_books():
    assert (
        context.get("hft").max_plausible_sharpe
        > context.get("equity_daily").max_plausible_sharpe
    )


def test_context_cost_is_turnover_times_spread():
    equity = context.get("equity_daily")
    expected = equity.round_trip_bps * equity.assumed_turnover
    assert equity.flat_cost_bps == pytest.approx(expected)


def test_a_fast_book_is_not_failed_for_being_fast():
    """Sharpe 6 is a red flag in daily equity and ordinary in market making."""
    rng = np.random.default_rng(61)
    n = 2000
    values = rng.normal(0.0, 0.001, n)
    values = values - values.mean()
    values = values + 6.0 * values.std(ddof=1) / np.sqrt(252)
    frame = pd.DataFrame(
        {"date": pd.bdate_range("2018-01-01", periods=n), "return": values}
    )

    as_equity, _ = evaluate(frame, "return", max_plausible_sharpe=2.5)
    as_hft, _ = evaluate(frame, "return", max_plausible_sharpe=10.0)

    equity_status = next(
        c.status for c in as_equity if c.key == "sharpe_prior"
    )
    hft_status = next(c.status for c in as_hft if c.key == "sharpe_prior")
    assert equity_status == checks.FAIL
    assert hft_status == checks.PASS


# ---------------------------------------------------------------------------
# Tearsheet helpers
# ---------------------------------------------------------------------------

def test_drawdown_table_finds_the_worst_episode():
    returns = pd.Series(
        [0.1, -0.3, 0.05, 0.4, -0.1, 0.2],
        index=pd.bdate_range("2022-01-03", periods=6),
    )
    table = metrics.drawdown_table(returns, 5)
    assert not table.empty
    assert table["depth"].iloc[0] == pytest.approx(
        metrics.max_drawdown(returns)
    )
    assert (table["trough"] >= table["peak"]).all()


def test_monthly_grid_has_one_row_per_month():
    frame = make_frame(n=500, seed=71)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    grid = metrics.monthly_return_grid(loaded.returns)
    assert set(grid.columns) == {"year", "month", "return"}
    assert len(grid) == len(grid[["year", "month"]].drop_duplicates())


# ---------------------------------------------------------------------------
# Bootstrap corrections
# ---------------------------------------------------------------------------

def test_block_length_grows_with_dependence():
    rng = np.random.default_rng(81)
    independent = pd.Series(rng.normal(0, 0.01, 1200))
    raw = rng.normal(0, 0.01, 1210)
    smoothed = pd.Series(np.convolve(raw, [0.5, 0.3, 0.2], mode="valid"))
    assert (
        diagnostics.dependence_block_length(smoothed)
        > diagnostics.dependence_block_length(independent)
    )


def test_bootstrap_reports_its_bias_correction():
    frame = make_frame(n=800, seed=82)
    loaded = loading.build_series(frame, value_col="return", date_col="date")
    result = diagnostics.bootstrap_sharpe(loaded.returns, 252, n_boot=800)
    assert np.isfinite(result["bias_z0"])
    assert 0.0 < result["lo_pct"] < result["hi_pct"] < 100.0
