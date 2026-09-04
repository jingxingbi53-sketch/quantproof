"""Upload a return series and find out whether the backtest holds up."""

import numpy as np
import pandas as pd
import streamlit as st

import ui
from qp import checks, loading, metrics, samples

DEFAULT_COST_BPS = {
    "Daily": 1.0,
    "Weekly": 5.0,
    "Monthly": 10.0,
    "Quarterly": 20.0,
    "Annual": 30.0,
}
KIND_LABELS = {
    "auto": "Detect automatically",
    "returns": "Period returns",
    "equity": "Equity / NAV curve",
}
SCALE_LABELS = {
    "auto": "Detect automatically",
    "decimal": "Decimals (0.01 = 1%)",
    "percent": "Percent (1.0 = 1%)",
}


def load_source(frame, label, note=None, trials=1):
    """Adopt a new table and forget how the last one was interpreted.

    The assumption defaults are set here, before the sidebar widgets are
    created, so a monthly file is costed per month from its very first render
    and a trial count carried over from an example never taints an upload.
    """
    st.session_state.frame = frame
    st.session_state.source_label = label
    st.session_state.source_note = note
    st.session_state.overrides = {}
    st.session_state.trials_default = trials
    st.session_state.cost_default = (
        DEFAULT_COST_BPS[loading.peek_frequency(frame)]
        if frame is not None
        else 1.0
    )


# ---------------------------------------------------------------------------
# Sidebar: where the data comes from and what we assume about it
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("**Your data**")
    uploaded = st.file_uploader(
        "Return file",
        type=["csv", "txt", "tsv", "xlsx", "xls"],
        help="A date column and a column of returns, or an equity curve.",
    )

    if uploaded is not None:
        upload_id = f"{uploaded.name}:{uploaded.size}"
        if upload_id != st.session_state.last_upload:
            try:
                load_source(
                    loading.read_table(uploaded.getvalue(), uploaded.name),
                    uploaded.name,
                )
                st.session_state.last_upload = upload_id
            except loading.LoadError as exc:
                st.session_state.last_upload = upload_id
                st.error(str(exc), icon=":material/error:")

    st.markdown("**Assumptions**")
    n_trials = st.number_input(
        "Strategy variants you tested",
        min_value=1,
        max_value=1_000_000,
        value=st.session_state.get("trials_default", 1),
        step=1,
        help=(
            "Every parameter set, universe and filter you tried before "
            "choosing this one. This is the input that drives the deflated "
            "Sharpe ratio, and the one people understate most."
        ),
    )
    rf_pct = st.number_input(
        "Risk-free rate (% a year)",
        min_value=0.0, max_value=25.0, value=0.0, step=0.25,
    )
    hurdle = st.number_input(
        "Sharpe ratio to beat",
        min_value=0.0, max_value=5.0, value=0.0, step=0.1,
        help="Set this above zero to test against a benchmark rather than "
             "against doing nothing.",
    )
    cost_bps = st.number_input(
        "Assumed cost per period (bps)",
        min_value=0.0, max_value=250.0,
        value=float(st.session_state.get("cost_default", 1.0)), step=0.5,
        help="Round-trip friction charged to every period: spread, slippage, "
             "commission and impact.",
    )
    with st.expander("Advanced", icon=":material/tune:"):
        confidence = st.select_slider(
            "Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
            format_func=lambda v: f"{v:.0%}",
        )
        n_boot = st.select_slider(
            "Bootstrap resamples", options=[500, 1000, 2000, 5000], value=2000,
        )

    if st.session_state.frame is not None:
        if st.button("Clear data", icon=":material/close:", width="stretch"):
            load_source(None, None)
            st.session_state.last_upload = None
            st.rerun()

    st.caption(
        "Files are analysed in memory for this session only and are never "
        "stored."
    )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------

st.title("Can this backtest be trusted?")

if st.session_state.frame is None:
    st.markdown(
        "Upload a return series and QuantProof runs the tests a risk desk "
        "would run before allocating to it: how much of the Sharpe ratio is "
        "noise, how much is survivorship of your own parameter search, and "
        "how much survives costs, serial correlation and a bad year."
    )

    st.subheader("Start with an example", icon=":material/science:")
    st.caption(
        "Four synthetic track records, each broken in a different way. Two of "
        "them look excellent until you test them."
    )

    example_columns = st.columns(2, gap="medium")
    for position, sample in enumerate(samples.all_samples()):
        with example_columns[position % 2]:
            with st.container(border=True, height="stretch"):
                st.markdown(f"**{sample.name}**")
                st.caption(sample.tagline)
                if st.button(
                    "Analyse this",
                    key=f"sample_{sample.key}",
                    icon=":material/play_arrow:",
                ):
                    load_source(
                        sample.frame,
                        sample.name,
                        sample.expectation,
                        trials=sample.suggested_trials,
                    )
                    st.rerun()

    st.subheader("Or upload your own", icon=":material/upload_file:")
    st.markdown(
        "CSV or Excel, one row per period. QuantProof works out which column "
        "holds the dates and which holds the numbers, whether they are "
        "returns or an equity curve, and whether they are decimals or "
        "percentages. You can correct any of that afterwards."
    )
    st.dataframe(samples.example_format(), hide_index=True, width="content")
    st.stop()


# ---------------------------------------------------------------------------
# Interpret the file
# ---------------------------------------------------------------------------

frame = st.session_state.frame
overrides = st.session_state.overrides

date_guess = loading.guess_date_column(frame)
value_guess = loading.guess_value_column(frame, date_guess)

date_col = overrides.get("date_col", date_guess)
value_col = overrides.get("value_col", value_guess)

if value_col is None:
    st.error(
        "No numeric column was found in that file. QuantProof needs a column "
        "of returns or portfolio values.",
        icon=":material/error:",
    )
    st.stop()

try:
    loaded = loading.build_series(
        frame,
        value_col=value_col,
        date_col=date_col,
        kind=overrides.get("kind", "auto"),
        scale=overrides.get("scale", "auto"),
        frequency=overrides.get("frequency", "auto"),
    )
except loading.LoadError as exc:
    st.error(str(exc), icon=":material/error:")
    st.stop()

if loaded.n < 20:
    st.error(
        f"Only {loaded.n} usable observations were found. At least 20 are "
        f"needed before any of these tests mean anything.",
        icon=":material/error:",
    )
    st.stop()

perf, diag, cfg, results, verdict = ui.evaluate(
    loaded,
    rf_annual=rf_pct / 100.0,
    n_trials=int(n_trials),
    benchmark_sharpe=float(hurdle),
    cost_bps=float(cost_bps),
    confidence=float(confidence),
    n_boot=int(n_boot),
)

st.caption(
    f"{st.session_state.source_label} — {loaded.n:,} "
    f"{loaded.frequency_label.lower()} observations, {perf.start:%d %b %Y} to "
    f"{perf.end:%d %b %Y}"
)
if st.session_state.source_note:
    st.caption(f":material/science: {st.session_state.source_note}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

with st.container(border=True):
    score_col, story_col = st.columns([1, 2.6], gap="large")

    with score_col:
        st.metric("Trust score", f"{verdict.score_int} / 100")
        st.progress(verdict.score / 100.0)
        st.badge(
            verdict.label,
            color=verdict.color,
            icon=":material/verified:" if verdict.n_fail == 0
            else ":material/report:",
        )

    with story_col:
        st.markdown(f"**Grade {verdict.grade}.** {verdict.summary}")
        st.markdown(
            f":green-badge[{verdict.n_pass} passed] "
            f":orange-badge[{verdict.n_warn} cautions] "
            f":red-badge[{verdict.n_fail} failed]"
        )
        if verdict.headline_issues:
            st.markdown("**What to look at first**")
            for issue in verdict.headline_issues[:3]:
                _, color, icon = ui.STATUS_META[issue.status]
                st.markdown(f"{icon} **{issue.name}** — {issue.headline}")
        else:
            st.markdown(
                "Every check passed. The remaining risk is in what a return "
                "series cannot show: how the universe was chosen, whether the "
                "prices were tradeable, and what you tried first."
            )


# ---------------------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------------------

roll = diag["stability"]["rolling_sharpe"]
equity = metrics.equity_curve(loaded.returns)
spark = equity.iloc[:: max(1, equity.size // 40)].tolist()

with st.container(horizontal=True):
    st.metric(
        "Annualised Sharpe",
        ui.num(perf.sharpe),
        border=True,
        help="Excess return per unit of volatility, scaled to one year.",
    )
    st.metric(
        "Probabilistic Sharpe",
        ui.num(diag["psr"]),
        border=True,
        help="Probability the true Sharpe beats the hurdle, given skew, fat "
             "tails and sample size.",
    )
    st.metric(
        "Deflated Sharpe",
        ui.num(diag["dsr"]),
        border=True,
        help=f"The same probability after correcting for {int(n_trials):,} "
             f"trial(s) of search.",
    )
    st.metric(
        "CAGR",
        ui.pct(perf.cagr),
        border=True,
        chart_data=spark,
        chart_type="line",
    )
    st.metric("Max drawdown", ui.pct(perf.max_drawdown), border=True)
    st.metric(
        "Annualised volatility", ui.pct(perf.ann_vol), border=True
    )


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

st.subheader("Findings", icon=":material/fact_check:")

attention = [c for c in results if c.status in (checks.WARN, checks.FAIL)]
view = st.segmented_control(
    "Which findings to show",
    options=["Needs attention", "All checks", "Passed"],
    default="Needs attention" if attention else "All checks",
    label_visibility="collapsed",
)

if view == "Needs attention":
    shown = attention
elif view == "Passed":
    shown = [c for c in results if c.status == checks.PASS]
else:
    shown = list(results)

if not shown:
    st.success(
        "Nothing in this category.", icon=":material/check_circle:"
    )

for category in checks.CATEGORIES:
    members = [c for c in shown if c.category == category]
    if not members:
        continue

    score = verdict.category_scores.get(category)
    heading = category if score is None else f"{category} · {score:.0f}/100"
    st.markdown(f"{checks.CATEGORY_ICONS[category]} **{heading}**")

    for check in members:
        label, color, icon = ui.STATUS_META[check.status]
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(label, color=color, icon=icon)
                st.markdown(f"**{check.name}**")
            st.markdown(check.headline)
            st.caption(check.detail)
            st.caption(f":material/arrow_forward: {check.advice}")


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

st.subheader("Evidence", icon=":material/insights:")

track, significance, path, costs, data = st.tabs(
    ["Track record", "Is it real?", "Path and smoothing", "Costs", "Data"]
)

with track:
    log_scale = st.toggle(
        "Log scale", value=perf.total_return > 3.0, key="log_scale"
    )
    st.altair_chart(ui.equity_chart(loaded.returns, log_scale))
    st.altair_chart(ui.drawdown_chart(loaded.returns))

    yearly = diag["stability"]["yearly_returns"]
    if yearly.size >= 2:
        st.markdown("**Calendar-year returns**")
        st.altair_chart(ui.yearly_chart(yearly))

    with st.container(horizontal=True):
        st.metric("Sortino", ui.num(perf.sortino), border=True)
        st.metric("Calmar", ui.num(perf.calmar), border=True)
        st.metric("Hit rate", ui.pct(perf.hit_rate, 0), border=True)
        st.metric(
            "Longest drawdown",
            f"{perf.longest_drawdown_days:,} days",
            border=True,
        )
        st.metric(
            "Time underwater", ui.pct(perf.time_underwater, 0), border=True
        )

with significance:
    boot = diag["bootstrap"]
    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.markdown("**Where the Sharpe ratio could plausibly be**")
        if boot.get("n_boot"):
            st.altair_chart(
                ui.distribution_chart(
                    boot["samples"],
                    perf.sharpe,
                    "Annualised Sharpe",
                    "Observed",
                    reference=0.0,
                    reference_label="No edge",
                )
            )
            st.caption(
                f"{boot['n_boot']:,} circular block bootstrap resamples with "
                f"blocks of {boot['block']} periods, which keeps short-run "
                f"dependence intact. The orange line is the number the "
                f"backtest reported; {boot['prob_positive']:.0%} of resamples "
                f"came out positive."
            )
        else:
            st.caption("Too few observations to resample.")

    with right:
        st.markdown("**Inference**")
        lo, hi = diag["sharpe_ci95"]
        st.dataframe(
            pd.DataFrame(
                {
                    "Quantity": [
                        "Sharpe (annualised)",
                        "Standard error",
                        "95% interval",
                        "Probabilistic Sharpe",
                        "Deflated Sharpe",
                        "Sharpe that search alone gives",
                        "Track record needed",
                        "Skew",
                        "Excess kurtosis",
                    ],
                    "Value": [
                        ui.num(perf.sharpe),
                        ui.num(diag["sharpe_se_annual"]),
                        f"{ui.num(lo)} to {ui.num(hi)}",
                        ui.num(diag["psr"]),
                        ui.num(diag["dsr"]),
                        ui.num(diag["dsr_benchmark_annual"]),
                        f"{diag['min_track_record_years']:.1f} years"
                        if np.isfinite(diag["min_track_record_years"])
                        else "unreachable",
                        ui.num(diag["skew"]),
                        ui.num(diag["excess_kurtosis"]),
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Trial-to-trial Sharpe dispersion was taken as "
            f"{ui.num(diag['trial_dispersion_annual'])}, estimated from the "
            f"{diag['trial_dispersion_source']}."
        )

with path:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("**Trailing one-year Sharpe**")
        if roll.dropna().size > 2:
            st.altair_chart(ui.rolling_sharpe_chart(roll, "1-year"))
        else:
            st.caption("Not enough history for a rolling window.")

        st.markdown("**Autocorrelation by lag**")
        st.altair_chart(ui.autocorrelation_chart(diag["smoothing"]["rho"]))
        st.caption(
            f"Bars far from zero mean this period's return predicts the next "
            f"one. That breaks the independence assumption behind the "
            f"annualised Sharpe: the naive "
            f"{ui.num(diag['smoothing']['naive_sharpe'])} becomes "
            f"{ui.num(diag['smoothing']['adjusted_sharpe'])} once corrected."
        )

    with right:
        st.markdown("**Drawdown under reshuffled returns**")
        perm = diag["permutation_dd"]
        if perm["samples"].size:
            st.altair_chart(
                ui.distribution_chart(
                    perm["samples"],
                    perm["observed"],
                    "Maximum drawdown",
                    "Observed",
                    value_format=".0%",
                )
            )
            st.caption(
                f"Each resample keeps every return this strategy produced and "
                f"only changes their order. The real path was smoother than "
                f"{1 - perm['percentile']:.0%} of those orderings."
            )
        else:
            st.caption("Too few observations to reshuffle.")

        conc = diag["concentration"]
        st.markdown("**Concentration**")
        st.metric(
            f"Share of profit from the best {conc['top_k']} periods",
            ui.pct(conc["top_k_share"]),
            border=True,
        )
        st.metric(
            f"Sharpe without those {conc['top_k']} periods",
            ui.num(diag["sharpe_without_best5"]),
            border=True,
        )

with costs:
    curve = diag["costs"]["curve"]
    cost_frame = pd.DataFrame(
        {
            "Cost per period (bps)": list(curve.keys()),
            "Annualised Sharpe": list(curve.values()),
        }
    ).sort_values("Cost per period (bps)")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown("**How fast the edge dies**")
        st.line_chart(
            cost_frame,
            x="Cost per period (bps)",
            y="Annualised Sharpe",
            height=ui.CHART_HEIGHT,
        )
    with right:
        st.metric(
            "Breakeven cost",
            f"{diag['costs']['breakeven_bps']:.2f} bps",
            border=True,
            help="The per-period cost that reduces the average return to "
                 "zero.",
        )
        st.dataframe(
            cost_frame,
            hide_index=True,
            column_config={
                "Annualised Sharpe": st.column_config.NumberColumn(
                    format="%.2f"
                ),
                "Cost per period (bps)": st.column_config.NumberColumn(
                    format="%.1f"
                ),
            },
        )
    st.caption(
        "Costs are applied as a flat drag on every period, which is only a "
        "rough stand-in for real trading friction. A backtest that reports "
        "turnover can do far better than this, and should."
    )

with data:
    st.markdown("**How this file was read**")
    for note in loaded.notes:
        st.caption(f":material/check: {note}")

    with st.form("interpretation", border=True):
        st.caption("Change anything QuantProof got wrong.")
        columns = list(frame.columns)
        numeric = loading.numeric_columns(frame)

        row_one = st.columns(3)
        new_date = row_one[0].selectbox(
            "Date column",
            options=["(none)"] + columns,
            index=(columns.index(date_col) + 1) if date_col in columns else 0,
        )
        new_value = row_one[1].selectbox(
            "Value column",
            options=numeric or columns,
            index=(numeric.index(value_col) if value_col in numeric else 0),
        )
        new_freq = row_one[2].selectbox(
            "Frequency",
            options=["auto"] + list(loading.FREQUENCIES),
            index=(
                ["auto"] + list(loading.FREQUENCIES)
            ).index(overrides.get("frequency", "auto")),
            format_func=lambda v: (
                "Detect automatically" if v == "auto"
                else f"{v} ({loading.FREQUENCIES[v]} periods a year)"
            ),
        )

        row_two = st.columns(2)
        new_kind = row_two[0].selectbox(
            "The column contains",
            options=list(KIND_LABELS),
            index=list(KIND_LABELS).index(overrides.get("kind", "auto")),
            format_func=KIND_LABELS.get,
        )
        new_scale = row_two[1].selectbox(
            "Units",
            options=list(SCALE_LABELS),
            index=list(SCALE_LABELS).index(overrides.get("scale", "auto")),
            format_func=SCALE_LABELS.get,
        )

        if st.form_submit_button("Re-analyse", icon=":material/refresh:"):
            st.session_state.overrides = {
                "date_col": None if new_date == "(none)" else new_date,
                "value_col": new_value,
                "kind": new_kind,
                "scale": new_scale,
                "frequency": new_freq,
            }
            st.rerun()

    st.markdown("**The returns being tested**")
    display = pd.DataFrame(
        {"date": loaded.returns.index, "return": loaded.returns.to_numpy()}
    )
    st.dataframe(
        display,
        hide_index=True,
        height=260,
        column_config={
            "date": st.column_config.DatetimeColumn(
                "Date", format="YYYY-MM-DD"
            ),
            "return": st.column_config.NumberColumn(
                "Return", format="percent"
            ),
        },
    )

    hy = loaded.hygiene
    with st.container(horizontal=True):
        st.metric("Rows read", f"{hy.rows_read:,}", border=True)
        st.metric("Exact zeros", ui.pct(hy.zero_fraction, 0), border=True)
        st.metric(
            "Longest repeat run", f"{hy.longest_repeat_run}", border=True
        )
        st.metric("Duplicate dates", f"{hy.duplicate_dates}", border=True)
        st.metric("Calendar gaps", f"{hy.calendar_gaps}", border=True)


# ---------------------------------------------------------------------------
# Take it away
# ---------------------------------------------------------------------------


def build_report() -> str:
    """A plain-text summary that survives being pasted into an email."""
    lines = [
        "# QuantProof report",
        "",
        f"Source: {st.session_state.source_label}",
        f"Period: {perf.start:%Y-%m-%d} to {perf.end:%Y-%m-%d} ({loaded.n:,} "
        f"{loaded.frequency_label.lower()} observations)",
        "",
        f"## Verdict: {verdict.score_int}/100 — {verdict.label} (grade "
        f"{verdict.grade})",
        "",
        verdict.summary,
        "",
        "## Headline numbers",
        "",
        f"- Annualised Sharpe: {ui.num(perf.sharpe)} (95% interval "
        f"{ui.num(diag['sharpe_ci95'][0])} to "
        f"{ui.num(diag['sharpe_ci95'][1])})",
        f"- Probabilistic Sharpe ratio: {ui.num(diag['psr'])}",
        f"- Deflated Sharpe ratio: {ui.num(diag['dsr'])} across "
        f"{int(n_trials):,} trial(s)",
        f"- CAGR: {ui.pct(perf.cagr)}",
        f"- Annualised volatility: {ui.pct(perf.ann_vol)}",
        f"- Maximum drawdown: {ui.pct(perf.max_drawdown)}",
        f"- Breakeven cost: {diag['costs']['breakeven_bps']:.2f} bps per "
        f"period",
        "",
        "## Checks",
        "",
    ]
    for check in results:
        label = ui.STATUS_META[check.status][0].upper()
        lines += [
            f"### [{label}] {check.name} ({check.category})",
            "",
            check.headline,
            "",
            check.detail,
            "",
            f"What to do: {check.advice}",
            "",
        ]
    lines += [
        "---",
        "",
        "Generated by QuantProof. A passing score is not investment advice, "
        "and no test of a return series can detect a universe chosen with "
        "hindsight.",
    ]
    return "\n".join(lines)


with st.container(horizontal=True, horizontal_alignment="right"):
    st.download_button(
        "Download report",
        data=build_report(),
        file_name="quantproof-report.md",
        mime="text/markdown",
        icon=":material/download:",
    )
    st.download_button(
        "Download clean returns",
        data=pd.DataFrame(
            {
                "date": loaded.returns.index.strftime("%Y-%m-%d"),
                "return": loaded.returns.to_numpy(),
            }
        ).to_csv(index=False),
        file_name="quantproof-returns.csv",
        mime="text/csv",
        icon=":material/table:",
    )
