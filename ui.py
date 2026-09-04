"""Presentation helpers: theme-aware colours, charts, and the cached pipeline.

The :mod:`qp` package knows nothing about Streamlit; everything that does lives
here or in ``app_pages/``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from qp import checks, diagnostics, metrics, scoring
from qp.loading import LoadedSeries

CHART_HEIGHT = 260

# Two hand-picked palettes rather than one flipped palette: each set is chosen
# against its own background so contrast holds in both modes.
PALETTES = {
    "light": {
        "accent": "#1859C4",
        "positive": "#067647",
        "negative": "#BC3B32",
        "caution": "#B54708",
        "muted": "#5B6B82",
        "grid": "#E1E7EF",
        "surface": "#FFFFFF",
    },
    "dark": {
        "accent": "#5AA0F2",
        "positive": "#3DD68C",
        "negative": "#F17C74",
        "caution": "#F0A63D",
        "muted": "#93A2B8",
        "grid": "#26334A",
        "surface": "#0D1421",
    },
}

STATUS_META = {
    checks.PASS: ("Pass", "green", ":material/check_circle:"),
    checks.WARN: ("Caution", "orange", ":material/warning:"),
    checks.FAIL: ("Fail", "red", ":material/cancel:"),
    checks.SKIP: ("Not run", "gray", ":material/remove:"),
}


def palette() -> dict[str, str]:
    """Colours for the theme the viewer is actually looking at."""
    mode = "dark"
    try:
        if getattr(st.context.theme, "type", None) == "light":
            mode = "light"
    except Exception:
        mode = "light"
    return PALETTES[mode]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def pct(value: float, digits: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Cached analysis
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=12)
def analyse(
    returns: pd.Series,
    periods_per_year: int,
    rf_annual: float,
    n_trials: int,
    benchmark_sharpe: float,
    cost_bps: float,
    confidence: float,
    n_boot: int,
    max_plausible_sharpe: float,
    turnover: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> tuple[metrics.Performance, dict, diagnostics.Settings]:
    """Run the whole battery once per distinct set of assumptions."""
    cfg = diagnostics.Settings(
        rf_annual=rf_annual,
        n_trials=n_trials,
        benchmark_sharpe=benchmark_sharpe,
        confidence=confidence,
        n_boot=n_boot,
        cost_bps=cost_bps,
        max_plausible_sharpe=max_plausible_sharpe,
    )
    perf = metrics.compute_performance(returns, periods_per_year, rf_annual)
    diag = diagnostics.run_all(
        returns,
        periods_per_year,
        cfg,
        turnover=turnover,
        benchmark_returns=benchmark_returns,
    )
    return perf, diag, cfg


def evaluate(
    loaded: LoadedSeries,
    benchmark_returns: pd.Series | None = None,
    **kwargs,
) -> tuple:
    """Analyse a loaded series and score it."""
    perf, diag, cfg = analyse(
        loaded.returns,
        loaded.periods_per_year,
        turnover=loaded.turnover,
        benchmark_returns=benchmark_returns,
        **kwargs,
    )
    results = checks.run_checks(perf, diag, loaded, cfg)
    return perf, diag, cfg, results, scoring.score_checks(results)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _axis(
    colors: dict[str, str],
    title: str | None = None,
    fmt: str | None = None,
):
    """A recessive axis: the data should carry the chart, not the furniture."""
    extra = {} if fmt is None else {"format": fmt}
    return alt.Axis(
        title=title,
        grid=True,
        gridColor=colors["grid"],
        gridOpacity=0.7,
        domain=False,
        tickColor=colors["grid"],
        labelColor=colors["muted"],
        titleColor=colors["muted"],
        labelFontSize=11,
        titleFontSize=11,
        titlePadding=8,
        **extra,
    )


def _thin(series: pd.Series, limit: int = 4000) -> pd.Series:
    """Keep long series plottable without changing their shape."""
    if series.size <= limit:
        return series
    step = int(np.ceil(series.size / limit))
    return series.iloc[::step]


def _hover(source: pd.DataFrame, colors: dict, tooltips: list) -> alt.Chart:
    """A crosshair rule that follows the pointer along the x axis."""
    selection = alt.selection_point(
        fields=["date"], nearest=True, on="pointerover", empty=False,
        clear="pointerout",
    )
    return (
        alt.Chart(source)
        .mark_rule(color=colors["muted"], strokeWidth=1)
        .encode(
            x=alt.X("date:T", axis=None),
            opacity=alt.condition(selection, alt.value(0.55), alt.value(0.0)),
            tooltip=tooltips,
        )
        .add_params(selection)
    )


def equity_chart(
    returns: pd.Series,
    log_scale: bool = False,
) -> alt.LayerChart:
    """Growth of one unit of capital."""
    colors = palette()
    curve = _thin(metrics.equity_curve(returns))
    source = pd.DataFrame({"date": curve.index, "equity": curve.to_numpy()})

    scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    line = (
        alt.Chart(source)
        .mark_line(color=colors["accent"], strokeWidth=2)
        .encode(
            x=alt.X("date:T", axis=_axis(colors)),
            y=alt.Y(
                "equity:Q",
                axis=_axis(colors, "Growth of 1.00"),
                scale=scale,
            ),
        )
    )
    tooltips = [
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("equity:Q", title="Value", format=",.3f"),
    ]
    return (line + _hover(source, colors, tooltips)).properties(
        height=CHART_HEIGHT
    )


def drawdown_chart(returns: pd.Series) -> alt.LayerChart:
    """Distance below the running peak -- the experience, not the summary."""
    colors = palette()
    dd = _thin(metrics.drawdown_series(returns))
    source = pd.DataFrame({"date": dd.index, "drawdown": dd.to_numpy()})

    area = (
        alt.Chart(source)
        .mark_area(
            color=colors["negative"], opacity=0.22,
            line={"color": colors["negative"], "strokeWidth": 1.5},
        )
        .encode(
            x=alt.X("date:T", axis=_axis(colors)),
            y=alt.Y("drawdown:Q", axis=_axis(colors, "Drawdown", ".0%")),
        )
    )
    tooltips = [
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("drawdown:Q", title="Drawdown", format=".2%"),
    ]
    return (area + _hover(source, colors, tooltips)).properties(height=170)


def rolling_sharpe_chart(roll: pd.Series, window_label: str) -> alt.LayerChart:
    """Trailing Sharpe, with zero marked so sign changes are obvious."""
    colors = palette()
    roll = _thin(roll.dropna())
    source = pd.DataFrame({"date": roll.index, "sharpe": roll.to_numpy()})

    line = (
        alt.Chart(source)
        .mark_line(color=colors["accent"], strokeWidth=2)
        .encode(
            x=alt.X("date:T", axis=_axis(colors)),
            y=alt.Y(
                "sharpe:Q",
                axis=_axis(colors, f"Trailing {window_label} Sharpe"),
                scale=alt.Scale(zero=False),
            ),
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color=colors["muted"], strokeDash=[4, 4], strokeWidth=1)
        .encode(y="y:Q")
    )
    tooltips = [
        alt.Tooltip("date:T", title="Window ending"),
        alt.Tooltip("sharpe:Q", title="Sharpe", format=".2f"),
    ]
    return (zero + line + _hover(source, colors, tooltips)).properties(
        height=CHART_HEIGHT
    )


def distribution_chart(
    samples: np.ndarray,
    observed: float,
    value_title: str,
    observed_label: str,
    value_format: str = ".2f",
    reference: float | None = None,
    reference_label: str = "No edge",
) -> alt.LayerChart:
    """A resampled distribution with the observed value marked on it."""
    colors = palette()
    source = pd.DataFrame({"value": np.asarray(samples, dtype=float)})

    bars = (
        alt.Chart(source)
        .mark_bar(color=colors["accent"], opacity=0.75)
        .encode(
            x=alt.X(
                "value:Q",
                bin=alt.Bin(maxbins=44),
                axis=_axis(colors, value_title, value_format),
            ),
            y=alt.Y("count():Q", axis=_axis(colors, "Resamples")),
            tooltip=[
                alt.Tooltip("count():Q", title="Resamples"),
                alt.Tooltip("value:Q", bin=alt.Bin(maxbins=44),
                            title=value_title, format=value_format),
            ],
        )
    )

    marks = [bars]
    if reference is not None and np.isfinite(reference):
        marks.append(
            alt.Chart(pd.DataFrame(
                {"x": [reference], "label": [reference_label]}
            ))
            .mark_rule(
                color=colors["muted"], strokeDash=[4, 4],
                strokeWidth=1.5,
            )
            .encode(x="x:Q", tooltip=alt.Tooltip("label:N", title=""))
        )
    if np.isfinite(observed):
        marks.append(
            alt.Chart(pd.DataFrame(
                {"x": [observed], "label": [observed_label]}
            ))
            .mark_rule(color=colors["caution"], strokeWidth=2.5)
            .encode(x="x:Q", tooltip=alt.Tooltip("label:N", title=""))
        )
    return alt.layer(*marks).properties(height=CHART_HEIGHT)


def yearly_chart(yearly: pd.Series) -> alt.Chart:
    """Calendar-year returns, coloured by sign."""
    colors = palette()
    source = pd.DataFrame(
        {
            "year": [str(y) for y in yearly.index],
            "ret": yearly.to_numpy(dtype=float),
        }
    )
    source["sign"] = np.where(source["ret"] >= 0, "Gain", "Loss")

    return (
        alt.Chart(source)
        .mark_bar(cornerRadiusEnd=4, size=26)
        .encode(
            x=alt.X("year:N", axis=_axis(colors), sort=None),
            y=alt.Y("ret:Q", axis=_axis(colors, "Return", ".0%")),
            color=alt.Color(
                "sign:N",
                scale=alt.Scale(
                    domain=["Gain", "Loss"],
                    range=[colors["positive"], colors["negative"]],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("year:N", title="Year"),
                alt.Tooltip("ret:Q", title="Return", format=".2%"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )


def dsr_curve_chart(
    curve: pd.DataFrame,
    reported_trials: int,
    breakeven: int,
    threshold: float,
) -> alt.LayerChart:
    """Deflated Sharpe against trial count, with the crossing point marked."""
    colors = palette()
    line = (
        alt.Chart(curve)
        .mark_line(color=colors["accent"], strokeWidth=2)
        .encode(
            x=alt.X(
                "trials:Q",
                scale=alt.Scale(type="log"),
                axis=_axis(colors, "Strategies tried before this one"),
            ),
            y=alt.Y(
                "dsr:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=_axis(colors, "Deflated Sharpe", ".1f"),
            ),
            tooltip=[
                alt.Tooltip("trials:Q", title="Trials", format=","),
                alt.Tooltip("dsr:Q", title="Deflated Sharpe", format=".3f"),
            ],
        )
    )
    bar = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color=colors["muted"], strokeDash=[4, 4], strokeWidth=1)
        .encode(y="y:Q")
    )
    marks = [bar, line]
    if breakeven > 0:
        marks.append(
            alt.Chart(pd.DataFrame({"x": [breakeven]}))
            .mark_rule(color=colors["positive"], strokeWidth=2)
            .encode(
                x="x:Q",
                tooltip=alt.Tooltip("x:Q", title="Survives to", format=","),
            )
        )
    if reported_trials > 1:
        marks.append(
            alt.Chart(pd.DataFrame({"x": [reported_trials]}))
            .mark_rule(color=colors["caution"], strokeWidth=2.5)
            .encode(
                x="x:Q",
                tooltip=alt.Tooltip("x:Q", title="You reported", format=","),
            )
        )
    return alt.layer(*marks).properties(height=CHART_HEIGHT)


def window_heatmap(windows: pd.DataFrame) -> alt.Chart:
    """Sharpe for every start and end date, as a diverging heatmap.

    Polarity is the point -- whether a window makes or loses money -- so the
    scale diverges from a neutral midpoint at zero rather than running through
    a single hue.
    """
    colors = palette()
    limit = float(
        np.nanmax(np.abs(windows["sharpe"].to_numpy(dtype=float)))
    ) if not windows.empty else 1.0

    return (
        alt.Chart(windows)
        .mark_rect()
        .encode(
            x=alt.X("start:T", axis=_axis(colors, "Sample starts")),
            y=alt.Y(
                "end:T",
                axis=_axis(colors, "Sample ends"),
                sort="descending",
            ),
            color=alt.Color(
                "sharpe:Q",
                scale=alt.Scale(
                    scheme="redyellowblue",
                    domain=[-limit, limit],
                    reverse=False,
                ),
                legend=alt.Legend(
                    title="Sharpe",
                    labelColor=colors["muted"],
                    titleColor=colors["muted"],
                ),
            ),
            tooltip=[
                alt.Tooltip("start:T", title="From"),
                alt.Tooltip("end:T", title="To"),
                alt.Tooltip("sharpe:Q", title="Sharpe", format=".2f"),
                alt.Tooltip("n:Q", title="Observations", format=","),
            ],
        )
        .properties(height=320)
    )


def monthly_heatmap(grid: pd.DataFrame) -> alt.Chart:
    """The classic year-by-month returns grid."""
    colors = palette()
    limit = float(
        np.nanmax(np.abs(grid["return"].to_numpy(dtype=float)))
    ) if not grid.empty else 0.1

    return (
        alt.Chart(grid)
        .mark_rect(stroke=colors["surface"], strokeWidth=2)
        .encode(
            x=alt.X("month:O", axis=_axis(colors, "Month")),
            y=alt.Y("year:O", axis=_axis(colors, "Year")),
            color=alt.Color(
                "return:Q",
                scale=alt.Scale(
                    scheme="redyellowgreen", domain=[-limit, limit]
                ),
                legend=alt.Legend(
                    title="Return",
                    format=".0%",
                    labelColor=colors["muted"],
                    titleColor=colors["muted"],
                ),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("month:O", title="Month"),
                alt.Tooltip("return:Q", title="Return", format=".2%"),
            ],
        )
        .properties(height=max(160, 26 * grid["year"].nunique()))
    )


def autocorrelation_chart(rho: np.ndarray) -> alt.LayerChart:
    """Autocorrelation by lag, against the band where noise would sit."""
    colors = palette()
    source = pd.DataFrame(
        {"lag": np.arange(1, len(rho) + 1), "rho": np.asarray(rho, dtype=float)}
    )
    bars = (
        alt.Chart(source)
        .mark_bar(color=colors["accent"], cornerRadiusEnd=3, size=18)
        .encode(
            x=alt.X("lag:O", axis=_axis(colors, "Lag")),
            y=alt.Y("rho:Q", axis=_axis(colors, "Autocorrelation", ".2f")),
            tooltip=[
                alt.Tooltip("lag:O", title="Lag"),
                alt.Tooltip("rho:Q", title="Correlation", format=".3f"),
            ],
        )
    )
    zero = (
        alt.Chart(pd.DataFrame({"y": [0.0]}))
        .mark_rule(color=colors["muted"], strokeWidth=1)
        .encode(y="y:Q")
    )
    return (zero + bars).properties(height=200)
