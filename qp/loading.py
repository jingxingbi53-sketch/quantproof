"""Parse an uploaded file into a clean periodic return series.

Users arrive with whatever their backtester produced: daily returns in a CSV, an
equity curve in dollars, percentages with a "%" sign, monthly numbers with gaps.
This module normalises all of that into one ``pd.Series`` of decimal
returns on a sorted ``DatetimeIndex``, and records every assumption it
had to make so the app can show its work.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATE_HINTS = (
    "date", "dates", "time", "timestamp", "datetime", "period",
    "month", "day", "asof", "as_of", "dt",
)
RETURN_HINTS = (
    "return", "returns", "ret", "rets", "pnl", "p&l", "daily", "strategy",
    "profit", "change", "pct",
)
EQUITY_HINTS = (
    "nav", "equity", "cum", "value", "balance", "portfolio",
    "wealth", "capital",
)

# Sampling frequency -> periods per year, used for every annualisation.
FREQUENCIES: dict[str, int] = {
    "Daily": 252,
    "Weekly": 52,
    "Monthly": 12,
    "Quarterly": 4,
    "Annual": 1,
}


@dataclass
class Hygiene:
    """Raw-file problems worth telling the user about."""

    rows_read: int = 0
    duplicate_dates: int = 0
    missing_values: int = 0
    non_finite: int = 0
    unsorted: bool = False
    calendar_gaps: int = 0
    zero_fraction: float = 0.0
    longest_repeat_run: int = 0
    unique_fraction: float = 1.0


@dataclass
class LoadedSeries:
    """A normalised return series plus the story of how it got that way."""

    returns: pd.Series
    periods_per_year: int
    frequency_label: str
    source_kind: str          # "returns" or "equity"
    scale: str                # "decimal" or "percent"
    value_column: str
    date_column: str | None
    notes: list[str] = field(default_factory=list)
    hygiene: Hygiene = field(default_factory=Hygiene)

    @property
    def n(self) -> int:
        return int(self.returns.size)


class LoadError(ValueError):
    """Raised when a file cannot be turned into a usable return series."""


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------

def read_table(data: bytes, filename: str) -> pd.DataFrame:
    """Read CSV/TSV/Excel bytes into a DataFrame, sniffing the delimiter."""
    name = (filename or "").lower()

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            return pd.read_excel(io.BytesIO(data))
        except ImportError as exc:  # openpyxl missing
            raise LoadError(
                "Reading Excel files needs the 'openpyxl' package. Export "
                "your data as CSV instead."
            ) from exc

    text = data.decode("utf-8-sig", errors="replace")
    if not text.strip():
        raise LoadError("The file is empty.")

    best: pd.DataFrame | None = None
    for sep in (None, ",", ";", "\t", "|"):
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
        except Exception:
            continue
        if len(df) == 0:
            continue
        if best is None or df.shape[1] > best.shape[1]:
            best = df

    if best is None:
        raise LoadError(
            "Could not parse the file as a table. CSV and Excel"
            " are supported."
        )
    return best


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _score_name(name: object, hints: tuple[str, ...]) -> int:
    lowered = str(name).strip().lower()
    for rank, hint in enumerate(hints):
        if lowered == hint:
            return 100 - rank
        if hint in lowered:
            return 50 - rank
    return 0


def _try_dates(series: pd.Series) -> pd.Series | None:
    """Parse a column as dates, returning None when that clearly fails."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    sample = series.dropna().head(200)
    if sample.empty:
        return None

    # Integer columns are usually row ids, not dates -- unless they look like
    # YYYYMMDD or YYYYMM stamps.
    if pd.api.types.is_numeric_dtype(series):
        as_int = pd.to_numeric(sample, errors="coerce").dropna()
        if as_int.empty:
            return None
        is_ymd = bool(as_int.between(19000101, 21001231).all())
        is_ym = bool(as_int.between(190001, 210012).all())
        if not (is_ymd or is_ym):
            return None
        fmt = "%Y%m%d" if is_ymd else "%Y%m"
        numbers = pd.to_numeric(series, errors="coerce")
        text = numbers.astype("Int64").astype("string")
        parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        return parsed if float(parsed.notna().mean()) > 0.8 else None

    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError):
            return None
    return parsed if float(parsed.notna().mean()) > 0.8 else None


def guess_date_column(df: pd.DataFrame) -> str | None:
    """Pick the column most likely to hold dates, or None if there isn't one."""
    best, best_score = None, 0
    for col in df.columns:
        score = _score_name(col, DATE_HINTS)
        parsed = _try_dates(df[col])
        if parsed is not None:
            # Parseability counts for more than the column's name.
            score += 60 + int(40 * float(parsed.notna().mean()))
        if score > best_score:
            best, best_score = col, score
    return best if best_score >= 60 else None


def coerce_numeric(series: pd.Series) -> pd.Series:
    """Turn '1.2%', '$1,234' and '(0.5)' into floats."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    text = series.astype("string").str.strip()
    text = text.str.replace(r"[,$€£\s]", "", regex=True)
    text = text.str.replace("%", "", regex=False)
    # Accounting negatives: (1.23) -> -1.23
    text = text.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(text, errors="coerce").astype(float)


def numeric_columns(df: pd.DataFrame, exclude: str | None = None) -> list[str]:
    """Columns that hold numbers, including ones stored as '1.2%' strings."""
    out: list[str] = []
    for col in df.columns:
        if exclude is not None and col == exclude:
            continue
        values = coerce_numeric(df[col])
        if int(values.notna().sum()) >= max(3, int(0.5 * len(df))):
            out.append(str(col))
    return out


def column_had_percent_sign(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    text = series.astype("string").dropna()
    if text.empty:
        return False
    return bool(float(text.str.contains("%", regex=False).mean()) > 0.5)


def guess_value_column(df: pd.DataFrame, date_col: str | None) -> str | None:
    """Pick the column most likely to hold the strategy's numbers."""
    candidates = numeric_columns(df, exclude=date_col)
    if not candidates:
        return None
    scored = [
        (
            _score_name(col, RETURN_HINTS)
            + _score_name(col, EQUITY_HINTS),
            i,
            col,
        )
        for i, col in enumerate(candidates)
    ]
    # Ties break toward the leftmost column, the usual convention.
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][2]


# ---------------------------------------------------------------------------
# Shape detection
# ---------------------------------------------------------------------------

def detect_kind(values: pd.Series) -> str:
    """Decide whether a column is period returns or a cumulative equity curve.

    Returns oscillate around zero; equity curves are strictly positive and
    dominated by their trend rather than their period-to-period noise.
    """
    clean = values.dropna()
    if clean.size < 3:
        return "returns"

    if bool((clean <= 0).any()):
        return "returns"

    median = float(clean.median())
    # A NAV series rarely sits inside the band that period returns live in.
    if median > 1.5 or float(clean.max()) > 10:
        return "equity"

    drift = abs(float(clean.iloc[-1]) - float(clean.iloc[0]))
    noise = float(clean.diff().abs().median() or 0.0)
    if median > 0.5 and noise > 0 and drift / noise > 20:
        return "equity"
    return "returns"


def detect_scale(values: pd.Series, kind: str) -> str:
    """Decide whether returns are decimals or percentage points."""
    if kind == "equity":
        return "decimal"
    clean = values.dropna().abs()
    if clean.empty:
        return "decimal"
    # A 5% period return is 0.05 in decimals. If typical magnitudes sit well
    # above that, the file is quoting percentage points.
    if float(clean.quantile(0.95)) > 1.0 or float(clean.std() or 0.0) > 1.0:
        return "percent"
    return "decimal"


def infer_frequency(index: pd.DatetimeIndex) -> tuple[str, int]:
    """Map the median spacing between observations to a sampling rate."""
    if index.size < 3:
        return "Daily", 252
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(float)
    gap = float(np.median(gaps))
    if gap <= 4:
        return "Daily", 252
    if gap <= 10:
        return "Weekly", 52
    if gap <= 45:
        return "Monthly", 12
    if gap <= 135:
        return "Quarterly", 4
    return "Annual", 1


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def peek_frequency(df: pd.DataFrame) -> str:
    """Guess the sampling frequency without doing the full conversion.

    The app needs this before it draws the sidebar, so that a monthly file
    arrives with a monthly cost assumption rather than a daily one.
    """
    date_col = guess_date_column(df)
    if date_col is None:
        return "Daily"
    dates = _try_dates(df[date_col])
    if dates is None:
        return "Daily"
    clean = pd.DatetimeIndex(dates.dropna().sort_values())
    return infer_frequency(clean)[0]


def build_series(
    df: pd.DataFrame,
    *,
    value_col: str,
    date_col: str | None = None,
    kind: str = "auto",
    scale: str = "auto",
    frequency: str = "auto",
) -> LoadedSeries:
    """Normalise one DataFrame column into decimal periodic returns."""
    if value_col not in df.columns:
        raise LoadError("Column '" + str(value_col) + "' is not in the file.")

    notes: list[str] = []
    hygiene = Hygiene(rows_read=len(df))

    raw = df[value_col]
    values = coerce_numeric(raw)
    hygiene.missing_values = int(values.isna().sum())
    numeric = values.to_numpy(dtype=float, na_value=np.nan)
    hygiene.non_finite = int(np.isinf(numeric).sum())

    # ---- index ----------------------------------------------------------
    if date_col is not None and date_col in df.columns:
        dates = _try_dates(df[date_col])
        if dates is None:
            raise LoadError(
                "Column '" + str(date_col)
                + "' could not be read as dates."
            )
        frame = pd.DataFrame(
            {"value": values.to_numpy(), "date": dates.to_numpy()}
        )
        frame = frame.dropna(subset=["date"])
        hygiene.unsorted = not bool(frame["date"].is_monotonic_increasing)
        hygiene.duplicate_dates = int(frame["date"].duplicated().sum())
        if hygiene.duplicate_dates:
            notes.append(
                "Dropped " + str(hygiene.duplicate_dates)
                + " duplicate timestamp(s), keeping the last row of each."
            )
            frame = frame.drop_duplicates(subset="date", keep="last")
        if hygiene.unsorted:
            notes.append("Rows were not in date order, so they were sorted.")
            frame = frame.sort_values("date")
        series = pd.Series(
            frame["value"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(frame["date"]),
            name=str(value_col),
        )
        used_date_col: str | None = str(date_col)
    else:
        # No usable dates: fall back to a synthetic business-day calendar so the
        # charts still have a sensible x-axis.
        clean = values.dropna()
        series = pd.Series(
            clean.to_numpy(dtype=float),
            index=pd.bdate_range("2000-01-03", periods=int(clean.size)),
            name=str(value_col),
        )
        used_date_col = None
        notes.append(
            "No date column was found, so observations were placed on a "
            "synthetic business-day calendar."
        )

    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.size < 2:
        raise LoadError(
            "Fewer than two usable observations were found in that"
            " column."
        )

    # ---- returns vs equity ----------------------------------------------
    resolved_kind = detect_kind(series) if kind == "auto" else kind
    if kind == "auto":
        notes.append(
            "Read the column as an equity/NAV curve and converted it to "
            "period returns."
            if resolved_kind == "equity"
            else "Read the column as period returns."
        )

    if resolved_kind == "equity":
        if bool((series <= 0).any()):
            raise LoadError(
                "An equity curve cannot contain zero or negative"
                " values."
            )
        returns = series.pct_change().dropna()
        resolved_scale = "decimal"
    else:
        resolved_scale = (
            detect_scale(series, resolved_kind)
            if scale == "auto"
            else scale
        )
        if scale == "auto" and column_had_percent_sign(raw):
            resolved_scale = "percent"
        returns = (
            series / 100.0 if resolved_scale == "percent"
            else series.copy()
        )
        if resolved_scale == "percent" and scale == "auto":
            notes.append(
                "Values looked like percentage points, so they"
                " were divided by 100."
            )

    returns = returns.astype(float)
    returns.name = "return"

    if bool((returns <= -1).any()):
        n_bad = int((returns <= -1).sum())
        notes.append(
            "Clipped " + str(n_bad) + " return(s) at -100%; a"
            " period cannot lose more than everything."
        )
        returns = returns.clip(lower=-0.9999)

    # ---- frequency -------------------------------------------------------
    if frequency == "auto":
        freq_label, ppy = infer_frequency(pd.DatetimeIndex(returns.index))
    else:
        freq_label, ppy = frequency, FREQUENCIES.get(frequency, 252)

    # ---- hygiene stats ---------------------------------------------------
    arr = returns.to_numpy(dtype=float)
    hygiene.zero_fraction = float(np.mean(arr == 0.0)) if arr.size else 0.0
    hygiene.longest_repeat_run = _longest_run(arr)
    hygiene.unique_fraction = (
        float(np.unique(np.round(arr, 12)).size / arr.size)
        if arr.size
        else 1.0
    )
    hygiene.calendar_gaps = _count_gaps(pd.DatetimeIndex(returns.index))

    return LoadedSeries(
        returns=returns,
        periods_per_year=int(ppy),
        frequency_label=freq_label,
        source_kind=resolved_kind,
        scale=resolved_scale,
        value_column=str(value_col),
        date_column=used_date_col,
        notes=notes,
        hygiene=hygiene,
    )


def _longest_run(arr: np.ndarray) -> int:
    """Longest streak of identical values -- a stale-price detector."""
    if arr.size == 0:
        return 0
    run, best = 1, 1
    for changed in np.diff(arr) != 0:
        run = 1 if changed else run + 1
        best = max(best, run)
    return int(best)


def _count_gaps(index: pd.DatetimeIndex) -> int:
    """Count spacings far longer than usual -- missing chunks of history."""
    if index.size < 3:
        return 0
    gaps = np.diff(index.values).astype("timedelta64[D]").astype(float)
    typical = float(np.median(gaps))
    if typical <= 0:
        return 0
    return int((gaps > 3 * typical + 3).sum())
