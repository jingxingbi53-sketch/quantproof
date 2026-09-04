"""Aggregate the check battery into one number and one sentence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .checks import CATEGORIES, FAIL, PASS, SKIP, WARN, Check

# Failures cap the score, so the headline can never disagree with the red flags
# listed underneath it. A weighted average alone is too forgiving: ten passes
# would drown out one fatal problem, and a backtest is only as trustworthy as
# its weakest link. The cap is driven by how severe the failures are and how
# much the failing checks matter, not by a count of them.
CAP_SLOPE = 42.0
CAP_CURVE = 0.75
CAP_FLOOR = 8.0

BANDS = (
    (80.0, "Credible", "green",
     "The record holds up under every test applied here. Remaining risk lies "
     "in what the file cannot show: how the data was assembled and what was "
     "tried before this."),
    (60.0, "Plausible, with caveats", "orange",
     "The core result survives, but specific weaknesses below would change "
     "how much size you should put behind it."),
    (40.0, "Weak evidence", "red",
     "The performance shown is not well supported by the data behind it. "
     "Treat the headline numbers as hypotheses, not findings."),
    (0.0, "Do not trust", "red",
     "Multiple independent tests fail. On this evidence the result is more "
     "likely an artefact of the backtest than a real edge."),
)


@dataclass
class Verdict:
    score: float
    grade: str
    label: str
    color: str
    summary: str
    n_pass: int
    n_warn: int
    n_fail: int
    category_scores: dict[str, float]
    headline_issues: list[Check]

    @property
    def score_int(self) -> int:
        return int(round(self.score))


def _grade(score: float) -> str:
    bands = ((90, "A"), (80, "B"), (70, "C"), (60, "D"))
    for cutoff, letter in bands:
        if score >= cutoff:
            return letter
    return "F"


def score_checks(checks: list[Check]) -> Verdict:
    """Weighted average of check credit, capped by any failures."""
    scored = [c for c in checks if c.status != SKIP]
    total_weight = sum(c.weight for c in scored)
    earned = sum(c.score * c.weight for c in scored)
    raw = 100.0 * earned / total_weight if total_weight else 50.0

    n_fail = sum(1 for c in checks if c.status == FAIL)
    n_warn = sum(1 for c in checks if c.status == WARN)
    n_pass = sum(1 for c in checks if c.status == PASS)

    severity = sum(
        c.weight * (1.0 - c.score)
        for c in checks
        if c.status == FAIL
    )
    cap = 100.0 if severity <= 0 else max(
        CAP_FLOOR, 100.0 - CAP_SLOPE * severity ** CAP_CURVE
    )
    score = float(np.clip(min(raw, cap), 0.0, 100.0))

    category_scores: dict[str, float] = {}
    for category in CATEGORIES:
        members = [c for c in scored if c.category == category]
        weight = sum(c.weight for c in members)
        if weight:
            earned = sum(c.score * c.weight for c in members)
            category_scores[category] = 100.0 * earned / weight

    label, color, summary = "Do not trust", "red", BANDS[-1][3]
    for cutoff, band_label, band_color, band_summary in BANDS:
        if score >= cutoff:
            label, color, summary = band_label, band_color, band_summary
            break

    # Lead with the failures, then the heaviest cautions.
    issues = sorted(
        [c for c in checks if c.status in (FAIL, WARN)],
        key=lambda c: (c.status != FAIL, -c.weight, c.score),
    )

    return Verdict(
        score=score,
        grade=_grade(score),
        label=label,
        color=color,
        summary=summary,
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        category_scores=category_scores,
        headline_issues=issues,
    )
