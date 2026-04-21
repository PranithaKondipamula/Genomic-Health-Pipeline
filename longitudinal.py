"""
Longitudinal Biomarker Trend Analysis
======================================
Confounder-weighted regression on serial blood panels with Benjamini–Hochberg
FDR correction and bootstrap CIs, designed for the low-n case (few draws
per biomarker).

Model
-----
For a single subject with n draws of a biomarker, this module fits a
weighted least-squares line to (day, value) pairs where weights are the
reliability (1 - confounder_score) of each draw. With n=3-5 draws the
resulting t-based p-values are anticonservative (too small) — they treat
the per-draw confounder weight as a known constant and assume independent
residuals that are in practice correlated. Two mitigations are implemented:

  * BH-FDR correction across all biomarkers analyzed in a single batch.
  * Optional non-parametric bootstrap CIs on the slope (``bootstrap_slope``).

For multi-subject extension (n>1 participants), switch to
``statsmodels.MixedLM`` with subject as a grouping variable; the WLS
formulation here is an n=1 simplification and should not be used beyond.

Author: Pranitha Kondipamula
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class BloodDraw:
    """A single blood draw with metadata for confounder assessment."""
    date: datetime
    location: str
    values: dict[str, float]  # marker_name → value
    confounders: dict[str, float] = field(default_factory=dict)

    @property
    def confounder_score(self) -> float:
        """
        Composite confounder score (0-1).

        Higher scores indicate less reliable measurements. The weighting
        is a structural choice, not an empirical estimate, and is flagged
        as such in the paper methods; population-level expansion should
        re-estimate these weights from data.
        """
        weights = {
            "travel": 0.4,
            "pollution_aqi_zscore": 0.3,
            "sleep_deficit_zscore": 0.2,
            "stress_score": 0.1,
        }
        score = sum(self.confounders.get(k, 0) * w for k, w in weights.items())
        return min(max(score, 0.0), 1.0)

    @property
    def reliability_weight(self) -> float:
        """Inverse of confounder score — higher = more trustworthy."""
        return 1.0 - self.confounder_score


@dataclass
class TrendResult:
    """Result of longitudinal trend analysis for a single biomarker."""
    marker: str
    slope_per_year: float
    slope_ci_low: float
    slope_ci_high: float
    p_value: float
    q_value: float = 1.0  # BH-adjusted p
    trend_direction: str = "stable"
    current_value: float = 0.0
    reference_range: tuple[float, float] = (0.0, 0.0)
    is_flagged: bool = False
    flag_reason: str = ""
    values_over_time: list[tuple[datetime, float]] = field(default_factory=list)
    weighted_mean: float = 0.0
    n_draws: int = 0


class LongitudinalAnalyzer:
    """
    Analyze serial blood panel data for clinically significant trends.
    """

    # (low, high, preference, midpoint-optional). For in-range markers,
    # the midpoint anchors "worsening" (moving away from midpoint) vs
    # "improving" (moving toward it) so that a hemoglobin of 16.5 → 16.0
    # is not spuriously flagged as worsening while staying comfortably
    # in range (13-17).
    REFERENCE_RANGES: dict[str, tuple[float, float, str]] = {
        "hdl": (40, 60, "higher_better"),
        "ldl": (0, 100, "lower_better"),
        "triglycerides": (0, 150, "lower_better"),
        "total_cholesterol": (0, 200, "lower_better"),
        "fasting_glucose": (70, 100, "lower_better"),
        "hba1c": (0, 5.7, "lower_better"),
        "alt": (10, 50, "lower_better"),
        "ast": (10, 50, "lower_better"),
        "crp": (0, 3.0, "lower_better"),
        "vitamin_d": (30, 100, "higher_better"),
        "vitamin_b12": (300, 900, "higher_better"),
        "rdw": (11.6, 14.0, "lower_better"),
        "hemoglobin": (13, 17, "in_range"),
        "testosterone": (300, 1000, "higher_better"),
    }

    # Flag thresholds on the BH-adjusted q-value rather than raw p.
    # These are still lenient for n=4-5 draws and should be tightened
    # for higher-n studies.
    FLAG_Q_OUT_OF_RANGE = 0.10    # flagged + out-of-range → q < 0.10
    FLAG_Q_IN_RANGE_NEW = 0.05    # flagged + in-range novel trend → q < 0.05

    def __init__(self, draws: list[BloodDraw]):
        self.draws = sorted(draws, key=lambda d: d.date)
        if len(self.draws) < 2:
            raise ValueError("At least 2 blood draws required for trend analysis")
        # Sanity-check draw dates: same-day draws (or earlier-dated follow-ups)
        # create degenerate designs. We keep them but log a warning.
        dates = [d.date for d in self.draws]
        if len(set(dates)) < len(dates):
            logger.warning(
                "Duplicate draw dates detected. Consider combining same-day "
                "draws (e.g., via weighted mean) before trend analysis; "
                "degenerate x-values inflate apparent precision."
            )

    def analyze_all(self) -> list[TrendResult]:
        """Analyze trends for all biomarkers, with BH-FDR across the batch."""
        all_markers = sorted(
            {m for d in self.draws for m in d.values.keys()}
        )

        results: list[TrendResult] = []
        for marker in all_markers:
            r = self.analyze_marker(marker)
            if r is not None:
                results.append(r)

        # Benjamini–Hochberg FDR across all biomarkers tested.
        self._apply_bh_correction(results)

        # Now that q-values are known, re-run the flagging logic on them.
        for r in results:
            r.is_flagged, r.flag_reason = self._final_flag(r)

        results.sort(key=lambda r: (not r.is_flagged, r.q_value))
        return results

    def analyze_marker(
        self, marker: str, n_bootstrap: int = 0,
    ) -> Optional[TrendResult]:
        """
        Analyze longitudinal trend for a single biomarker.

        Uses weighted least squares where weights are the reliability
        (1 - confounder_score) of each draw. With n_bootstrap > 0, a
        non-parametric bootstrap on the slope replaces the WLS t-based CI
        (recommended for n < ~8 draws, where t-based CIs rely on a
        normal-residuals assumption we cannot verify).
        """
        points = []
        for draw in self.draws:
            if marker in draw.values:
                days = (draw.date - self.draws[0].date).days
                points.append((
                    days, draw.values[marker], draw.reliability_weight, draw.date,
                ))

        if len(points) < 2:
            return None

        days = np.array([p[0] for p in points], dtype=float)
        values = np.array([p[1] for p in points], dtype=float)
        weights = np.maximum(np.array([p[2] for p in points], dtype=float), 0.1)

        slope, intercept, std_err, p_value = self._weighted_linreg(
            days, values, weights
        )
        slope_per_year = slope * 365.25

        # CI on slope — bootstrap if requested, else t-based WLS.
        if n_bootstrap > 0 and len(points) >= 3:
            lo, hi = self._bootstrap_slope_ci(days, values, weights, n_bootstrap)
        else:
            if std_err > 0 and len(points) > 2:
                half = stats.t.ppf(0.975, df=len(points) - 2) * std_err
            else:
                half = 0.0
            lo, hi = (slope - half) * 365.25, (slope + half) * 365.25

        weighted_mean = float(np.average(values, weights=weights))
        ref = self.REFERENCE_RANGES.get(marker)
        current_value = float(values[-1])
        direction = self._direction(slope_per_year, current_value, ref)

        return TrendResult(
            marker=marker,
            slope_per_year=slope_per_year,
            slope_ci_low=lo,
            slope_ci_high=hi,
            p_value=p_value,
            trend_direction=direction,
            current_value=current_value,
            reference_range=(ref[0], ref[1]) if ref else (0.0, 0.0),
            values_over_time=[(p[3], p[1]) for p in points],
            weighted_mean=weighted_mean,
            n_draws=len(points),
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _weighted_linreg(
        x: np.ndarray, y: np.ndarray, w: np.ndarray
    ) -> tuple[float, float, float, float]:
        """WLS linear regression; returns (slope, intercept, std_err, p_value)."""
        w_sum = w.sum()
        x_mean = float(np.average(x, weights=w))
        y_mean = float(np.average(y, weights=w))

        cov_xy = float(np.sum(w * (x - x_mean) * (y - y_mean)) / w_sum)
        var_x = float(np.sum(w * (x - x_mean) ** 2) / w_sum)

        if var_x == 0:
            return 0.0, y_mean, 0.0, 1.0

        slope = cov_xy / var_x
        intercept = y_mean - slope * x_mean

        residuals = y - (slope * x + intercept)
        var_resid = float(np.sum(w * residuals ** 2) / w_sum)
        std_err = float(np.sqrt(var_resid / (var_x * w_sum))) if var_x > 0 else 0.0

        n = len(x)
        if std_err > 0 and n > 2:
            t_stat = slope / std_err
            p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2)))
        else:
            p_value = 1.0
        return slope, intercept, std_err, p_value

    @staticmethod
    def _bootstrap_slope_ci(
        x: np.ndarray, y: np.ndarray, w: np.ndarray, n_boot: int,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Percentile-bootstrap 95% CI on the per-year slope."""
        rng = np.random.default_rng(seed)
        n = len(x)
        slopes = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            xs, ys, ws = x[idx], y[idx], w[idx]
            if np.ptp(xs) == 0:  # degenerate resample (all same x)
                slopes[b] = np.nan
                continue
            s, _, _, _ = LongitudinalAnalyzer._weighted_linreg(xs, ys, ws)
            slopes[b] = s
        valid = slopes[~np.isnan(slopes)]
        if len(valid) < 10:
            return float("nan"), float("nan")
        lo, hi = np.percentile(valid, [2.5, 97.5])
        return float(lo * 365.25), float(hi * 365.25)

    @staticmethod
    def _apply_bh_correction(results: list[TrendResult]) -> None:
        """Benjamini–Hochberg FDR across the batch (mutates q_value in place)."""
        if not results:
            return
        m = len(results)
        # Rank by raw p ascending, apply BH formula, enforce monotonicity.
        order = sorted(range(m), key=lambda i: results[i].p_value)
        q_sorted = np.empty(m)
        prev = 1.0
        for rank_from_largest, idx_pos in enumerate(reversed(order)):
            k = m - rank_from_largest  # 1-indexed rank from smallest p
            p = results[idx_pos].p_value
            q = min(prev, p * m / k)
            q_sorted[idx_pos] = q
            prev = q
        for i, r in enumerate(results):
            r.q_value = float(min(1.0, max(0.0, q_sorted[i])))

    # ------------------------------------------------------------------
    # Clinical interpretation
    # ------------------------------------------------------------------
    @staticmethod
    def _direction(
        slope_per_year: float,
        current_value: float,
        ref: Optional[tuple],
    ) -> str:
        """Direction relative to the clinically-preferred direction."""
        if ref is None:
            if abs(slope_per_year) < 1:
                return "stable"
            return "increasing" if slope_per_year > 0 else "decreasing"

        low, high, preference = ref
        if preference == "higher_better":
            direction = "improving" if slope_per_year > 0 else "worsening"
        elif preference == "lower_better":
            direction = "improving" if slope_per_year < 0 else "worsening"
        else:  # in_range: use signed distance from midpoint
            mid = (low + high) / 2.0
            # If moving toward midpoint, improving; away, worsening.
            # (We approximate "moving" with sign of (current - mid) vs slope.)
            signed = (current_value - mid) * slope_per_year
            direction = "improving" if signed < 0 else "worsening"

        if abs(slope_per_year) < 0.5:
            direction = "stable"
        return direction

    def _final_flag(self, r: TrendResult) -> tuple[bool, str]:
        """Flagging decision using BH-adjusted q-values."""
        ref = self.REFERENCE_RANGES.get(r.marker)
        if ref is None:
            return False, ""
        low, high, _ = ref
        out_of_range = r.current_value < low or r.current_value > high

        if out_of_range and r.trend_direction == "worsening" \
                and r.q_value < self.FLAG_Q_OUT_OF_RANGE:
            return True, (
                f"Out of range ({r.current_value} vs {low}-{high}) and "
                f"worsening (slope={r.slope_per_year:.1f}/yr, q={r.q_value:.3f})"
            )
        if out_of_range and r.trend_direction == "stable":
            return True, (
                f"Persistently out of range ({r.current_value} vs "
                f"{low}-{high}) across {r.n_draws} draws"
            )
        if (not out_of_range) and r.trend_direction == "worsening" \
                and r.q_value < self.FLAG_Q_IN_RANGE_NEW:
            return True, (
                f"In range but significant worsening trend "
                f"(slope={r.slope_per_year:.1f}/yr, q={r.q_value:.3f})"
            )
        return False, ""
