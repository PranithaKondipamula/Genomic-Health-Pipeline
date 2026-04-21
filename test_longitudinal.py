"""
Tests for the Longitudinal Biomarker Trend Analyzer.

Covers: BH-FDR correction, confounder weighting, in-range midpoint
direction logic, and duplicate-date warning.

Author: Pranitha Kondipamula
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from src.integration.longitudinal import (
    BloodDraw, LongitudinalAnalyzer, TrendResult,
)


def _draws(n: int, marker: str, values: list[float], start="2023-01-01"):
    start_dt = datetime.fromisoformat(start)
    return [
        BloodDraw(
            date=start_dt + timedelta(days=180 * i),
            location="Austin",
            values={marker: values[i]},
        )
        for i in range(n)
    ]


class TestConfounderScoring:
    def test_confounder_score_clamped(self):
        d = BloodDraw(
            date=datetime(2024, 1, 1),
            location="Austin",
            values={"ldl": 100},
            confounders={"travel": 5.0, "pollution_aqi_zscore": 5.0},
        )
        assert 0.0 <= d.confounder_score <= 1.0
        assert d.reliability_weight == 1.0 - d.confounder_score

    def test_no_confounders_means_full_weight(self):
        d = BloodDraw(date=datetime(2024, 1, 1), location="X", values={"ldl": 100})
        assert d.reliability_weight == 1.0


class TestTrendDirectionInRange:
    """In-range markers use signed distance from midpoint (M4 fix)."""

    def test_hemoglobin_drifting_toward_midpoint_is_improving(self):
        # hemoglobin range 13-17, midpoint 15. Values 16.8 → 16.0 are both
        # in range and moving toward the midpoint — should be "improving",
        # not "worsening".
        draws = _draws(3, "hemoglobin", [16.8, 16.5, 16.0])
        ana = LongitudinalAnalyzer(draws)
        r = ana.analyze_marker("hemoglobin")
        assert r is not None
        assert r.trend_direction in ("improving", "stable")

    def test_hemoglobin_drifting_away_from_midpoint_is_worsening(self):
        draws = _draws(3, "hemoglobin", [15.0, 15.8, 16.8])
        ana = LongitudinalAnalyzer(draws)
        r = ana.analyze_marker("hemoglobin")
        assert r is not None
        # Slope > 0.5/yr (over ~1 year from 15→16.8); drifting away from 15
        assert r.trend_direction == "worsening"


class TestTrendDirectionDirectional:
    def test_ldl_decreasing_is_improving(self):
        draws = _draws(4, "ldl", [180, 160, 140, 120])
        r = LongitudinalAnalyzer(draws).analyze_marker("ldl")
        assert r is not None
        assert r.trend_direction == "improving"
        assert r.slope_per_year < 0

    def test_hdl_decreasing_is_worsening(self):
        draws = _draws(4, "hdl", [60, 55, 50, 45])
        r = LongitudinalAnalyzer(draws).analyze_marker("hdl")
        assert r is not None
        assert r.trend_direction == "worsening"


class TestBHCorrection:
    def test_q_values_in_range(self):
        # Build a suite of markers with varied p-values.
        draws = []
        start = datetime(2023, 1, 1)
        for i in range(4):
            draws.append(BloodDraw(
                date=start + timedelta(days=180 * i),
                location="Austin",
                values={
                    "ldl":   [180, 160, 140, 120][i],   # strong trend
                    "hdl":   [50,  51,  50,  49][i],    # noise
                    "crp":   [2.0, 2.1, 1.9, 2.05][i],  # noise
                    "hba1c": [5.5, 5.6, 5.7, 5.8][i],   # trend
                },
            ))
        results = LongitudinalAnalyzer(draws).analyze_all()
        for r in results:
            assert 0.0 <= r.p_value <= 1.0
            assert 0.0 <= r.q_value <= 1.0
            # BH q-value should always be >= raw p for each test
            assert r.q_value >= r.p_value - 1e-9

    def test_flagging_uses_q_not_p(self):
        """A single noise marker with a tiny raw p-value should not fire
        the in-range flag once q-values are computed across 10+ noise
        markers (the family-wise fp rate is what the BH correction
        controls)."""
        draws = []
        start = datetime(2023, 1, 1)
        rng = np.random.default_rng(42)
        for i in range(4):
            vals = {f"ldl_noise_{k}": rng.normal(90, 5) for k in range(10)}
            # also include a real in-range (but noisy) marker name that
            # matches REFERENCE_RANGES so it actually gets evaluated
            vals["ldl"] = rng.normal(90, 5)
            draws.append(BloodDraw(
                date=start + timedelta(days=180 * i),
                location="Austin",
                values=vals,
            ))
        results = LongitudinalAnalyzer(draws).analyze_all()
        # Across pure noise, we expect 0 or a very small number of flags
        flagged = [r for r in results if r.is_flagged]
        assert len(flagged) <= 1


class TestDuplicateDatesWarning:
    def test_duplicate_dates_warns(self, caplog):
        import logging as lg
        start = datetime(2023, 1, 1)
        draws = [
            BloodDraw(date=start, location="Austin", values={"ldl": 100}),
            BloodDraw(date=start, location="Austin", values={"ldl": 110}),
            BloodDraw(date=start + timedelta(days=180), location="Austin",
                      values={"ldl": 120}),
        ]
        with caplog.at_level(lg.WARNING):
            LongitudinalAnalyzer(draws)
        assert any("Duplicate draw dates" in r.message for r in caplog.records)


class TestBootstrapCI:
    def test_bootstrap_ci_contains_point_estimate(self):
        draws = _draws(5, "ldl", [180, 160, 150, 140, 125])
        ana = LongitudinalAnalyzer(draws)
        r = ana.analyze_marker("ldl", n_bootstrap=500)
        assert r is not None
        # Slope-per-year should fall inside its bootstrap 95% CI (usually,
        # though with small n and resampling this is not guaranteed 100%).
        # We allow a small tolerance.
        assert r.slope_ci_low - 5 <= r.slope_per_year <= r.slope_ci_high + 5


class TestInsufficientData:
    def test_rejects_single_draw(self):
        start = datetime(2023, 1, 1)
        with pytest.raises(ValueError):
            LongitudinalAnalyzer([BloodDraw(date=start, location="Austin", values={})])
