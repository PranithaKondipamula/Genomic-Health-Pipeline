"""
Monte Carlo Calibration Test for Population Normalization
==========================================================
The critical invariant the review called out: a person whose genotype is
drawn at random from Hardy-Weinberg equilibrium at every locus should
have E[log(normalized_OR)] ≈ 0, i.e. E[normalized_OR] ≈ 1.

This test samples many simulated people under HWE, runs their
per-variant ORs through the normalization, and checks that the
geometric mean of normalized combined ORs converges to 1.

If the HWE formula is implemented correctly (H2O fix applied),
this passes. If the old 2·MAF·log(OR_het) formula is in use, the
geometric mean sits noticeably above or below 1 depending on
whether OR_hom deviates from OR_het².

Author: Pranitha Kondipamula
"""

from __future__ import annotations

import math
from statistics import mean

import numpy as np
import pytest

from src.engines.disease_risk import DiseaseEntry, DiseaseRiskEngine


# A mixed panel: common/uncommon variants, multiplicative and non-multiplicative
# homozygote effects, and one protective variant.
PANEL: list[dict] = [
    # rsid, disease, maf, or_het, or_hom
    {"rsid": "rs1", "maf": 0.29, "or_het": 1.25, "or_hom": 1.51},  # TCF7L2-like
    {"rsid": "rs2", "maf": 0.42, "or_het": 1.20, "or_hom": 1.44},  # FTO-like (OR_hom ≈ OR_het²)
    {"rsid": "rs3", "maf": 0.08, "or_het": 1.40, "or_hom": 2.30},  # uncommon, non-multiplicative
    {"rsid": "rs4", "maf": 0.50, "or_het": 0.85, "or_hom": 0.72},  # protective
    {"rsid": "rs5", "maf": 0.15, "or_het": 1.60, "or_hom": 2.00},  # dominant-ish
]


def _build_entries() -> list[DiseaseEntry]:
    """Build a disease database where each variant is the only contributor."""
    entries = []
    for v in PANEL:
        entries.append(DiseaseEntry(
            rsid=v["rsid"],
            disease="Calibration Disease",
            risk_genotypes=[("A", "T"), ("T", "T")],  # het + hom-risk
            or_het=v["or_het"], or_hom=v["or_hom"],
            ci_95=(v["or_het"] * 0.85, v["or_het"] * 1.15),
            base_rate={"m": 0.10, "f": 0.10},
            age_rates={}, bmi_modifier={},
            maf=v["maf"],
            confidence="high",
        ))
    return entries


def _sample_user_or(rng: np.random.Generator, v: dict) -> float:
    """
    Sample a single person's OR at this variant under HWE.

    Genotypes:
        hom-nonrisk (A/A): freq (1-MAF)²,  OR = 1.0
        het (A/T):         freq 2·MAF(1-MAF), OR = or_het
        hom-risk (T/T):    freq MAF²,      OR = or_hom
    """
    maf = v["maf"]
    u = rng.random()
    p_nonrisk = (1 - maf) ** 2
    p_het = 2 * maf * (1 - maf)
    if u < p_nonrisk:
        return 1.0
    if u < p_nonrisk + p_het:
        return v["or_het"]
    return v["or_hom"]


class TestHWECalibration:
    """
    Under correct HWE normalization, the geometric mean of normalized
    combined ORs across random HWE-sampled people should converge to 1.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_geometric_mean_approximately_one(self, seed):
        n_people = 20_000
        rng = np.random.default_rng(seed)
        entries = _build_entries()

        log_normalized_ors = []
        for _ in range(n_people):
            # Per-person: draw each variant independently under HWE.
            # Every variant the synthetic person is 'genotyped' at goes
            # into the normalization sum — including hom-nonrisk carriers
            # with OR=1.0. Only variants with OR ≠ 1 contribute to
            # combined_or (OR=1 is a no-op in multiplication), but all
            # genotyped variants must appear in the expectation sum for
            # the HWE invariant to hold.
            combined = 1.0
            for v in PANEL:
                combined *= _sample_user_or(rng, v)
            normalized = DiseaseRiskEngine._normalize_or(combined, entries)
            log_normalized_ors.append(math.log(max(normalized, 1e-12)))

        mean_log = mean(log_normalized_ors)
        geom_mean = math.exp(mean_log)
        # 20k samples, 5 variants, typical per-variant log-OR ~0.2 →
        # per-person SD on log OR ~0.3, SE on the mean ~0.002. A
        # tolerance of 0.015 is comfortably above that noise floor while
        # still catching a systematic bias at the few-percent level.
        assert abs(geom_mean - 1.0) < 0.015, (
            f"Geometric mean of normalized ORs = {geom_mean:.4f} "
            f"(log mean {mean_log:+.4f}). Under correct HWE normalization "
            f"this should be ≈1. A systematic offset indicates either the "
            f"old 2·MAF·log(OR_het) formula, a missing OR_hom term, or a "
            f"contributing-vs-genotyped bookkeeping bug."
        )

    def test_catches_broken_formula(self):
        """
        Sanity check: replace the correct HWE normalization with the
        *buggy* formula (2·MAF·log(OR_het), no OR_hom term) and verify the
        geometric mean shifts noticeably away from 1.

        This proves the invariant test above has teeth. We deliberately
        use a *dominant-effect* panel (OR_hom ≈ OR_het, not OR_het²) so
        the bug is diagnostic: when OR_hom ≈ OR_het², the dropped hom
        term and the missing (1-MAF) factor cancel to first order, and
        the bug is undetectable on that specific panel (as the review
        itself called out). On a dominant-effect panel there is no
        cancellation and the buggy formula misses by several percent.
        """
        # Dominant-effect variants: OR_hom ≈ OR_het, *not* OR_het².
        dominant_panel = [
            {"rsid": "d1", "maf": 0.40, "or_het": 2.00, "or_hom": 2.20},
            {"rsid": "d2", "maf": 0.30, "or_het": 1.80, "or_hom": 2.00},
            {"rsid": "d3", "maf": 0.50, "or_het": 1.50, "or_hom": 1.70},
        ]
        dominant_entries = [
            DiseaseEntry(
                rsid=v["rsid"], disease="X",
                risk_genotypes=[("A", "T"), ("T", "T")],
                or_het=v["or_het"], or_hom=v["or_hom"],
                ci_95=(v["or_het"] * 0.85, v["or_het"] * 1.15),
                base_rate={"m": 0.1, "f": 0.1},
                age_rates={}, bmi_modifier={},
                maf=v["maf"], confidence="high",
            )
            for v in dominant_panel
        ]

        rng = np.random.default_rng(0)
        n_people = 20_000

        def buggy_normalize(combined_or, genotyped):
            # Buggy: 2·MAF·log(OR_het) with no (1-MAF), no OR_hom term,
            # summed over the full genotyped list.
            mean_log = sum(
                2 * e.maf * math.log(e.or_het)
                for e in genotyped if e.maf and e.or_het
            )
            return combined_or / math.exp(mean_log) if mean_log > 0 else combined_or

        logs = []
        for _ in range(n_people):
            combined = 1.0
            for v in dominant_panel:
                combined *= _sample_user_or(rng, v)
            normalized = buggy_normalize(combined, dominant_entries)
            logs.append(math.log(max(normalized, 1e-12)))

        geom_mean_buggy = math.exp(mean(logs))
        assert abs(geom_mean_buggy - 1.0) > 0.05, (
            f"Buggy formula geom mean = {geom_mean_buggy:.4f}. On a "
            f"dominant-effect panel the buggy formula should miss 1 by "
            f">5%; the correct formula's test (tolerance 0.015) would "
            f"catch this."
        )

        # And as a positive control: the correct formula still passes on
        # this harder panel.
        rng2 = np.random.default_rng(0)
        logs_correct = []
        for _ in range(n_people):
            combined = 1.0
            for v in dominant_panel:
                combined *= _sample_user_or(rng2, v)
            normalized = DiseaseRiskEngine._normalize_or(combined, dominant_entries)
            logs_correct.append(math.log(max(normalized, 1e-12)))
        geom_mean_correct = math.exp(mean(logs_correct))
        assert abs(geom_mean_correct - 1.0) < 0.02, (
            f"Correct formula failed on dominant panel: geom mean "
            f"{geom_mean_correct:.4f}"
        )


class TestCalibrationMonotonicity:
    """Sanity properties of the normalization that should hold regardless."""

    def test_empty_genotyped_list_is_noop(self):
        """With no genotyped variants, normalization doesn't change OR."""
        assert DiseaseRiskEngine._normalize_or(1.0, []) == 1.0

    def test_hom_risk_carrier_above_one(self):
        """A hom-risk carrier at every variant should have normalized OR > 1."""
        entries = _build_entries()
        combined = math.prod(v["or_hom"] for v in PANEL)
        normalized = DiseaseRiskEngine._normalize_or(combined, entries)
        assert normalized > 1.0

    def test_hom_nonrisk_genotyped_below_one(self):
        """
        Hom-nonrisk at every variant should have normalized OR < 1:
        combined_OR=1 is divided by exp(Σ E[log OR]) > 1 for non-protective
        variants, pushing normalized OR below 1. This is the desired
        behavior — it reflects that being consistently wildtype is
        genuinely below the population average on the OR scale.
        """
        entries = _build_entries()
        # A non-protective-only subset so Σ E[log OR] is unambiguously > 0.
        nonprotective = [e for e in entries if e.or_het > 1.0]
        normalized = DiseaseRiskEngine._normalize_or(1.0, nonprotective)
        assert normalized < 1.0
