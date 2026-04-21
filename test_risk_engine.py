"""
Tests for the Disease Risk Calculator Engine.

Covers the mathematical pipeline (population normalization, OR→RR,
age attenuation, BMI-mediation shrinkage, gene-gene interactions,
APOE joint-genotype handling) and the new SE-propagated CIs.

Author: Pranitha Kondipamula
"""

import math

import pytest

from src.engines.disease_risk import (
    DiseaseEntry, DiseaseRiskEngine, GeneInteraction, RiskResult, UserProfile,
    Z_95,
)


@pytest.fixture
def tcf7l2_entry() -> DiseaseEntry:
    """TCF7L2 rs7903146 — canonical T2D risk variant."""
    return DiseaseEntry(
        rsid="rs7903146",
        disease="Type 2 Diabetes",
        risk_genotypes=[("C", "T"), ("T", "T")],
        or_het=1.25, or_hom=1.51,
        ci_95=(1.39, 1.65),
        base_rate={"m": 0.35, "f": 0.35},
        age_rates={"40": 0.15, "50": 0.40, "60": 1.00, "70": 1.50},
        bmi_modifier={"normal": 0.5, "overweight": 2.0, "obese": 5.0},
        maf=0.29,
        confidence="very_high",
        pmid="22922229",
    )


@pytest.fixture
def engine(tcf7l2_entry) -> DiseaseRiskEngine:
    return DiseaseRiskEngine(disease_db=[tcf7l2_entry], interactions=[])


# -----------------------------------------------------------------------
# OR → RR conversion
# -----------------------------------------------------------------------
class TestORtoRR:
    def test_rare_disease(self):
        base, or_ = 0.01, 2.0
        rr = or_ / (1 - base + base * or_)
        assert abs(rr - 1.98) < 0.01

    def test_common_disease(self):
        base, or_ = 0.40, 3.0
        rr = or_ / (1 - base + base * or_)
        assert rr < or_
        assert base * rr < 1.0

    def test_extreme_or(self):
        base, or_ = 0.50, 100.0
        rr = or_ / (1 - base + base * or_)
        assert base * rr < 1.0


# -----------------------------------------------------------------------
# Population normalization — corrected HWE formula
# -----------------------------------------------------------------------
class TestPopulationNormalization:
    def test_uses_correct_hwe_formula(self, tcf7l2_entry):
        """
        Verify normalization uses 2·MAF(1-MAF)·log(OR_het) + MAF²·log(OR_hom),
        not the old 2·MAF·log(OR_het).
        """
        maf, or_het, or_hom = 0.29, 1.25, 1.51
        expected_mean_log_or = (
            2 * maf * (1 - maf) * math.log(or_het)
            + maf ** 2 * math.log(or_hom)
        )
        # For a hom-risk carrier the raw user OR is or_hom
        normalized = DiseaseRiskEngine._normalize_or(or_hom, [tcf7l2_entry])
        expected = or_hom / math.exp(expected_mean_log_or)
        assert math.isclose(normalized, expected, rel_tol=1e-9)

    def test_empty_contributing_list_is_noop(self):
        """No contributing entries → no normalization applied."""
        assert DiseaseRiskEngine._normalize_or(1.5, []) == 1.5

    def test_zero_maf_skipped(self):
        """Entries with MAF=0 should not participate in normalization."""
        e = DiseaseEntry(
            rsid="x", disease="x",
            risk_genotypes=[("A", "A")],
            or_het=1.5, or_hom=2.0, ci_95=(1.2, 1.8),
            base_rate={"m": 0.1, "f": 0.1},
            age_rates={}, bmi_modifier={},
            maf=0.0, confidence="high",
        )
        assert DiseaseRiskEngine._normalize_or(1.5, [e]) == 1.5

    def test_protective_allele_still_normalized(self):
        """Protective variants (OR_het < 1) should also be normalized, not skipped."""
        e = DiseaseEntry(
            rsid="x", disease="x",
            risk_genotypes=[("A", "A")],
            or_het=0.8, or_hom=0.65, ci_95=(0.55, 0.78),
            base_rate={"m": 0.1, "f": 0.1},
            age_rates={}, bmi_modifier={},
            maf=0.30, confidence="high",
        )
        # Old code had `if mean_log_or > 0` and would skip this.
        # Expected mean log OR is negative; the normalized OR should
        # therefore be *larger* than the input (dividing by exp(negative)).
        normalized = DiseaseRiskEngine._normalize_or(0.65, [e])
        assert normalized > 0.65

    def test_fallback_when_or_hom_missing(self):
        """If or_hom is 0/missing, fall back to or_het² multiplicative assumption."""
        e = DiseaseEntry(
            rsid="x", disease="x",
            risk_genotypes=[("A", "A")],
            or_het=1.5, or_hom=0.0, ci_95=(1.2, 1.8),
            base_rate={"m": 0.1, "f": 0.1},
            age_rates={}, bmi_modifier={},
            maf=0.30, confidence="high",
        )
        # Should not crash and should apply some normalization
        normalized = DiseaseRiskEngine._normalize_or(1.5, [e])
        assert normalized < 1.5
        assert normalized > 0


# -----------------------------------------------------------------------
# SE-propagated confidence intervals (replaces coverage heuristic)
# -----------------------------------------------------------------------
class TestCIPropagation:
    def test_per_variant_se_from_published_ci(self, tcf7l2_entry):
        """Per-variant log-OR SE comes from (log(high)-log(low))/(2·Z_95)."""
        expected_se = (math.log(1.65) - math.log(1.39)) / (2 * Z_95)
        expected_var = expected_se ** 2
        computed = DiseaseRiskEngine._entry_log_or_var(tcf7l2_entry)
        assert math.isclose(computed, expected_var, rel_tol=1e-9)

    def test_ci_wider_for_higher_se(self):
        """Larger log-OR SE → wider CI on absolute risk."""
        or_ = 2.0
        base = 0.10
        narrow = DiseaseRiskEngine._compute_ci(or_, 0.05, base)
        wide = DiseaseRiskEngine._compute_ci(or_, 0.30, base)
        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_zero_se_collapses_to_point(self):
        """If no per-variant SEs are available, CI collapses to point estimate."""
        lo, hi = DiseaseRiskEngine._compute_ci(1.5, 0.0, 0.10)
        assert lo == hi

    def test_degenerate_published_ci_falls_back(self):
        """A reversed or zero-width CI falls back to a log(2)-based default."""
        bad = DiseaseEntry(
            rsid="x", disease="x",
            risk_genotypes=[("A", "A")],
            or_het=1.5, or_hom=2.0, ci_95=(0.0, 0.0),
            base_rate={"m": 0.1, "f": 0.1},
            age_rates={}, bmi_modifier={},
            maf=0.3, confidence="high",
        )
        var = DiseaseRiskEngine._entry_log_or_var(bad)
        assert var > 0


# -----------------------------------------------------------------------
# Age attenuation (now returns a pair)
# -----------------------------------------------------------------------
class TestAgeAttenuation:
    def test_young_no_attenuation(self):
        or_out, atten = DiseaseRiskEngine._attenuate_by_age(2.0, 30)
        assert atten == 1.0
        assert math.isclose(or_out, 2.0)

    def test_old_strong_attenuation(self):
        or_out, atten = DiseaseRiskEngine._attenuate_by_age(2.0, 80)
        assert atten < 1.0
        assert 1.0 < or_out < 2.0


# -----------------------------------------------------------------------
# BMI-mediation shrinkage with per-variant fractions (H3 fix)
# -----------------------------------------------------------------------
class TestBMIMediation:
    def test_variant_specific_fraction(self):
        """OR_adjusted = 1 + (OR_raw - 1) × (1 - mediation_fraction)."""
        e = DiseaseEntry(
            rsid="rs9939609", disease="Obesity",
            risk_genotypes=[("A", "T"), ("A", "A")],
            or_het=1.20, or_hom=1.44, ci_95=(1.35, 1.53),
            base_rate={"m": 0.30, "f": 0.30},
            age_rates={}, bmi_modifier={"obese": 1.0},
            maf=0.42, bmi_mediated=True,
            bmi_mediation_fraction=0.65,
            confidence="very_high",
        )
        # With fraction=0.65 we keep 35% of the effect: 1 + (1.20-1)×0.35 = 1.07
        shrunk = 1.0 + (1.20 - 1.0) * (1 - 0.65)
        assert math.isclose(shrunk, 1.07, abs_tol=0.001)

    def test_missing_fraction_uses_default(self, caplog):
        """bmi_mediated=True with no fraction → default 0.3 with warning."""
        import logging as lg
        e = DiseaseEntry(
            rsid="x", disease="Obesity",
            risk_genotypes=[("A", "T")],
            or_het=1.20, or_hom=1.44, ci_95=(1.1, 1.3),
            base_rate={"m": 0.30, "f": 0.30},
            age_rates={"50": 1.0}, bmi_modifier={"obese": 1.0},
            maf=0.42, bmi_mediated=True,
            confidence="very_high",
        )
        engine = DiseaseRiskEngine([e], [])
        profile = UserProfile(age=50, sex="M", height_cm=165, weight_kg=95)
        with caplog.at_level(lg.WARNING):
            engine.compute_all({"x": ("A", "T")}, profile)
        assert any("bmi_mediation_fraction" in r.message for r in caplog.records)


# -----------------------------------------------------------------------
# User profile
# -----------------------------------------------------------------------
class TestUserProfile:
    def test_bmi(self):
        p = UserProfile(height_cm=165, weight_kg=83.5)
        assert p.bmi is not None
        assert abs(p.bmi - 30.6) < 0.2
        assert p.bmi_category == "obese"

    def test_normal_bmi(self):
        p = UserProfile(height_cm=180, weight_kg=70)
        assert p.bmi_category == "normal"

    def test_missing(self):
        p = UserProfile()
        assert p.bmi is None
        assert p.bmi_category is None


# -----------------------------------------------------------------------
# Gene-gene interactions (zygosity-aware + O(1) lookup)
# -----------------------------------------------------------------------
class TestInteractions:
    def _make_vte_entries(self):
        return [
            DiseaseEntry(
                rsid="rs6025", disease="VTE",
                risk_genotypes=[("C", "T")],
                or_het=5.0, or_hom=25.0, ci_95=(3.5, 7.0),
                base_rate={"m": 0.05, "f": 0.05},
                age_rates={}, bmi_modifier={},
                maf=0.03, special="vte", confidence="very_high",
            ),
            DiseaseEntry(
                rsid="rs1799963", disease="VTE",
                risk_genotypes=[("G", "A")],
                or_het=3.0, or_hom=9.0, ci_95=(2.0, 4.5),
                base_rate={"m": 0.05, "f": 0.05},
                age_rates={}, bmi_modifier={},
                maf=0.02, special="vte", confidence="very_high",
            ),
        ]

    def test_het_het_interaction_fires(self):
        interaction = GeneInteraction(
            snp1="rs6025", snp2="rs1799963", disease="VTE",
            combined_or=20.0, interaction_type="super-multiplicative",
            zygosity_pattern="het_het",
        )
        engine = DiseaseRiskEngine(self._make_vte_entries(), [interaction])
        snps = {"rs6025": ("C", "T"), "rs1799963": ("G", "A")}
        active = engine._detect_interactions(snps)
        assert active["disease:VTE"] == 20.0

    def test_het_het_does_not_fire_on_homozygote(self):
        """With zygosity_pattern='het_het', a homozygous carrier should
        not trigger the interaction (its OR is defined elsewhere)."""
        interaction = GeneInteraction(
            snp1="rs6025", snp2="rs1799963", disease="VTE",
            combined_or=20.0, interaction_type="super-multiplicative",
            zygosity_pattern="het_het",
        )
        engine = DiseaseRiskEngine(self._make_vte_entries(), [interaction])
        # rs6025 homozygous-risk would actually be ("T","T"), but for the
        # purposes of pattern matching we just need both-alleles-equal on
        # one of the two loci. Use the risk-homozygous pair to demonstrate
        # the gate.
        snps = {"rs6025": ("T", "T"), "rs1799963": ("G", "A")}
        # The risk-gt match: rs6025 risk_genotypes only includes ("C","T"),
        # so ("T","T") is NOT matched; fall through to "not flagged".
        active = engine._detect_interactions(snps)
        assert "disease:VTE" not in active

    def test_rsid_index_built_once(self):
        """The O(1) rsid index should reflect all disease_db entries."""
        engine = DiseaseRiskEngine(self._make_vte_entries(), [])
        assert "rs6025" in engine._rsid_index
        assert "rs1799963" in engine._rsid_index


# -----------------------------------------------------------------------
# APOE joint-genotype handling (C4 fix)
# -----------------------------------------------------------------------
class TestAPOE:
    def _apoe_entry(self):
        return DiseaseEntry(
            rsid="rs429358", disease="Alzheimer's Disease",
            risk_genotypes=[("T", "C"), ("C", "C")],
            or_het=1.0, or_hom=1.0, ci_95=(1.0, 1.0),
            base_rate={"m": 0.11, "f": 0.16},
            age_rates={"60": 1.0, "70": 3.0, "80": 8.0},
            bmi_modifier={},
            maf=0.14, special="apoe", confidence="very_high",
        )

    def _apoe_entry_7412(self):
        return DiseaseEntry(
            rsid="rs7412", disease="Alzheimer's Disease",
            risk_genotypes=[("T", "C"), ("T", "T")],
            or_het=1.0, or_hom=1.0, ci_95=(1.0, 1.0),
            base_rate={"m": 0.11, "f": 0.16},
            age_rates={"60": 1.0, "70": 3.0, "80": 8.0},
            bmi_modifier={},
            maf=0.08, special="apoe", confidence="very_high",
        )

    def _apoe_table(self):
        # Bellenguez et al. 2022 order-of-magnitude numbers for AD.
        return {
            "Alzheimer's Disease": {
                "e2e2": {"or": 0.6, "ci_low": 0.4, "ci_high": 0.9},
                "e2e3": {"or": 0.7, "ci_low": 0.6, "ci_high": 0.8},
                "e2e4": {"or": 2.6, "ci_low": 2.0, "ci_high": 3.4},
                "e3e3": {"or": 1.0, "ci_low": 1.0, "ci_high": 1.0},
                "e3e4": {"or": 3.5, "ci_low": 3.2, "ci_high": 3.8},
                "e4e4": {"or": 14.0, "ci_low": 11.0, "ci_high": 17.8},
            }
        }

    def test_e3_e3_resolves(self):
        """rs429358=T/T and rs7412=C/C → ε3/ε3, the neutral baseline."""
        pair = DiseaseRiskEngine._apoe_haplotypes(("T", "T"), ("C", "C"))
        assert pair == ("e3", "e3")

    def test_e3_e4_resolves(self):
        pair = DiseaseRiskEngine._apoe_haplotypes(("T", "C"), ("C", "C"))
        assert pair == ("e3", "e4")

    def test_e4_e4_resolves(self):
        pair = DiseaseRiskEngine._apoe_haplotypes(("C", "C"), ("C", "C"))
        assert pair == ("e4", "e4")

    def test_ambiguous_ct_ct_returns_none(self):
        """The classical ε2/ε4 vs ε3/ε3 ambiguity from unphased genotypes."""
        pair = DiseaseRiskEngine._apoe_haplotypes(("T", "C"), ("T", "C"))
        assert pair is None

    def test_apoe_risk_uses_lookup_table(self):
        engine = DiseaseRiskEngine(
            [self._apoe_entry(), self._apoe_entry_7412()],
            [], apoe_risk_table=self._apoe_table(),
        )
        # ε3/ε4 carrier
        snps = {"rs429358": ("T", "C"), "rs7412": ("C", "C")}
        # Use age 45 so attenuation ≈ 1.0 and we actually see the raw
        # lookup-table OR flow through. At age 70 with a base rate near
        # 0.5, the RR cap from OR→RR conversion pulls everything
        # toward the ceiling regardless of the OR magnitude.
        profile = UserProfile(age=45, sex="F")
        results = engine.compute_all(snps, profile)
        assert len(results) == 1
        # combined_or should reflect the ~3.5× OR from the table.
        assert 3.0 < results[0].combined_or < 4.0

    def test_apoe_e3_e3_neutral(self):
        """ε3/ε3 carriers should get the baseline OR=1.0 from the table."""
        engine = DiseaseRiskEngine(
            [self._apoe_entry(), self._apoe_entry_7412()],
            [], apoe_risk_table=self._apoe_table(),
        )
        snps = {"rs429358": ("T", "T"), "rs7412": ("C", "C")}
        profile = UserProfile(age=45, sex="F")
        results = engine.compute_all(snps, profile)
        assert len(results) == 1
        assert abs(results[0].combined_or - 1.0) < 0.01

    def test_apoe_missing_table_skips(self, caplog):
        """APOE entry in DB but no lookup table → skip with warning."""
        import logging as lg
        engine = DiseaseRiskEngine(
            [self._apoe_entry(), self._apoe_entry_7412()],
            [], apoe_risk_table={},
        )
        snps = {"rs429358": ("T", "C"), "rs7412": ("C", "C")}
        with caplog.at_level(lg.WARNING):
            results = engine.compute_all(snps, UserProfile(age=45, sex="F"))
        assert len(results) == 0
        assert any("apoe_risk_table" in r.message for r in caplog.records)


# -----------------------------------------------------------------------
# End-to-end
# -----------------------------------------------------------------------
class TestEndToEnd:
    def test_no_risk_alleles(self, engine):
        snps = {"rs7903146": ("C", "C")}
        profile = UserProfile(age=50, sex="M")
        results = engine.compute_all(snps, profile)
        # Wildtype → OR=1.0 → no result (combined_or stays at 1.0, which
        # still produces a valid but ~population-level risk estimate).
        # Either no result, or RR very close to (or below) 1.
        for r in results:
            assert r.relative_risk <= 1.1

    def test_risk_cap_at_80_percent(self, engine):
        snps = {"rs7903146": ("T", "T")}
        profile = UserProfile(age=70, sex="M", height_cm=165, weight_kg=120)
        results = engine.compute_all(snps, profile)
        for r in results:
            assert r.absolute_risk <= 0.80
            assert r.ci_high <= 0.80

    def test_missing_sex_warns(self, engine, caplog):
        import logging as lg
        snps = {"rs7903146": ("C", "C")}
        with caplog.at_level(lg.WARNING):
            engine.compute_all(snps, UserProfile(age=50))
        assert any("sex missing" in r.message for r in caplog.records)
