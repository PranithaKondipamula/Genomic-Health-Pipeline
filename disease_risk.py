"""
Disease Risk Calculator Engine
==============================
Quantitative polygenic risk estimation with Hardy-Weinberg population
normalization, SE-propagated confidence intervals, age/BMI/height/sex
adjustment, OR→RR conversion, APOE joint-genotype handling, and
gene-gene interaction detection.

Mathematical Framework
----------------------
1. Parse user genotypes from raw SNP file.
2. Match against curated disease-variant database.
3. Compute individual OR per variant (het/hom).
4. Apply BMI-mediation shrinkage using per-variant mediation fractions
   when available.
5. Multiply ORs across variants (with interaction overrides).
6. Population-normalize the combined OR under Hardy-Weinberg:
       E[log OR_i] = 2·MAF(1-MAF)·log(OR_het) + MAF²·log(OR_hom)
       log(OR_normalized) = log(OR_combined) - Σ E[log OR_i]
   where the sum is over variants the user actually contributed to.
7. Adjust base rate for age-specific incidence, BMI category, and height.
8. Apply age attenuation to the genetic log-OR.
9. Convert OR → RR: RR = OR / (1 - P₀ + P₀·OR).
10. Compute absolute risk; derive a 95% CI on the combined OR by
    propagating per-variant published log-OR standard errors, then map
    the OR bounds through the OR→RR transform.

Author: Pranitha Kondipamula
License: MIT
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# 95% CI half-width in standard errors (normal approximation).
Z_95 = 1.959963984540054


@dataclass
class DiseaseEntry:
    """
    A single disease-variant association from the curated database.

    Attributes
    ----------
    rsid : str
        dbSNP identifier.
    disease : str
        Human-readable disease label; used as a grouping key.
    risk_genotypes : list[tuple[str, str]]
        Genotypes considered risk-carrying. Zygosity is inferred at match
        time (homozygous if both alleles equal; otherwise heterozygous).
    or_het, or_hom : float
        Heterozygous and homozygous odds ratios. If or_hom is missing
        (None or 0), a multiplicative fallback or_hom = or_het**2 is used
        with a logged warning.
    ci_95 : tuple[float, float]
        Published 95% CI on the OR, used to derive the per-variant
        standard error on the log-OR via (log(high)-log(low))/(2·Z_95).
        Must be on the same scale as or_het.
    base_rate : dict[str, float]
        Sex-stratified lifetime (or reference-age) risk.
    age_rates : dict[str, float]
        Decadal multipliers on the base rate. Keys are decades as strings
        ("40", "50", ...).
    bmi_modifier : dict[str, float]
        Multiplicative adjustment on the base rate by BMI category. This
        is applied to the rate, not the OR.
    maf : float
        Minor allele frequency in the population used for normalization.
        If the patient's ancestry differs substantially from the reference
        population, prefer an ancestry-matched value.
    bmi_mediated : bool
        Whether a nontrivial fraction of this variant's effect on the
        disease is mediated through BMI. When True and a BMI is provided,
        the OR is shrunk by bmi_mediation_fraction to avoid double-counting
        with the bmi_modifier applied to the base rate.
    bmi_mediation_fraction : float | None
        Fraction of the OR attributable to BMI mediation (0-1). If None
        and bmi_mediated is True, a conservative default of 0.3 is used
        and a warning is logged. Variant-specific values should be
        sourced from published mediation analyses.
    confidence : str
        Qualitative grade: "very_high", "high", "moderate", "low".
    pmid : str
        Primary supporting PubMed ID.
    special : str | None
        Flag for locus-specific handlers. Currently recognized:
        "apoe" — joint rs429358/rs7412 genotype processed via the
        APOE ε-allele lookup table instead of the additive OR model.
    """
    rsid: str
    disease: str
    risk_genotypes: list[tuple[str, str]]
    or_het: float
    or_hom: float
    ci_95: tuple[float, float]
    base_rate: dict[str, float]
    age_rates: dict[str, float]
    bmi_modifier: dict[str, float]
    maf: float
    bmi_mediated: bool = False
    bmi_mediation_fraction: Optional[float] = None
    confidence: str = "high"
    pmid: str = ""
    special: Optional[str] = None

    # Pre-normalized risk-genotype set for O(1) matching.
    _risk_set: frozenset = field(default_factory=frozenset, init=False, repr=False)

    def __post_init__(self):
        self._risk_set = frozenset(
            tuple(sorted(rg)) for rg in self.risk_genotypes
        )


@dataclass
class GeneInteraction:
    """
    Gene-gene interaction that overrides the multiplicative OR model.

    The combined OR applies only when the configured zygosity pattern is
    met. "het_het" — the default — matches the classical Factor V Leiden
    × Prothrombin G20210A reporting convention. For homozygous-combination
    interactions, add a second GeneInteraction entry with the appropriate
    pattern and empirically-measured OR.

    Attributes
    ----------
    zygosity_pattern : str
        One of: "het_het" (both heterozygous), "any_hom" (at least one
        homozygous carrier), "both_hom", "any" (any risk genotype on
        both loci — the legacy permissive behavior).
    """
    snp1: str
    snp2: str
    disease: str
    combined_or: float
    interaction_type: str  # "super-multiplicative", "epistatic"
    zygosity_pattern: str = "het_het"
    pmid: str = ""


@dataclass
class UserProfile:
    """Patient demographic and anthropometric data."""
    age: Optional[int] = None
    sex: Optional[str] = None  # "M" or "F"
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None

    def __post_init__(self):
        if self.height_cm and self.weight_kg and self.height_cm > 0:
            self.bmi = self.weight_kg / (self.height_cm / 100) ** 2

    @property
    def bmi_category(self) -> Optional[str]:
        if self.bmi is None:
            return None
        if self.bmi < 25:
            return "normal"
        elif self.bmi < 30:
            return "overweight"
        return "obese"


@dataclass
class RiskResult:
    """Output of a single disease risk calculation."""
    disease: str
    absolute_risk: float
    population_risk: float
    relative_risk: float
    ci_low: float
    ci_high: float
    snps_found: int
    snps_total: int
    confidence: str
    combined_or: float = 1.0
    # Standard error on the combined log-OR, propagated from per-variant
    # published CIs. Useful for downstream meta-analysis.
    log_or_se: float = 0.0
    pmid: str = ""


class DiseaseRiskEngine:
    """
    Quantitative disease risk calculator.

    Implements the full 10-step pipeline documented in the module header.
    The engine is stateless with respect to user data; a single engine
    instance can score many individuals via repeated ``compute_all`` calls.
    """

    def __init__(
        self,
        disease_db: list[DiseaseEntry],
        interactions: list[GeneInteraction],
        apoe_risk_table: Optional[dict] = None,
    ):
        self.disease_db = disease_db
        self.interactions = interactions
        # Mapping of APOE ε-genotype ("e3e3", "e3e4", "e4e4", ...) to a
        # dict with keys "or", "ci_low", "ci_high" per disease. Keyed by
        # disease name at the top level.
        self.apoe_risk_table = apoe_risk_table or {}
        self._disease_groups: dict[str, list[DiseaseEntry]] = {}
        self._rsid_index: dict[str, DiseaseEntry] = {}
        self._group_entries()

    def _group_entries(self) -> None:
        """Build disease-grouped and rsid-indexed views of the database."""
        for entry in self.disease_db:
            self._disease_groups.setdefault(entry.disease, []).append(entry)
            # If the same rsid appears in multiple diseases, the first one
            # wins for interaction lookups; interactions are disease-scoped
            # anyway, so this ambiguity does not affect correctness.
            self._rsid_index.setdefault(entry.rsid, entry)

    def compute_all(
        self,
        user_snps: dict[str, tuple[str, str]],
        profile: UserProfile,
    ) -> list[RiskResult]:
        """
        Compute risk estimates for every disease in the database.

        Parameters
        ----------
        user_snps : dict
            Mapping of rsID → (allele1, allele2) from the parsed genotype
            file. Alleles are assumed to be on the same strand as the
            database entries (strand correction is the caller's job).
        profile : UserProfile
            Demographics for adjustment. Warns rather than silently
            defaulting if sex or age is missing.

        Returns
        -------
        list[RiskResult]
            Risk estimates sorted by relative risk (descending).
        """
        if profile.sex is None:
            logger.warning(
                "UserProfile.sex missing — defaulting to 'M' for base-rate "
                "lookup. This can materially shift estimates for sex-dimorphic "
                "diseases; pass an explicit value for published work."
            )
        if profile.age is None:
            logger.warning(
                "UserProfile.age missing — defaulting to 50 for age-rate "
                "and attenuation calculations. Pass an explicit value for "
                "published work."
            )
        sex = profile.sex or "M"
        age = profile.age if profile.age is not None else 50
        bmi_cat = profile.bmi_category

        height_mods = self._compute_height_modifiers(profile.height_cm)
        active_interactions = self._detect_interactions(user_snps)

        results = []
        for disease, entries in self._disease_groups.items():
            result = self._compute_single_disease(
                disease, entries, user_snps, sex, age, bmi_cat,
                height_mods, active_interactions,
            )
            if result is not None:
                results.append(result)

        results.sort(key=lambda r: r.relative_risk, reverse=True)
        return results

    def _compute_single_disease(
        self,
        disease: str,
        entries: list[DiseaseEntry],
        user_snps: dict,
        sex: str,
        age: int,
        bmi_cat: Optional[str],
        height_mods: dict[str, float],
        active_interactions: dict[str, float],
    ) -> Optional[RiskResult]:
        """Compute risk for a single disease across all associated variants."""
        base_key = "m" if sex == "M" else "f"
        base_rate = entries[0].base_rate.get(base_key, 0)
        if base_rate == 0:
            return None

        # --- APOE short-circuit: locus-specific joint handler -----------
        # APOE ε-alleles are defined by the joint genotype at rs429358 and
        # rs7412. ε4/ε4 vs ε3/ε4 vs ε2/ε4 risks are non-multiplicative, so
        # the additive-OR machinery below would produce wrong numbers.
        if any(e.special == "apoe" for e in entries):
            return self._compute_apoe_risk(
                disease, entries, user_snps, sex, age, bmi_cat, height_mods,
            )

        # --- Standard multiplicative path -------------------------------
        # We track two sets of entries:
        #   * genotyped — every entry the user has a genotype call for
        #     (regardless of whether their OR is 1.0 or not). This is the
        #     set over which the HWE normalization expectation is summed.
        #     Including hom-nonrisk carriers here is essential: their
        #     X_i = log(1) = 0 contributes nothing to log(combined_OR),
        #     but E[X_i] ≠ 0 and must still be subtracted to keep
        #     E[log normalized_OR] = 0 under random HWE sampling.
        #   * carriers — entries where user OR ≠ 1 and we propagate an SE.
        # Excluding hom-nonrisk from the normalization sum is the
        # bookkeeping bug that breaks the HWE-invariance invariant.
        snp_ors: list[float] = []
        snp_log_or_vars: list[float] = []
        genotyped: list[DiseaseEntry] = []
        snps_found = 0
        snps_total = len(entries)
        best_confidence = "low"

        for entry in entries:
            gt = user_snps.get(entry.rsid)
            if gt is None:
                continue
            snps_found += 1
            genotyped.append(entry)

            # Skip if this SNP is handled by an interaction override
            interaction_key = f"{entry.rsid}:{disease}"
            if interaction_key in active_interactions:
                continue

            user_or = self._get_user_or(gt, entry)
            if user_or is not None and user_or != 1.0:
                if entry.bmi_mediated and bmi_cat is not None:
                    frac = entry.bmi_mediation_fraction
                    if frac is None:
                        logger.warning(
                            f"{entry.rsid}: bmi_mediated=True but no "
                            "bmi_mediation_fraction provided; using default "
                            "0.3. Populate this field with a published "
                            "mediation estimate for defensible results."
                        )
                        frac = 0.3
                    user_or = 1.0 + (user_or - 1.0) * (1.0 - frac)
                snp_ors.append(user_or)
                snp_log_or_vars.append(self._entry_log_or_var(entry))

            if entry.confidence in ("very_high", "high"):
                best_confidence = entry.confidence

        if snps_found == 0:
            return None

        # Combined OR (multiplicative model), with interaction override.
        combined_or = math.prod(snp_ors) if snp_ors else 1.0
        interaction_or = active_interactions.get(f"disease:{disease}")
        if interaction_or is not None:
            combined_or = interaction_or
            log_or_var = (
                max(snp_log_or_vars) if snp_log_or_vars else (math.log(2) / Z_95) ** 2
            )
        else:
            log_or_var = sum(snp_log_or_vars)

        # Step 6: Hardy-Weinberg population normalization over genotyped
        # entries (including hom-nonrisk carriers — see above).
        combined_or = self._normalize_or(combined_or, genotyped)

        # Step 7: age-specific base rate + BMI/height modifiers on rate.
        adjusted_base = self._adjust_base_for_age(base_rate, age, entries[0])
        if bmi_cat is not None:
            bmi_rr = entries[0].bmi_modifier.get(bmi_cat, 1.0)
            adjusted_base = min(adjusted_base * bmi_rr, 0.95)
        adjusted_base = self._apply_height_modifier(
            adjusted_base, disease, height_mods
        )

        # Step 8: age attenuation of the genetic log-OR.
        final_or, atten = self._attenuate_by_age(combined_or, age)
        # Attenuation scales log-OR linearly, so variance scales by the
        # square of the attenuation factor.
        final_log_or_var = log_or_var * (atten ** 2)
        final_log_or_se = math.sqrt(final_log_or_var) if final_log_or_var > 0 else 0.0

        # Step 9: OR → RR conversion and absolute risk (capped at 80%).
        rr = final_or / (1 - adjusted_base + adjusted_base * final_or)
        absolute_risk = min(adjusted_base * rr, 0.80)

        # Step 10: propagated 95% CI on the combined OR, mapped through
        # the OR→RR transform.
        ci_low, ci_high = self._compute_ci(
            final_or, final_log_or_se, adjusted_base
        )

        return RiskResult(
            disease=disease,
            absolute_risk=absolute_risk,
            population_risk=adjusted_base,
            relative_risk=rr,
            ci_low=ci_low,
            ci_high=ci_high,
            snps_found=snps_found,
            snps_total=snps_total,
            confidence=best_confidence,
            combined_or=final_or,
            log_or_se=final_log_or_se,
            pmid=entries[0].pmid,
        )

    # ------------------------------------------------------------------
    # APOE joint-genotype handler
    # ------------------------------------------------------------------
    def _compute_apoe_risk(
        self,
        disease: str,
        entries: list[DiseaseEntry],
        user_snps: dict,
        sex: str,
        age: int,
        bmi_cat: Optional[str],
        height_mods: dict[str, float],
    ) -> Optional[RiskResult]:
        """
        Score APOE-linked risk using the joint rs429358 / rs7412 genotype.

        The two SNPs together define the ε2/ε3/ε4 alleles; the risk table
        is looked up by ε-genotype rather than multiplied across SNPs.
        Requires ``apoe_risk_table`` to be populated with at least the
        disease of interest.
        """
        gt1 = user_snps.get("rs429358")
        gt2 = user_snps.get("rs7412")
        if gt1 is None or gt2 is None:
            return None

        allele_pair = self._apoe_haplotypes(gt1, gt2)
        if allele_pair is None:
            logger.info(
                "APOE: could not resolve ε-genotype unambiguously from the "
                "SNP-array genotypes (phase ambiguity). Skipping."
            )
            return None
        e_genotype = "".join(sorted(allele_pair))  # e.g. "e3e4"

        disease_table = self.apoe_risk_table.get(disease)
        if disease_table is None:
            logger.warning(
                f"APOE entry present for {disease} but no apoe_risk_table "
                f"supplied for it; skipping."
            )
            return None
        row = disease_table.get(e_genotype)
        if row is None:
            logger.warning(
                f"APOE ε-genotype {e_genotype} not found in the risk table "
                f"for {disease}; skipping."
            )
            return None

        or_val = float(row["or"])
        ci_low_or, ci_high_or = float(row["ci_low"]), float(row["ci_high"])

        # Base rate and modifiers (same pipeline as the standard path).
        base_key = "m" if sex == "M" else "f"
        base_rate = entries[0].base_rate.get(base_key, 0)
        if base_rate == 0:
            return None
        adjusted_base = self._adjust_base_for_age(base_rate, age, entries[0])
        if bmi_cat is not None:
            adjusted_base = min(
                adjusted_base * entries[0].bmi_modifier.get(bmi_cat, 1.0), 0.95
            )
        adjusted_base = self._apply_height_modifier(
            adjusted_base, disease, height_mods
        )

        # Age attenuation of the genetic log-OR.
        final_or, atten = self._attenuate_by_age(or_val, age)
        # Propagate CI through attenuation.
        log_se = (math.log(ci_high_or) - math.log(ci_low_or)) / (2 * Z_95)
        final_log_se = log_se * atten
        rr = final_or / (1 - adjusted_base + adjusted_base * final_or)
        absolute_risk = min(adjusted_base * rr, 0.80)
        ci_low, ci_high = self._compute_ci(final_or, final_log_se, adjusted_base)

        return RiskResult(
            disease=disease,
            absolute_risk=absolute_risk,
            population_risk=adjusted_base,
            relative_risk=rr,
            ci_low=ci_low,
            ci_high=ci_high,
            snps_found=2,
            snps_total=2,
            confidence="very_high",
            combined_or=final_or,
            log_or_se=final_log_se,
            pmid=entries[0].pmid,
        )

    @staticmethod
    def _apoe_haplotypes(
        gt_rs429358: tuple[str, str],
        gt_rs7412: tuple[str, str],
    ) -> Optional[tuple[str, str]]:
        """
        Resolve APOE ε-allele pair from the two defining SNPs.

        APOE allele definitions (forward strand):
            rs429358  rs7412   → allele
              T         T      →  ε2
              T         C      →  ε3
              C         C      →  ε4

        From unphased SNP-array genotypes, the ε-allele pair is uniquely
        resolvable except in the CT/CT case, where it could represent
        either ε2/ε4 or ε3/ε3 (we return None there to flag the ambiguity).
        """
        def _count(gt: tuple[str, str], allele: str) -> int:
            return sum(1 for a in gt if a == allele)

        c_at_429 = _count(gt_rs429358, "C")
        t_at_7412 = _count(gt_rs7412, "T")

        # Each genotype has exactly 2 alleles; count combinations.
        # (c_at_429, t_at_7412) determines the unphased diplotype uniquely
        # except for (1, 1) which is the ε2/ε4 vs ε3/ε3 ambiguity.
        resolution = {
            (0, 0): ("e3", "e3"),
            (0, 1): ("e2", "e3"),
            (0, 2): ("e2", "e2"),
            (1, 0): ("e3", "e4"),
            (2, 0): ("e4", "e4"),
            (2, 1): None,
            (2, 2): None,
            (1, 2): None,
            (1, 1): None,  # true ambiguity
        }
        return resolution.get((c_at_429, t_at_7412))

    # ------------------------------------------------------------------
    # Normalization, attenuation, CI propagation
    # ------------------------------------------------------------------
    @staticmethod
    def _entry_log_or_var(entry: DiseaseEntry) -> float:
        """Per-variant variance of the log-OR from the published 95% CI."""
        low, high = entry.ci_95
        if low <= 0 or high <= 0 or high <= low:
            # Fallback: 2-fold CI-width on the OR scale.
            return (math.log(2) / Z_95) ** 2
        se = (math.log(high) - math.log(low)) / (2 * Z_95)
        return se ** 2

    @staticmethod
    def _normalize_or(
        combined_or: float, genotyped: list[DiseaseEntry]
    ) -> float:
        """
        Hardy-Weinberg population normalization of a combined OR.

        Under HWE, the expected log-OR contribution of variant *i* to a
        person drawn at random is:

            E[log OR_i] = 2·MAF(1-MAF)·log(OR_het) + MAF²·log(OR_hom)

        We subtract the sum of these expectations over variants the user
        was *genotyped* at — including hom-nonrisk carriers. Their
        log-OR of zero does not affect ``combined_or``, but their
        expectation still needs to be subtracted so that a random HWE
        draw satisfies E[log normalized_OR] = 0 at each variant
        independently (and therefore on the product too).

        Summing over the entire DB (including variants the user was not
        genotyped at) would subtract expectations the user never had a
        chance to contribute to, biasing downward with incomplete coverage.
        Conversely, summing only over variants where the user OR ≠ 1 —
        the bug — produces a systematic upward bias proportional to the
        expected log-OR at hom-nonrisk carriers.

        Parameters
        ----------
        combined_or : float
            Product of per-variant user ORs (after any interaction overrides).
        genotyped : list[DiseaseEntry]
            Every entry the user had a genotype call for at this disease.
        """
        mean_log_or = 0.0
        for e in genotyped:
            if not e.maf or not e.or_het:
                continue
            or_hom = e.or_hom if e.or_hom and e.or_hom > 0 else e.or_het ** 2
            maf = e.maf
            het_w = 2.0 * maf * (1.0 - maf)
            hom_w = maf * maf
            mean_log_or += het_w * math.log(e.or_het) + hom_w * math.log(or_hom)
        # Always normalize — protective-allele-dominant diseases can
        # produce a negative expectation, and skipping there would bias
        # results upward.
        return combined_or / math.exp(mean_log_or)

    @staticmethod
    def _adjust_base_for_age(
        base_rate: float, age: int, entry: DiseaseEntry
    ) -> float:
        """Apply age-specific incidence multiplier to the base rate."""
        age_rates = entry.age_rates
        if not age_rates:
            return base_rate
        decade = str((age // 10) * 10)
        decades = sorted(age_rates.keys(), key=int)
        if decade in age_rates:
            mult = age_rates[decade]
        elif int(decade) < int(decades[0]):
            mult = age_rates[decades[0]] * 0.5
        else:
            mult = age_rates[decades[-1]]
        return min(base_rate * mult, 0.95)

    @staticmethod
    def _attenuate_by_age(combined_or: float, age: int) -> tuple[float, float]:
        """
        Age attenuation of the genetic log-OR.

        Returns a (attenuated_or, attenuation_factor) pair so that
        callers can also attenuate the per-variant log-OR variance.

        Attenuation factors are illustrative — published PRS age-dependence
        (Jukarainen et al. 2022; Mostafavi et al. 2020) varies by trait,
        and a defensible value ideally comes from trait-specific survival
        analysis. A sensitivity analysis sweeping these values should
        accompany any published estimate.
        """
        if age < 45:
            attenuation = 1.00
        elif age < 55:
            attenuation = 0.95
        elif age < 65:
            attenuation = 0.85
        elif age < 75:
            attenuation = 0.70
        else:
            attenuation = 0.55

        log_or = math.log(max(combined_or, 0.01))
        return math.exp(log_or * attenuation), attenuation

    @staticmethod
    def _compute_height_modifiers(
        height_cm: Optional[float],
    ) -> dict[str, float]:
        """Compute height-based risk modifiers for cancer, CVD, VTE."""
        if height_cm is None:
            return {"cancer": 1.0, "cvd": 1.0, "vte": 1.0}
        delta = (height_cm - 170) / 10.0
        return {
            "cancer": max(0.5, 1.0 + 0.12 * delta),
            "cvd": max(0.5, 1.0 - 0.09 * delta),
            "vte": max(0.5, 1.0 + 0.25 * delta),
        }

    @staticmethod
    def _apply_height_modifier(
        base: float, disease: str, mods: dict[str, float]
    ) -> float:
        """Apply height modifier based on disease category."""
        cancer_diseases = {
            "Colorectal Cancer", "Melanoma", "Prostate Cancer",
            "Pancreatic Cancer",
        }
        cvd_diseases = {"Coronary Artery Disease", "MI (Lp(a)-driven)"}
        vte_diseases = {"Venous Thromboembolism"}

        if disease in cancer_diseases:
            return min(base * mods["cancer"], 0.95)
        elif disease in cvd_diseases:
            return min(base * mods["cvd"], 0.95)
        elif disease in vte_diseases:
            return min(base * mods["vte"], 0.95)
        return base

    # ------------------------------------------------------------------
    # Interactions
    # ------------------------------------------------------------------
    def _detect_interactions(
        self, user_snps: dict
    ) -> dict[str, float]:
        """
        Detect gene-gene interactions that override the multiplicative model.

        Uses a pre-built rsID→entry index for O(1) lookups and respects
        the interaction's declared zygosity_pattern (e.g. "het_het" is
        the classical Factor V Leiden × Prothrombin G20210A convention).
        """
        active = {}
        for inter in self.interactions:
            gt1 = user_snps.get(inter.snp1)
            gt2 = user_snps.get(inter.snp2)
            if gt1 is None or gt2 is None:
                continue
            e1 = self._rsid_index.get(inter.snp1)
            e2 = self._rsid_index.get(inter.snp2)
            if not (e1 and e2):
                continue
            if not (self._has_risk_gt(gt1, e1) and self._has_risk_gt(gt2, e2)):
                continue
            if not self._zygosity_matches(gt1, gt2, inter.zygosity_pattern):
                continue
            active[f"{inter.snp1}:{inter.disease}"] = inter.combined_or
            active[f"{inter.snp2}:{inter.disease}"] = inter.combined_or
            active[f"disease:{inter.disease}"] = inter.combined_or
            logger.info(
                f"Interaction detected: {inter.snp1} × {inter.snp2} "
                f"({inter.zygosity_pattern}) → combined OR={inter.combined_or} "
                f"for {inter.disease}"
            )
        return active

    @staticmethod
    def _zygosity_matches(
        gt1: tuple[str, str], gt2: tuple[str, str], pattern: str,
    ) -> bool:
        hom1 = gt1[0] == gt1[1]
        hom2 = gt2[0] == gt2[1]
        if pattern == "any":
            return True
        if pattern == "het_het":
            return (not hom1) and (not hom2)
        if pattern == "any_hom":
            return hom1 or hom2
        if pattern == "both_hom":
            return hom1 and hom2
        logger.warning(
            f"Unknown interaction zygosity_pattern '{pattern}'; "
            f"treating as permissive ('any')."
        )
        return True

    @staticmethod
    def _has_risk_gt(
        genotype: tuple[str, str], entry: DiseaseEntry
    ) -> bool:
        """Check if user genotype matches any risk genotype (strand-agnostic)."""
        return tuple(sorted(genotype)) in entry._risk_set

    @staticmethod
    def _get_user_or(
        genotype: tuple[str, str], entry: DiseaseEntry
    ) -> Optional[float]:
        """Determine user's odds ratio based on zygosity."""
        if tuple(sorted(genotype)) not in entry._risk_set:
            return 1.0
        return entry.or_hom if genotype[0] == genotype[1] else entry.or_het

    @staticmethod
    def _compute_ci(
        final_or: float,
        log_or_se: float,
        adjusted_base: float,
    ) -> tuple[float, float]:
        """
        95% CI on the absolute risk, derived by propagating per-variant
        published log-OR standard errors through:

          (a) summation across independent variants (multiplicative model),
          (b) age attenuation (linear on log-OR → squares the variance),
          (c) OR → RR transform (mapped at both bounds).

        Labeled in results as a 95% CI "assuming independent published SEs"
        — it does not capture between-study heterogeneity, and does not
        extend to the age-attenuation factor itself. A parametric bootstrap
        over per-variant log-ORs is the natural next step and a planned
        extension.
        """
        if log_or_se <= 0:
            # No usable uncertainty information — return a point estimate
            # as both bounds rather than fabricating a band.
            rr = final_or / (1 - adjusted_base + adjusted_base * final_or)
            point = max(min(adjusted_base * rr, 0.80), 0.001)
            return point, point

        log_or = math.log(max(final_or, 1e-6))
        or_low = math.exp(log_or - Z_95 * log_or_se)
        or_high = math.exp(log_or + Z_95 * log_or_se)

        rr_low = or_low / (1 - adjusted_base + adjusted_base * or_low)
        rr_high = or_high / (1 - adjusted_base + adjusted_base * or_high)
        risk_low = max(adjusted_base * rr_low, 0.001)
        risk_high = min(adjusted_base * rr_high, 0.80)
        return risk_low, risk_high
