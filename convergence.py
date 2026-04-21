"""
Genotype × Phenotype Convergence Scoring
==========================================
Cross-references genetic predictions against biomarker data to classify
each variant as actively expressing, likely silent, or contraindicated —
enabling evidence-weighted intervention decisions.

Scoring framework
-----------------
The convergence score combines three 0-1 component scores (genetic,
phenotypic, intervention) with a safety override. The weighting
(default: 0.25 / 0.50 / 0.25) is a *structural* parameterization of the
inference principle "posterior probability of active dysfunction =
prior (genetic) × likelihood update (phenotypic)" that intentionally
privileges phenotypic confirmation over genotype alone. The weights are
not derived empirically from n=1 data and should not be interpreted that
way; ``ConvergenceEngine.sensitivity_sweep`` is provided so users (and
reviewers) can check how classification stability varies across the
weight simplex.

Author: Pranitha Kondipamula
License: MIT
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ExpressionStatus(Enum):
    """Classification of genotype-phenotype convergence."""
    ACTIVELY_EXPRESSING = "actively_expressing"
    LIKELY_EXPRESSING = "likely_expressing"
    UNCERTAIN = "uncertain"
    LIKELY_SILENT = "likely_silent"
    NO_EXPRESSION = "no_expression"
    CONTRAINDICATED = "contraindicated"


class InterventionAction(Enum):
    """Evidence-weighted intervention recommendation."""
    DIAL_UP = "dial_up"
    MAINTAIN = "maintain"
    DIAL_DOWN = "dial_down"
    SKIP = "skip"
    CONTRAINDICATED = "contraindicated"


# Explicit sort priority for grouped output (contraindicated first — it's
# the clinically urgent category). False < True in Python is legacy magic;
# a named map is clearer.
_ACTION_PRIORITY: dict[InterventionAction, int] = {
    InterventionAction.CONTRAINDICATED: 0,
    InterventionAction.DIAL_UP: 1,
    InterventionAction.MAINTAIN: 2,
    InterventionAction.DIAL_DOWN: 3,
    InterventionAction.SKIP: 4,
}


@dataclass
class GeneticEvidence:
    """
    Evidence from the genotype itself.

    Notes
    -----
    ``is_population_default`` should be True only when the majority
    *genotype* in the patient's ancestry matches the patient's genotype
    at this locus — not merely when the minor allele is common. Under
    Hardy-Weinberg, that requires the major-allele homozygote frequency
    to exceed 0.5, which implies MAF < ~0.29 (since (1-MAF)² > 0.5).
    For patient-hom-minor carriers, the locus is "default" when MAF > ~0.71.
    Use ``GeneticEvidence.infer_population_default`` as a helper.
    """
    rsid: str
    gene: str
    genotype: tuple[str, str]
    gwas_replicated: bool
    effect_size_pct: float  # % trait variance explained
    penetrance: str  # "high", "moderate", "low"
    ancestry_maf: float  # MAF in patient's ancestry
    is_population_default: bool

    @staticmethod
    def infer_population_default(
        genotype: tuple[str, str],
        ancestry_maf: float,
        minor_allele: str,
    ) -> bool:
        """
        True if the patient's genotype is the majority genotype in their
        ancestry under Hardy-Weinberg.
        """
        n_minor = sum(1 for a in genotype if a == minor_allele)
        if n_minor == 0:
            # Hom-major. Majority if major-hom frequency > 0.5.
            return (1.0 - ancestry_maf) ** 2 > 0.5
        if n_minor == 2:
            # Hom-minor. Majority if minor-hom frequency > 0.5.
            return ancestry_maf ** 2 > 0.5
        # Het. Majority if 2·MAF·(1-MAF) > 0.5 (requires MAF very close to 0.5).
        return 2.0 * ancestry_maf * (1.0 - ancestry_maf) > 0.5

    @property
    def evidence_score(self) -> float:
        """0-1 score for genetic evidence strength."""
        score = 0.0
        if self.gwas_replicated:
            score += 0.4
        if self.effect_size_pct > 5:
            score += 0.3
        elif self.effect_size_pct > 1:
            score += 0.15
        if self.penetrance == "high":
            score += 0.2
        elif self.penetrance == "moderate":
            score += 0.1
        if self.is_population_default:
            # Symmetric penalty: population-default variants carry no
            # discriminatory information regardless of which direction the
            # action would otherwise go. The penalty lives at the score
            # level so both DIAL_UP and DIAL_DOWN recommendations are
            # correctly damped, not just DIAL_UP (legacy behavior).
            score -= 0.2
        return max(0.0, min(score, 1.0))


@dataclass
class PhenotypicEvidence:
    """Evidence from biomarkers, symptoms, and wearable data."""
    biomarker_name: Optional[str] = None
    biomarker_value: Optional[float] = None
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    is_out_of_range: bool = False
    trend_direction: str = "stable"  # "improving", "worsening", "stable"
    symptom_present: bool = False
    wearable_confirms: bool = False

    @property
    def expression_score(self) -> float:
        """0-1 score for phenotypic expression evidence."""
        score = 0.0
        if self.is_out_of_range:
            score += 0.4
        if self.symptom_present:
            score += 0.3
        if self.wearable_confirms:
            score += 0.2
        if self.trend_direction == "worsening":
            score += 0.1
        return min(score, 1.0)


@dataclass
class InterventionEvidence:
    """Evidence for supplement/intervention efficacy."""
    supplement_name: str
    dose: str
    has_rct_in_carriers: bool
    has_rct_general: bool
    bioavailability_concern: bool = False
    known_contraindications: list[str] = field(default_factory=list)

    @property
    def intervention_score(self) -> float:
        """0-1 score for intervention evidence."""
        score = 0.0
        if self.has_rct_in_carriers:
            score += 0.6
        elif self.has_rct_general:
            score += 0.3
        if self.bioavailability_concern:
            score -= 0.2
        return max(0.0, min(score, 1.0))


@dataclass
class SafetyCheck:
    """Safety assessment against current medications and biomarker levels."""
    current_medications: list[str] = field(default_factory=list)
    relevant_biomarker_level: Optional[float] = None
    biomarker_threshold: Optional[float] = None
    threshold_direction: str = "above"  # "above" or "below"
    is_contraindicated: bool = False
    contraindication_reason: str = ""

    def __post_init__(self):
        # If caller declares a contraindication, require a reason. Silent
        # "CONTRAINDICATED: " output with no explanation is a failure mode,
        # not a feature.
        if self.is_contraindicated and not self.contraindication_reason:
            raise ValueError(
                "SafetyCheck marked contraindicated without a "
                "contraindication_reason. Populate the reason so downstream "
                "reasoning strings are actionable."
            )

    def evaluate(self) -> bool:
        """Check if the supplement is contraindicated."""
        if self.is_contraindicated:
            return True
        if (self.relevant_biomarker_level is not None
                and self.biomarker_threshold is not None):
            if self.threshold_direction == "below":
                if self.relevant_biomarker_level < self.biomarker_threshold:
                    self.is_contraindicated = True
                    self.contraindication_reason = (
                        f"Biomarker ({self.relevant_biomarker_level}) is below "
                        f"threshold ({self.biomarker_threshold}) — intervention "
                        f"would worsen the current physiological state"
                    )
            elif self.threshold_direction == "above":
                if self.relevant_biomarker_level > self.biomarker_threshold:
                    self.is_contraindicated = True
                    self.contraindication_reason = (
                        f"Biomarker ({self.relevant_biomarker_level}) exceeds "
                        f"threshold ({self.biomarker_threshold})"
                    )
        return self.is_contraindicated


@dataclass
class ConvergenceResult:
    """Complete convergence assessment for a gene-supplement pair."""
    gene: str
    rsid: str
    supplement: str
    genetic_evidence: GeneticEvidence
    phenotypic_evidence: PhenotypicEvidence
    intervention_evidence: InterventionEvidence
    safety_check: SafetyCheck
    expression_status: ExpressionStatus = ExpressionStatus.UNCERTAIN
    recommended_action: InterventionAction = InterventionAction.MAINTAIN
    convergence_score: float = 0.0
    reasoning: str = ""

    # Default weighting: phenotypic evidence is weighted highest as a
    # structural choice (see module docstring). Overridable for
    # sensitivity analysis.
    DEFAULT_WEIGHTS = (0.25, 0.50, 0.25)  # (genetic, phenotypic, intervention)

    def compute(
        self,
        weights: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """
        Compute convergence score and recommended action.

        Parameters
        ----------
        weights : (w_g, w_p, w_i), optional
            Non-negative weights summing to 1 for the (genetic, phenotypic,
            intervention) component scores. Defaults to DEFAULT_WEIGHTS.
        """
        w_g, w_p, w_i = weights if weights is not None else self.DEFAULT_WEIGHTS

        # Safety check first — absolute override.
        if self.safety_check.evaluate():
            self.expression_status = ExpressionStatus.CONTRAINDICATED
            self.recommended_action = InterventionAction.CONTRAINDICATED
            self.convergence_score = -1.0
            self.reasoning = (
                f"CONTRAINDICATED: {self.safety_check.contraindication_reason}"
            )
            return

        gen_score = self.genetic_evidence.evidence_score
        phen_score = self.phenotypic_evidence.expression_score
        int_score = self.intervention_evidence.intervention_score

        self.convergence_score = (
            gen_score * w_g + phen_score * w_p + int_score * w_i
        )

        # Expression status is phenotype-driven (this is the whole point
        # of the pipeline — phenotype confirms or refutes the genetic
        # prediction).
        if phen_score >= 0.6:
            self.expression_status = ExpressionStatus.ACTIVELY_EXPRESSING
        elif phen_score >= 0.3:
            self.expression_status = ExpressionStatus.LIKELY_EXPRESSING
        elif gen_score >= 0.5 and phen_score < 0.1:
            self.expression_status = ExpressionStatus.LIKELY_SILENT
        elif phen_score == 0 and not self.phenotypic_evidence.symptom_present:
            self.expression_status = ExpressionStatus.NO_EXPRESSION
        else:
            self.expression_status = ExpressionStatus.UNCERTAIN

        # Action thresholds: these are posterior-probability-like cutoffs
        # on the convergence score.
        if self.convergence_score >= 0.6:
            self.recommended_action = InterventionAction.DIAL_UP
            self.reasoning = (
                f"Strong convergence: genetic evidence ({gen_score:.2f}) + "
                f"phenotypic confirmation ({phen_score:.2f}) + "
                f"intervention evidence ({int_score:.2f})"
            )
        elif self.convergence_score >= 0.35:
            self.recommended_action = InterventionAction.MAINTAIN
            self.reasoning = (
                "Moderate convergence: monitor and reassess at next draw"
            )
        elif self.convergence_score >= 0.15:
            self.recommended_action = InterventionAction.DIAL_DOWN
            self.reasoning = (
                "Weak convergence: genetic signal present but phenotype "
                "does not confirm expression"
            )
        else:
            self.recommended_action = InterventionAction.SKIP
            self.reasoning = (
                "Insufficient evidence: genetic signal weak or phenotype "
                "contradicts prediction"
            )

        # Population-default annotation — informational only. The penalty
        # is already applied symmetrically at the genetic evidence_score
        # level; we do not re-override the action on top of that because
        # the penalty already dampens both DIAL_UP (toward MAINTAIN) and
        # would dampen DIAL_DOWN (toward SKIP), which is the correct
        # symmetric behavior.
        if self.genetic_evidence.is_population_default:
            self.reasoning += (
                " [Note: variant is population default in patient's "
                "ancestry — genetic evidence is down-weighted accordingly]"
            )


class ConvergenceEngine:
    """
    Batch convergence scoring for a complete supplement protocol.
    """

    def __init__(self, pairs: list[ConvergenceResult]):
        self.pairs = pairs
        self.computed = False
        self._active_weights: tuple[float, float, float] = (
            ConvergenceResult.DEFAULT_WEIGHTS
        )

    def compute_all(
        self,
        weights: Optional[tuple[float, float, float]] = None,
    ) -> list[ConvergenceResult]:
        """Compute convergence scores for all gene-supplement pairs."""
        if weights is not None:
            self._active_weights = weights
        for pair in self.pairs:
            pair.compute(self._active_weights)
        self.computed = True
        self.pairs.sort(key=lambda p: (
            _ACTION_PRIORITY[p.recommended_action], -p.convergence_score,
        ))
        return self.pairs

    def summary(self) -> dict[str, list[ConvergenceResult]]:
        """Group results by recommended action."""
        if not self.computed:
            self.compute_all()
        groups: dict[str, list[ConvergenceResult]] = {
            "contraindicated": [], "dial_up": [], "maintain": [],
            "dial_down": [], "skip": [],
        }
        for p in self.pairs:
            groups[p.recommended_action.value].append(p)
        return groups

    @property
    def reduction_rate(self) -> float:
        """Fraction of pairs that end up SKIP or CONTRAINDICATED."""
        if not self.computed:
            self.compute_all()
        total = len(self.pairs)
        eliminated = sum(
            1 for p in self.pairs
            if p.recommended_action in (
                InterventionAction.SKIP, InterventionAction.CONTRAINDICATED,
            )
        )
        return eliminated / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Weight sensitivity analysis
    # ------------------------------------------------------------------
    def sensitivity_sweep(
        self,
        grid_step: float = 0.1,
        reference_weights: Optional[tuple[float, float, float]] = None,
    ) -> dict:
        """
        Sweep the weight simplex and report classification stability.

        At each grid point (w_g, w_p, w_i) with w_g+w_p+w_i=1 and all ≥ 0,
        recomputes every pair and records the fraction of pairs whose
        recommended action differs from the reference (DEFAULT_WEIGHTS by
        default). A well-posed scoring scheme keeps most pairs stable
        across a reasonable neighborhood; a fragile one flips easily.

        Returns
        -------
        dict with keys:
            - 'grid': list of (w_g, w_p, w_i, flip_fraction) tuples
            - 'max_flip_fraction': worst-case flip rate across the grid
            - 'reference_weights': the weights used for the baseline
            - 'n_pairs': number of pairs considered
        """
        ref = reference_weights or ConvergenceResult.DEFAULT_WEIGHTS

        # Snapshot baseline actions.
        for p in self.pairs:
            p.compute(ref)
        baseline = [p.recommended_action for p in self.pairs]

        # Enumerate all (a, b, c) with a+b+c=1 on the given grid.
        steps = int(round(1.0 / grid_step))
        points: list[tuple[float, float, float, float]] = []
        for i, j in itertools.product(range(steps + 1), repeat=2):
            k = steps - i - j
            if k < 0:
                continue
            w_g, w_p, w_i = i * grid_step, j * grid_step, k * grid_step
            for p in self.pairs:
                p.compute((w_g, w_p, w_i))
            flipped = sum(
                1 for p, b in zip(self.pairs, baseline)
                if p.recommended_action != b
            )
            points.append((w_g, w_p, w_i, flipped / max(len(self.pairs), 1)))

        # Restore the active weights on the pairs for downstream use.
        for p in self.pairs:
            p.compute(self._active_weights)

        return {
            "grid": points,
            "max_flip_fraction": max((p[3] for p in points), default=0.0),
            "reference_weights": ref,
            "n_pairs": len(self.pairs),
        }
