"""
Genotype File Parser
====================
Parses raw genotype files from consumer DNA testing services (AncestryDNA,
23andMe) into a standardized dictionary format. Distinguishes no-calls
from indels (I/D) — both are excluded from the returned SNP dict, but the
counts are reported separately so that downstream code can know whether
a missing rsID is "unreadable on this array" (indel) vs "failed QC on
this sample" (no-call).

Note on indels
--------------
Some clinically relevant variants (e.g., 5-HTTLPR) are insertion/deletion
polymorphisms that SNP arrays cannot resolve directly. Conflating these
with no-calls hides a real limitation of the data source; the
``ParseReport`` returned by :func:`parse_ancestry_file_detailed` surfaces
the split for transparent reporting in Methods sections.

Author: Pranitha Kondipamula
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# Alleles that mean "no signal" from the array — these are QC failures,
# not biology. The sample may need re-processing.
NO_CALL_VALUES = {"0", "-", "--", "N"}

# Alleles that encode insertion/deletion state. The array could not
# resolve a biallelic SNP call; downstream SNP-centric analysis must
# skip these regardless, but they are a data-source limitation, not a
# QC failure, and should be reported distinctly.
INDEL_VALUES = {"I", "D", "II", "DD", "DI", "ID"}


@dataclass
class ParseReport:
    """
    Structured report of what the parser did.

    Exposes counts separately for indels vs no-calls so the Methods
    section can state, e.g., 'of 648,558 rsIDs, 644,201 callable SNPs,
    3,217 no-calls, 1,140 indels (unresolvable on this array)'.
    """
    callable_snps: int = 0
    no_calls: int = 0
    indels: int = 0
    malformed_lines: int = 0
    total_data_lines: int = 0
    header_lines: int = 0
    indel_rsids: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"{self.callable_snps:,} callable SNPs, {self.no_calls:,} no-calls, "
            f"{self.indels:,} indels (unresolvable on this array), "
            f"{self.malformed_lines:,} malformed lines out of "
            f"{self.total_data_lines:,} data lines."
        )


def parse_ancestry_file(
    filepath: Union[str, Path],
    min_quality: Optional[float] = None,
) -> dict[str, tuple[str, str]]:
    """
    Parse an AncestryDNA or 23andMe raw data file into {rsid: (a1, a2)}.

    Excludes no-calls and indels. For a version that also returns the
    parse counts (useful for Methods-section reporting), see
    :func:`parse_ancestry_file_detailed`.
    """
    snps, _report = parse_ancestry_file_detailed(filepath, min_quality)
    return snps


def parse_ancestry_file_detailed(
    filepath: Union[str, Path],
    min_quality: Optional[float] = None,
    collect_indel_rsids: bool = False,
) -> tuple[dict[str, tuple[str, str]], ParseReport]:
    """
    Parse raw genotype file, returning both the SNP dict and a ``ParseReport``.

    Parameters
    ----------
    filepath : str or Path
        Tab-delimited raw genotype file.
    min_quality : float, optional
        Reserved for future file formats that include per-call quality.
    collect_indel_rsids : bool
        If True, record the rsIDs of skipped indels on ``ParseReport``.
        Useful when downstream code wants to warn users that a specific
        clinically relevant indel (e.g. 5-HTTLPR) could not be called.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Genotype file not found: {filepath}")

    snps: dict[str, tuple[str, str]] = {}
    report = ParseReport()

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                report.header_lines += 1
                continue
            # 23andMe / AncestryDNA both start the data region with a header
            # line beginning with "rsid" — skip once.
            lowered = line.lower()
            if lowered.startswith("rsid") and "chromosome" in lowered:
                report.header_lines += 1
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 4:
                report.malformed_lines += 1
                continue

            report.total_data_lines += 1
            rsid = parts[0]
            a1 = parts[3].upper()
            a2 = parts[4].upper() if len(parts) > 4 else a1

            # Classify: no-call, indel, or callable SNP.
            if a1 in NO_CALL_VALUES or a2 in NO_CALL_VALUES:
                report.no_calls += 1
                continue
            if a1 in INDEL_VALUES or a2 in INDEL_VALUES:
                report.indels += 1
                if collect_indel_rsids:
                    report.indel_rsids.append(rsid)
                continue
            if len(a1) != 1 or len(a2) != 1:
                report.malformed_lines += 1
                continue
            if a1 not in "ACGT" or a2 not in "ACGT":
                report.malformed_lines += 1
                continue

            snps[rsid] = (a1, a2)
            report.callable_snps += 1

    logger.info(f"{filepath.name}: {report.summary_line()}")
    return snps, report


def flip_strand(genotype: tuple[str, str]) -> tuple[str, str]:
    """Convert genotype to opposite strand (complement)."""
    return (COMPLEMENT[genotype[0]], COMPLEMENT[genotype[1]])


def normalize_genotype(genotype: tuple[str, str]) -> tuple[str, str]:
    """Sort alleles alphabetically for consistent comparison."""
    return tuple(sorted(genotype))
