"""Parser for GO Annotation File (GAF 2.x) data."""

from __future__ import annotations

import gzip
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# Evidence code groups (Gene Ontology Consortium).
EXPERIMENTAL = {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP",
                "HGI", "HEP"}
PHYLOGENETIC = {"IBA", "IBD", "IKR", "IRD"}
COMPUTATIONAL = {"ISS", "ISO", "ISA", "ISM", "IGC", "RCA"}
AUTHOR = {"TAS", "NAS"}
CURATOR = {"IC"}
ELECTRONIC = {"IEA"}
NO_DATA = {"ND"}

ASPECTS = {"F": "molecular_function", "P": "biological_process",
           "C": "cellular_component"}


@dataclass(frozen=True)
class Annotation:
    gene: str            # DB object symbol
    uniprot: str         # DB object ID
    go_id: str
    evidence: str
    aspect: str          # F, P or C
    negated: bool
    taxon: str


def _open(path: str | Path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_gaf(
    path: str | Path,
    allowed_evidence: set[str] | None = None,
    exclude_electronic: bool = False,
    include_negated: bool = False,
    taxon: str | None = None,
) -> list[Annotation]:
    """Parse a GAF file into annotations.

    By default every evidence code except ND (no biological data) is kept.
    `exclude_electronic=True` additionally drops IEA, the standard choice when
    only curated annotations should count.
    """
    out: list[Annotation] = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 15:
                continue
            qualifier, go_id, evidence, aspect = f[3], f[4], f[6], f[8]
            negated = "NOT" in qualifier.split("|")
            if evidence in NO_DATA:
                continue
            if exclude_electronic and evidence in ELECTRONIC:
                continue
            if allowed_evidence is not None and evidence not in allowed_evidence:
                continue
            if not include_negated and negated:
                continue
            if taxon is not None and f[12] != taxon:
                continue
            out.append(Annotation(gene=f[2], uniprot=f[1], go_id=go_id,
                                  evidence=evidence, aspect=aspect,
                                  negated=negated, taxon=f[12]))
    return out


def gene_to_terms(annotations: list[Annotation], aspect: str | None = None
                  ) -> dict[str, set[str]]:
    """Map gene symbol to the set of GO terms directly annotated to it."""
    out: dict[str, set[str]] = defaultdict(set)
    for a in annotations:
        if aspect and a.aspect != aspect:
            continue
        out[a.gene].add(a.go_id)
    return dict(out)


def term_to_genes(annotations: list[Annotation], aspect: str | None = None
                  ) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for a in annotations:
        if aspect and a.aspect != aspect:
            continue
        out[a.go_id].add(a.gene)
    return dict(out)
