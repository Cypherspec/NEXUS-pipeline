"""Parser for the OBO 1.2 flat-file format used by the Gene Ontology."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OboTerm:
    id: str
    name: str = ""
    namespace: str = ""
    definition: str = ""
    is_a: list[str] = field(default_factory=list)
    part_of: list[str] = field(default_factory=list)
    regulates: list[str] = field(default_factory=list)
    alt_ids: list[str] = field(default_factory=list)
    synonyms: list[str] = field(default_factory=list)
    subsets: list[str] = field(default_factory=list)
    is_obsolete: bool = False
    replaced_by: str = ""
    consider: list[str] = field(default_factory=list)


def _unquote(value: str) -> str:
    if value.startswith('"'):
        end = 1
        while end < len(value):
            if value[end] == '"' and value[end - 1] != "\\":
                break
            end += 1
        return value[1:end].replace('\\"', '"')
    return value


def parse_obo(path: str | Path) -> tuple[dict[str, OboTerm], dict[str, str]]:
    """Parse an OBO file.

    Returns (terms by id, header fields). Only [Term] stanzas are kept;
    [Typedef] and other stanzas are skipped.
    """
    terms: dict[str, OboTerm] = {}
    header: dict[str, str] = {}
    current: OboTerm | None = None
    in_term = False

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("["):
                if current is not None:
                    terms[current.id] = current
                    current = None
                in_term = line.strip() == "[Term]"
                continue
            if ": " not in line and not line.endswith(":"):
                continue
            key, _, value = line.partition(": ")
            value = value.strip()
            if not in_term and current is None:
                header.setdefault(key, value)
                continue
            if not in_term:
                continue
            if key == "id":
                current = OboTerm(id=value)
            elif current is None:
                continue
            elif key == "name":
                current.name = value
            elif key == "namespace":
                current.namespace = value
            elif key == "def":
                current.definition = _unquote(value)
            elif key == "is_a":
                current.is_a.append(value.split(" ! ")[0].strip())
            elif key == "relationship":
                parts = value.split(" ! ")[0].split()
                if len(parts) >= 2:
                    kind, target = parts[0], parts[1]
                    if kind == "part_of":
                        current.part_of.append(target)
                    elif kind in ("regulates", "positively_regulates",
                                  "negatively_regulates"):
                        current.regulates.append(target)
            elif key == "alt_id":
                current.alt_ids.append(value)
            elif key == "synonym":
                current.synonyms.append(_unquote(value))
            elif key == "subset":
                current.subsets.append(value)
            elif key == "is_obsolete":
                current.is_obsolete = value.lower() == "true"
            elif key == "replaced_by":
                current.replaced_by = value
            elif key == "consider":
                current.consider.append(value)
    if current is not None:
        terms[current.id] = current
    return terms, header
