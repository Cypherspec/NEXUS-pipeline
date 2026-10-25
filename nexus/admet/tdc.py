"""Loaders for the Therapeutics Data Commons ADMET collection.

Eighteen endpoints spanning absorption, distribution, metabolism, excretion and
toxicity, each a curated public dataset with measured values. Files come from
TDC's Harvard Dataverse records and are cached under data/admet/, in the same
shape as nexus/data.py caches MoleculeNet.

Every endpoint names its source, because the provenance is what makes a
prediction auditable later:

    caco2_wang              Wang et al. (2016), J. Chem. Inf. Model. 56, 763
    hia_hou                 Hou et al. (2007), J. Chem. Inf. Model. 47, 2408
    pgp_broccatelli         Broccatelli et al. (2011), J. Med. Chem. 54, 1740
    bioavailability_ma      Ma et al. (2008), J. Pharm. Sci. 97, 2755
    solubility_aqsoldb      Sorkun et al. (2019), Sci. Data 6, 143
    lipophilicity_az        AstraZeneca, deposited at ChEMBL
    bbb_martins             Martins et al. (2012), J. Chem. Inf. Model. 52, 1686
    ppbr_az                 AstraZeneca, deposited at ChEMBL
    vdss_lombardo           Lombardo & Jing (2016), J. Chem. Inf. Model. 56, 2042
    cyp2c9_veith            Veith et al. (2009), Nat. Biotechnol. 27, 1050
    cyp2d6_veith            Veith et al. (2009)
    cyp3a4_veith            Veith et al. (2009)
    half_life_obach         Obach et al. (2008), Drug Metab. Dispos. 36, 1385
    clearance_hepatocyte    AstraZeneca, deposited at ChEMBL
    herg_karim              Karim et al. (2021), J. Cheminform. 13, 60
    ames                    Xu et al. (2012), J. Chem. Inf. Model. 52, 2840
    dili                    Xu et al. (2015), Toxicol. Sci. 145, 286
    ld50_zhu                Zhu et al. (2009), Chem. Res. Toxicol. 22, 1913

Units and direction differ between endpoints and are recorded on the spec
rather than normalised away, so a reported error can always be read back in the
units the assay was measured in.
"""

from __future__ import annotations

import csv
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "admet"
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile"


@dataclass(frozen=True)
class EndpointSpec:
    """One ADMET endpoint: where it comes from and how to read its values."""

    name: str
    file_id: int
    task: str                 # "regression" or "classification"
    category: str             # absorption / distribution / metabolism / excretion / toxicity
    units: str
    description: str
    #: True when a larger measured value is the more desirable one. Recorded
    #: because it is not guessable: high permeability is good, high clearance
    #: is not, and the panel needs to know which way to colour a prediction.
    higher_is_better: bool = True
    #: Applied to the raw column before use. Several endpoints are distributed
    #: as linear values whose error is multiplicative, so they are modelled in
    #: log space and reported back in it.
    transform: str = ""


ENDPOINTS: dict[str, EndpointSpec] = {
    # ---- Absorption
    "caco2_wang": EndpointSpec(
        "caco2_wang", 4259569, "regression", "absorption", "log cm/s",
        "Caco-2 cell permeability, the standard in vitro gut-wall model"),
    "hia_hou": EndpointSpec(
        "hia_hou", 4259591, "classification", "absorption", "fraction absorbed > 30%",
        "Human intestinal absorption, measured in vivo"),
    "pgp_broccatelli": EndpointSpec(
        "pgp_broccatelli", 4259597, "classification", "absorption", "inhibitor",
        "P-glycoprotein inhibition; Pgp effluxes drugs back out of the cell",
        higher_is_better=False),
    "bioavailability_ma": EndpointSpec(
        "bioavailability_ma", 4259567, "classification", "absorption", "F > 20%",
        "Oral bioavailability in humans"),
    "solubility_aqsoldb": EndpointSpec(
        "solubility_aqsoldb", 4259610, "regression", "absorption", "log mol/L",
        "Aqueous solubility, AqSolDB's curated union of nine public sets"),
    "lipophilicity_az": EndpointSpec(
        "lipophilicity_az", 4259595, "regression", "absorption", "logD 7.4",
        "Octanol/water distribution coefficient at pH 7.4"),

    # ---- Distribution
    "bbb_martins": EndpointSpec(
        "bbb_martins", 4259566, "classification", "distribution", "penetrant",
        "Blood-brain barrier penetration"),
    "ppbr_az": EndpointSpec(
        "ppbr_az", 6413140, "regression", "distribution", "% bound",
        "Plasma protein binding rate; only the unbound fraction is active",
        higher_is_better=False),
    "vdss_lombardo": EndpointSpec(
        "vdss_lombardo", 4267387, "regression", "distribution", "log L/kg",
        "Steady-state volume of distribution", transform="log10"),

    # ---- Metabolism
    "cyp2c9_veith": EndpointSpec(
        "cyp2c9_veith", 4259577, "classification", "metabolism", "inhibitor",
        "CYP2C9 inhibition, a common route to drug-drug interaction",
        higher_is_better=False),
    "cyp2d6_veith": EndpointSpec(
        "cyp2d6_veith", 4259580, "classification", "metabolism", "inhibitor",
        "CYP2D6 inhibition", higher_is_better=False),
    "cyp3a4_veith": EndpointSpec(
        "cyp3a4_veith", 4259582, "classification", "metabolism", "inhibitor",
        "CYP3A4 inhibition; CYP3A4 handles the largest share of marketed drugs",
        higher_is_better=False),

    # ---- Excretion
    "half_life_obach": EndpointSpec(
        "half_life_obach", 4266799, "regression", "excretion", "log hours",
        "Terminal half-life in humans", transform="log10"),
    "clearance_hepatocyte_az": EndpointSpec(
        "clearance_hepatocyte_az", 4266187, "regression", "excretion",
        "log uL/min/10^6 cells", "Hepatocyte intrinsic clearance",
        higher_is_better=False, transform="log10"),

    # ---- Toxicity
    "herg_karim": EndpointSpec(
        "herg_karim", 6822246, "classification", "toxicity", "blocker",
        "hERG channel blockade, the cardiac liability that withdraws drugs",
        higher_is_better=False),
    "ames": EndpointSpec(
        "ames", 4259564, "classification", "toxicity", "mutagenic",
        "Ames test for bacterial mutagenicity", higher_is_better=False),
    "dili": EndpointSpec(
        "dili", 4259585, "classification", "toxicity", "injury",
        "Drug-induced liver injury", higher_is_better=False),
    "ld50_zhu": EndpointSpec(
        "ld50_zhu", 4267146, "regression", "toxicity", "-log10(mol/kg)",
        "Acute oral toxicity in rats; higher means a smaller lethal dose",
        higher_is_better=False),
}


def _download(url: str, path: Path, attempts: int = 5) -> None:
    """Fetch with the same backoff nexus/data.py uses for MoleculeNet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            # Dataverse refuses a request with no User-Agent.
            request = urllib.request.Request(url, headers={"User-Agent": "nexus-admet"})
            with urllib.request.urlopen(request, timeout=120) as response:
                path.write_bytes(response.read())
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download {url}: {last}")


def raw_path(name: str) -> Path:
    return CACHE / f"{name}.tab"


def fetch(name: str) -> Path:
    """Return the cached file for one endpoint, downloading it once."""
    spec = ENDPOINTS[name]
    path = raw_path(name)
    if not path.exists() or path.stat().st_size == 0:
        _download(f"{DATAVERSE}/{spec.file_id}", path)
    return path


def load_endpoint(name: str, parse: bool = True):
    """Load one endpoint.

    @param name Key in `ENDPOINTS`.
    @param parse Parse the SMILES into molecules. False returns the strings,
                 which is much faster when only counting or de-duplicating.
    @returns (molecules or smiles, y, spec). Rows whose SMILES cannot be parsed
             or whose value is missing are dropped, so the two arrays always
             line up.
    """
    spec = ENDPOINTS[name]
    path = fetch(name)

    smiles: list[str] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            # TDC ships two column layouts: Drug_ID/Drug/Y and ID/X/Y.
            text = (row.get("Drug") or row.get("X") or "").strip().strip('"')
            raw = row.get("Y")
            if not text or raw in (None, "", "NA"):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if spec.transform == "log10":
                # A handful of rows carry zero or negative values that a log
                # cannot take; they are dropped rather than clipped, which
                # would invent a measurement.
                if value <= 0:
                    continue
                value = float(np.log10(value))
            smiles.append(text)
            values.append(value)

    y = np.asarray(values, dtype=float)
    if not parse:
        return smiles, y, spec

    from nexus.chem.smiles import read_smiles

    mols, keep = [], []
    for i, text in enumerate(smiles):
        mol = read_smiles(text)
        if mol is None:
            continue
        mols.append(mol)
        keep.append(i)
    return mols, y[np.asarray(keep, dtype=int)], spec


def endpoint_names(category: str | None = None) -> list[str]:
    if category is None:
        return list(ENDPOINTS)
    return [k for k, v in ENDPOINTS.items() if v.category == category]
