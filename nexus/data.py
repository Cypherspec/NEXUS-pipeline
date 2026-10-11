"""Public dataset loaders (MoleculeNet, DeepDTA). Everything is cached locally."""

from __future__ import annotations

import csv
import gzip
import io
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent.parent / "data" / "molnet"
MOLNET = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    smiles_column: str
    task_columns: tuple[str, ...]
    task_type: str          # "classification" or "regression"
    description: str = ""

    @property
    def n_tasks(self) -> int:
        return len(self.task_columns)


DATASETS: dict[str, DatasetSpec] = {
    "bace": DatasetSpec("bace", f"{MOLNET}/bace.csv", "mol", ("Class",),
                        "classification", "BACE-1 inhibition (binary)"),
    "bbbp": DatasetSpec("bbbp", f"{MOLNET}/BBBP.csv", "smiles", ("p_np",),
                        "classification", "Blood-brain barrier penetration"),
    "hiv": DatasetSpec("hiv", f"{MOLNET}/HIV.csv", "smiles", ("HIV_active",),
                       "classification", "HIV replication inhibition"),
    "clintox": DatasetSpec("clintox", f"{MOLNET}/clintox.csv.gz", "smiles",
                           ("FDA_APPROVED", "CT_TOX"), "classification",
                           "Clinical trial toxicity"),
    "tox21": DatasetSpec(
        "tox21", f"{MOLNET}/tox21.csv.gz", "smiles",
        ("NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"),
        "classification", "Tox21 nuclear receptor and stress response panels"),
    "esol": DatasetSpec("esol", f"{MOLNET}/delaney-processed.csv", "smiles",
                        ("measured log solubility in mols per litre",), "regression",
                        "Aqueous solubility (log mol/L)"),
    "freesolv": DatasetSpec("freesolv", f"{MOLNET}/SAMPL.csv", "smiles", ("expt",),
                            "regression", "Hydration free energy (kcal/mol)"),
    "lipo": DatasetSpec("lipo", f"{MOLNET}/Lipophilicity.csv", "smiles", ("exp",),
                        "regression", "Octanol/water distribution coefficient logD"),
}


def _download(url: str, path: Path, attempts: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            path.write_bytes(data)
            return
        except (urllib.error.URLError, TimeoutError):
            if i == attempts - 1:
                raise
            time.sleep(2 ** (i + 1))


def _read_csv(spec: DatasetSpec) -> list[dict[str, str]]:
    path = CACHE / Path(spec.url).name
    if not path.exists():
        _download(spec.url, path)
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text)))


def load_dataset(name: str, parse: bool = True):
    """Return (molecules | smiles, Y, spec). Y is (n, n_tasks) with NaN for missing.

    Rows whose SMILES cannot be parsed are dropped and counted in spec-level
    statistics returned by `load_report`.
    """
    from nexus.chem.smiles import read_smiles

    spec = DATASETS[name]
    rows = _read_csv(spec)
    mols, ys, kept_smiles = [], [], []
    n_unparseable = 0
    for row in rows:
        smi = (row.get(spec.smiles_column) or "").strip()
        if not smi:
            continue
        y = []
        for col in spec.task_columns:
            val = (row.get(col) or "").strip()
            y.append(float(val) if val not in ("", "nan", "NA") else np.nan)
        if parse:
            mol = read_smiles(smi)
            if mol is None or mol.n_atoms == 0:
                n_unparseable += 1
                continue
            mols.append(mol)
        kept_smiles.append(smi)
        ys.append(y)
    Y = np.array(ys, dtype=float)
    out = mols if parse else kept_smiles
    return out, Y, spec, {"n_rows": len(rows), "n_unparseable": n_unparseable,
                          "n_kept": len(ys)}


def load_davis():
    """DeepDTA Davis kinase-inhibitor binding affinities (Kd -> pKd grid)."""
    base = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis"
    files = {"ligands": "ligands_can.txt", "proteins": "proteins.txt", "Y": "Y"}
    paths = {}
    for key, fn in files.items():
        p = CACHE.parent / "davis" / fn
        if not p.exists():
            _download(f"{base}/{fn}", p)
        paths[key] = p
    import json
    import pickle
    ligands = json.loads(paths["ligands"].read_text())
    proteins = json.loads(paths["proteins"].read_text())
    with open(paths["Y"], "rb") as fh:
        Y = pickle.load(fh, encoding="latin1")
    return list(ligands.values()), list(proteins.values()), np.asarray(Y, dtype=float)
