"""Seed a Mantis space from the ESOL solubility set, laid out by NEXUS.

The normal ingestion path embeds text with an external service and projects it
with UMAP; neither is reachable here, and neither is what a chemical map wants
anyway. The coordinates written below are a PCA of Morgan fingerprints computed
by the NEXUS package itself, so position on this map means chemical similarity
rather than text similarity.

Run through the MantisAPI venv:
    manage.py shell < seed_space.py
"""

import sys
import uuid

import numpy as np

sys.path.insert(0, "/home/user/NEXUS-pipeline")

from db.models import DataRow, Dataset, Idea, Map, RawEmbedding, Space, Users, UserSpaceJunction

from nexus.chem.descriptors import compute_descriptors
from nexus.chem.fingerprint import fingerprint_matrix
from nexus.chem.smiles import write_smiles
from nexus.data import load_dataset

N_POINTS = 400
EMAIL = "local@mantis.dev"

dataset_mols, labels, _, _ = load_dataset("esol")
mols = dataset_mols[:N_POINTS]
values = [float(labels[i][0]) for i in range(len(mols))]

# Coordinates: PCA of the fingerprint matrix, computed here rather than fetched.
X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
X = X - X.mean(axis=0, keepdims=True)
_, _, vt = np.linalg.svd(X, full_matrices=False)
coords = X @ vt[:2].T
# Scale to a comfortable span; the map rescales anyway, but keep it sane.
coords = 40.0 * coords / max(1e-9, np.abs(coords).max())

user, _ = Users.objects.get_or_create(
    email=EMAIL, defaults={"user_id": uuid.uuid4(), "name": "Local acceptance user"})

space = Space.objects.create(
    id=uuid.uuid4(), name="ESOL aqueous solubility (NEXUS layout)",
    visibility=Space.UNLISTED, public=False,
    ai_summary="1128-molecule aqueous solubility set; 400 shown. Point positions "
               "are a PCA of Morgan fingerprints computed by the NEXUS package, "
               "so proximity is chemical similarity.")
UserSpaceJunction.objects.create(user=user, space=space, owner=True)

dataset = Dataset.objects.create(id=uuid.uuid4())
mantis_map = Map.objects.create(
    id=uuid.uuid4(), dataset=dataset, map_name="ESOL molecules",
    description="Molecules laid out by fingerprint PCA.",
    status="ready", point_count=len(mols),
    field_types={"smiles": "semantic", "measured_logS": "numeric",
                 "mw": "numeric", "logp": "numeric", "tpsa": "numeric",
                 "drug_likeness": "numeric"},
)

# Attach the map to the space through whichever junction this schema uses.
for field in ("space", "spaces"):
    if hasattr(mantis_map, field):
        try:
            if field == "space":
                setattr(mantis_map, field, space)
                mantis_map.save(update_fields=[field])
            else:
                getattr(mantis_map, field).add(space)
            break
        except Exception as exc:                                  # noqa: BLE001
            print(f"  could not attach via {field}: {exc}")

rows, ideas, embeddings = [], [], []
for i, mol in enumerate(mols):
    smiles = write_smiles(mol)
    d = compute_descriptors(mol)
    raw = {"smiles": smiles, "measured_logS": values[i],
           "mw": round(float(d["mw"]), 2), "logp": round(float(d["logp"]), 3),
           "tpsa": round(float(d["tpsa"]), 2),
           "drug_likeness": round(float(d["drug_likeness"]), 4)}
    row = DataRow(id=uuid.uuid4(), dataset=dataset, source_id=f"esol-{i}",
                  source_position=i, raw_data=raw)
    rows.append(row)
    idea = Idea(id=uuid.uuid4(), map=mantis_map, data_row=row,
                title=mol.formula(), summary=smiles,
                embedding=[float(coords[i, 0]), float(coords[i, 1])])
    ideas.append(idea)

DataRow.objects.bulk_create(rows, batch_size=500)
Idea.objects.bulk_create(ideas, batch_size=500)

for i, idea in enumerate(ideas):
    embeddings.append(RawEmbedding(
        id=uuid.uuid4(), idea=idea, model_name="nexus-morgan-pca", field="smiles",
        dimension=2, coordinates=[float(coords[i, 0]), float(coords[i, 1])]))
RawEmbedding.objects.bulk_create(embeddings, batch_size=500)

print("SPACE_ID", space.id)
print("MAP_ID", mantis_map.id)
print("POINTS", Idea.objects.filter(map=mantis_map).count())
print("USER", user.email, user.user_id)
