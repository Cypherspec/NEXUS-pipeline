# NEXUS, as a Mantis extension

A packaged Mantis extension: a panel, an extension host, and a Python backend
that is the `nexus` package itself. Installing it changes **no file in the
Mantis repository** — it goes in through Verticals → Import Extension, which is
the mechanism Mantis provides for exactly this.

## What it does on the map

The extension SDK has no API for colouring points, so results are surfaced the
way Mantis itself surfaces a subset:

| Result | How it reaches the map |
| --- | --- |
| A scaffold family | `bags.create` then `bags.highlight` — the family becomes a named bag |
| The proposed assay batch | `selection.set` — the batch becomes the selection |
| Chemical neighbours | `edges.set` on `point.clicked` — an overlay owned by this panel |

Each is released when the panel closes.

## Backend actions

`backend/main.py` dispatches four actions into the vendored `nexus` package.
Mantis caps a backend invocation at 30 seconds, so each action bounds its own
input and reports what it dropped rather than working an unbounded selection.

| Action | Returns |
| --- | --- |
| `analyze` | 24 descriptors per molecule, Ro5/Veber status, scaffold families |
| `neighbors` | Tanimoto nearest neighbours of one molecule |
| `screen` | An acquisition-ranked batch, diversity-penalised |
| `predict` | Predictions with split-conformal intervals, constant-width or adaptive |

## Build

```bash
node build.mjs
```

Bundles the panel, vendors `nexus/` under `backend/`, checks the limits Mantis
enforces (25 MB package, 5 MB per file, UTF-8 text only — the package may carry
no binary, so a stray `.pyc` fails the build rather than the import), and writes
`nexus-drug-discovery-<version>.zip`.

## Install

1. Open a space in Mantis.
2. Verticals → **Import Extension**.
3. Choose `nexus-drug-discovery-0.1.0.zip`.
4. Review the trust dialog — it names the six permissions and the backend.
5. **Trust Publisher & Install**, then open NEXUS from the Extensions section.

## Permissions, and why each is needed

| Permission | Used for |
| --- | --- |
| `maps:read` | Reading the points and their metadata, and the click event |
| `selection:read` | Reading the current selection and following it |
| `selection:write` | Selecting a proposed batch; highlighting a scaffold bag |
| `bags:write` | Creating the scaffold-family bag |
| `edges:write` | The neighbour overlay |
| `backend:invoke` | Calling the four Python actions |

Nothing is declared that the code does not call, which is the rule the install
validator reads the other way round.
