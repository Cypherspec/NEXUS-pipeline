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

## Running the whole stack locally

Both Mantis repositories are used read-only; nothing in either is modified.

```bash
# Postgres needs pgvector >= 0.7 (halfvec). Ubuntu ships 0.6, so build 0.8:
#   git clone --branch v0.8.0 https://github.com/pgvector/pgvector && make && make install
pg_ctlcluster 16 main start
redis-server --daemonize yes --dir /tmp
createdb -O mantis mantis_dev && psql -d mantis_dev -c 'CREATE EXTENSION vector'

# MantisAPI, in a venv kept outside the repo
uv venv --python 3.11 ~/mantis-venv
VIRTUAL_ENV=~/mantis-venv uv pip sync --index-strategy unsafe-best-match src/requirements-cpu.txt
DATABASE_URL=postgresql://mantis:mantis@localhost:5432/mantis_dev DJANGO_LOCAL=True \
  ~/mantis-venv/bin/python manage.py migrate && ... manage.py runserver 0.0.0.0:8000

# Frontend. NEXTAUTH_SECRET must equal the API's TOKEN_SECRET, and the backend
# URL must share a host with the frontend: the NextAuth cookie is set on
# `localhost`, and a browser will not send it to `127.0.0.1`.
NEXTAUTH_SECRET=<same as API TOKEN_SECRET> NEXT_PUBLIC_BACKEND_URL=http://localhost:8000 yarn dev
```

`seed_mantis_space.py` then creates a space from the ESOL set, laid out by a PCA
of Morgan fingerprints computed by this package rather than by a text embedding
service — the normal ingestion path needs vLLM or an OpenAI key, and neither is
what a chemical map wants anyway.

## Harness

`harness/` drives the built panel outside Mantis so it can be exercised and
screenshotted: `bridge.py` serves the real `backend/main.py` over HTTP, and
`mantisShim.js` implements only the SDK surface the panel calls. The bundle,
the stylesheets and the Python are the installed ones; nothing in `harness/`
ships inside the package.

```bash
python harness/bridge.py --port 8770      # the real backend, over HTTP
node harness/shoot.mjs                    # drive the panel, write screenshots
```

It earns its place: the first run through it caught that the panel's chart
components use Tailwind utility classes, which the host supplies in-tree but an
extension iframe does not, so the installed panel was laying out with inert
classes. `src/utilities.css` now ships the enumerated set those components use.
