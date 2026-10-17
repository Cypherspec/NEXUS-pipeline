# NEXUS as a Mantis panel

This directory holds a drop-in Mantis panel that exposes the NEXUS pipeline on the
molecule points of a Mantis map. **No file in the Mantis repository is modified by this
project** — the three files here are copied into Mantis, and three short additions
register them. Those additions are listed below as exact diffs.

Verified against `KellisLab/Mantis` at commit `5277ccc`, Next.js app `mantis-ai` 0.9.25,
Node 26.

## What the panel does

The panel reads the molecules behind the points you select on a map (any metadata column
named `smiles`, `SMILES`, `canonical_smiles`, `structure`, …) and offers three modes:

| Mode | What it answers | Backed by |
|---|---|---|
| **Analyze** | What are these molecules, and how drug-like? | descriptors, Lipinski/Veber, structural alerts, Bemis-Murcko scaffolds |
| **Next batch** | Which compounds should I measure next? | GP surrogate + acquisition function, similarity-penalised batch selection |
| **Predict** | What is the property of the unmeasured ones, and how sure are we? | Tanimoto-kernel GP + conformal prediction intervals |

Analyze also paints the map: pick a descriptor and the selected points are coloured on a
sequential ramp. Colours are written into a panel-owned layer (`'nexus'`), so clearing
them or closing the panel restores whatever colouring was underneath rather than wiping
the map.

The **Predict** mode is the one worth singling out. Its intervals come from split
conformal prediction, so they carry a finite-sample coverage guarantee rather than a model
confidence score: at α = 0.1 the interval contains the true value at least 90% of the
time, whatever the model does. The panel shows the coverage measured on the held-out
calibration set beside the nominal level, so a user can see the guarantee holding.

## Files

| File | Destination in Mantis |
|---|---|
| `src/types/nexus.ts` | `src/types/nexus.ts` |
| `src/app/services/nexusApi.ts` | `src/app/services/nexusApi.ts` |
| `src/app/components/panels/NexusPanel.tsx` | `src/app/components/panels/NexusPanel.tsx` |

```bash
cp mantis/src/types/nexus.ts                          $MANTIS/src/types/
cp mantis/src/app/services/nexusApi.ts                $MANTIS/src/app/services/
cp mantis/src/app/components/panels/NexusPanel.tsx    $MANTIS/src/app/components/panels/
```

## The three registrations

**1. `src/app/components/panels.tsx`** — add the lazy import beside the other panels:

```diff
 const MantisLensesPanel = React.lazy(() => import('./panels/MantisLensesPanel'));
+const NexusPanel = React.lazy(() => import('./panels/NexusPanel'));
```

and a case in the panel factory, next to `alphagenome`:

```diff
             case 'alphagenome':
                 return (
                     <ResizableWrapper node={node}>
                         <Wrapper requirements={{ metadata: true }}>
                             <LazyPanel>
                                 <AlphaGenomePanel />
                             </LazyPanel>
                         </Wrapper>
                     </ResizableWrapper>
                 );
+            case 'nexus':
+                return (
+                    <ResizableWrapper node={node}>
+                        <Wrapper requirements={{ metadata: true }}>
+                            <LazyPanel>
+                                <NexusPanel />
+                            </LazyPanel>
+                        </Wrapper>
+                    </ResizableWrapper>
+                );
```

`requirements={{ metadata: true }}` matters: the panel reads structures out of point
metadata, so it must not render before metadata is hydrated.

**2. `src/app/components/panel_config.tsx`** — register it so it appears in the panel
list and as a selection action:

```diff
+    nexus: {
+        name: 'NEXUS',
+        component: 'nexus',
+        helpText: 'Analyze the molecules behind selected points: descriptors and drug-likeness, scaffold families, the next batch to screen, and property predictions with calibrated uncertainty.',
+        pointAction: { actionId: 'nexus_on_select', label: 'Analyze Molecules with NEXUS', category: 'analyze', minPoints: 1, priority: 30, canHandle: () => true },
+    },
```

**3. `src/app/components/layout/panelIcons.ts`** — optional; without it the panel falls
back to the default icon, as `alphagenome` does:

```diff
 const ICON_BY_COMPONENT: Record<string, LucideIcon> = {
+    nexus: FlaskConical,
```

(and add `FlaskConical` to the `lucide-react` import list).

## Running the backend

The panel talks to the NEXUS service, which is the Python package in this repository and
depends on NumPy alone:

```bash
pip install -e .
python -m nexus.server --port 8765
```

Point the front end at it by setting `NEXT_PUBLIC_NEXUS_URL` in a gitignored
`.env.development.local` (the panel defaults to `http://127.0.0.1:8765`, so for local
development nothing needs setting):

```
NEXT_PUBLIC_NEXUS_URL=http://127.0.0.1:8765
```

The header shows the service version when it is reachable and "service offline" when it is
not, and a failed request reports the URL it tried, so a missing backend is visible rather
than silent.

## Service endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | version and capability report |
| `POST /analyze` | descriptors, drug-likeness, alerts, scaffold per molecule |
| `POST /scaffolds` | Bemis-Murcko scaffold grouping |
| `POST /similarity` | Tanimoto ranking of a library against a query |
| `POST /predict` | GP prediction with conformal intervals |
| `POST /screen` | next batch to measure, by active learning |
| `POST /topology` | persistent homology and Ricci curvature descriptors |
| `POST /enrichment` | GO enrichment for a gene list |

All take and return JSON; errors come back as `{"error": "..."}` with HTTP 400.

## Verification

Typechecked against the real Mantis tree with Mantis's own `node_modules` and path
aliases, using a scratch overlay so the clone itself was never written to:

```
total errors with NEXUS panel included:   121
total errors with only Mantis's own panel: 121
errors attributable to NEXUS files:          0
```

The 121 are pre-existing errors in Mantis's `data_provider.tsx`, surfaced because the
minimal check config omits Mantis's `next` TypeScript plugin. The panel adds none.
