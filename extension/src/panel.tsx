/**
 * The NEXUS extension panel.
 *
 * Runs in Mantis's sandboxed panel iframe and reaches the application only
 * through `window.mantis`. Everything it shows comes from the Python backend
 * in this same package, so the panel ships no chemistry of its own.
 *
 * What it writes back to the map is the part worth reading twice. The
 * extension SDK has no API for colouring points, so results are surfaced the
 * way Mantis itself surfaces a subset: a scaffold family becomes a bag and is
 * highlighted, a proposed assay batch becomes the selection, and chemical
 * neighbours of a clicked molecule become an edge overlay. Each of those is
 * owned by this panel and released when it closes.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';

import {
    CoverageBar, Histogram, IntervalStrip, PersistenceDiagram, RankedBars, RuleDots,
    ScatterPlot, StatTile, useNumberFormat,
    type PersistenceFeature, type ScatterDatum,
} from '../../mantis/src/app/components/panels/nexus/charts';
import {
    Molecule3D, Space3D, type ConformerAtom, type ConformerBond, type SpacePoint,
} from '../../mantis/src/app/components/panels/nexus/three';
import { gradientText, nexusTheme } from '../../mantis/src/app/components/panels/nexus/theme';

/** Metadata keys searched, in order, for a gene or protein identifier. */
const GENE_KEYS = ['gene', 'genes', 'symbol', 'gene_symbol', 'target', 'targets',
                   'target_gene', 'protein', 'uniprot'];

/** Metadata keys searched, in order, for a structure string. */
const SMILES_KEYS = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure', 'mol'];

/** Bag this panel writes scaffold families into, so it only clears its own. */
const BAG_PREFIX = 'NEXUS scaffold';

type Mode = 'analyze' | 'screen' | 'predict' | 'shape' | 'targets';
type Phase = 'idle' | 'running' | 'done' | 'error';

interface MantisPoint {
    id: string;
    title?: string;
    metadata?: Record<string, unknown>;
}

interface Molecule {
    id: string;
    smiles: string;
    formula: string;
    veberPass: boolean;
    [descriptor: string]: string | number | boolean | null;
}

interface AnalyzeResult {
    molecules: Molecule[];
    scaffoldFamilies: { scaffold: string; count: number; pointIds: string[] }[];
    nScaffolds: number;
    nAnalyzed: number;
    nDropped: number;
    elapsedSeconds: number;
}

interface ScreenResult {
    batch: { id: string; smiles: string; score: number }[];
    nMeasured: number;
    nPool: number;
    rationale: string;
}

interface EnrichmentTerm {
    id: string;
    name: string;
    studyHits: number;
    termSize: number;
    expected: number;
    foldEnrichment: number;
    pValue: number;
    fdr: number;
    genes: string[];
}

interface EnrichResult {
    terms: EnrichmentTerm[];
    nTested: number;
    nSignificant: number;
    nStudyGenes: number;
    nUnmatched: number;
    nResolvedNames: number;
    unmatched: string[];
    population: number;
    background: string;
    alpha: number;
    window: number[];
    aspect: string;
}

interface ShapeResult {
    smiles: string;
    formula: string;
    atoms: { symbol: string; x: number; y: number; z: number }[];
    bonds: { a: number; b: number; order: number; inRing: boolean;
             ollivier: number; forman: number }[];
    features: PersistenceFeature[];
    h1: Record<string, number>;
    curvature: { min: number; max: number; mean: number; magnitude: number };
    restraintError: number;
    ringBonds: number;
}

interface PredictResult {
    predictions: { id: string; smiles: string; prediction: number; low: number; high: number }[];
    nLabelled: number;
    nCalibration: number;
    nominalCoverage: number;
    calibrationCoverage: number;
    intervalMode: string;
    meanIntervalWidth: number | null;
}

declare global {
    interface Window { mantis: any }
}

/** Finds the structure string on a map point. */
function smilesOf(point: MantisPoint): string | null {
    const meta = (point.metadata ?? {}) as Record<string, unknown>;
    for (const key of SMILES_KEYS) {
        const value = meta[key];
        if (typeof value === 'string' && value.trim()) return value.trim();
    }
    return null;
}

/** Numeric metadata columns, offered as the measured-value source. */
function numericColumns(points: MantisPoint[]): string[] {
    const counts = new Map<string, number>();
    for (const point of points) {
        const meta = (point.metadata ?? {}) as Record<string, unknown>;
        for (const [key, value] of Object.entries(meta)) {
            if (SMILES_KEYS.includes(key)) continue;
            const numeric = typeof value === 'number' ? value : Number.parseFloat(String(value));
            if (Number.isFinite(numeric)) counts.set(key, (counts.get(key) ?? 0) + 1);
        }
    }
    return [...counts.entries()].filter(([, n]) => n >= 5)
        .sort((a, b) => b[1] - a[1]).map(([key]) => key);
}


/**
 * Keeps one failing view from taking the panel with it.
 *
 * A panel is a single bundle in a bare iframe: an exception thrown while
 * rendering any chart unmounts the whole tree, and what the user sees is an
 * empty dock with no indication that anything went wrong. That happened during
 * development -- a missing import blanked the panel -- and it would be
 * indistinguishable from the panel simply not loading. The boundary reports
 * what failed and leaves the rest of the panel usable.
 */
class ViewBoundary extends React.Component<
    { label: string; theme: ReturnType<typeof nexusTheme>; children: React.ReactNode },
    { error: Error | null }
> {
    state: { error: Error | null } = { error: null };

    static getDerivedStateFromError(error: Error) {
        return { error };
    }

    render() {
        const { error } = this.state;
        if (!error) return this.props.children;
        const { theme, label } = this.props;
        return (
            <div className="nexus-error"
                style={{ background: `${theme.warn}14`, color: theme.warn,
                    border: `1px solid ${theme.warn}40` }}>
                <strong>{label}</strong> could not be drawn: {error.message}
            </div>
        );
    }
}

const NexusExtensionPanel: React.FC = () => {
    // Panels get no theme signal from the host, so dark is the panel's own
    // choice and matches the Mantis workspace it is docked into.
    const theme = nexusTheme(true);
    const fmt = useNumberFormat();

    const [mode, setMode] = useState<Mode>('analyze');
    const [phase, setPhase] = useState<Phase>('idle');
    const [error, setError] = useState<string | null>(null);
    const [mapId, setMapId] = useState<string | null>(null);
    const [points, setPoints] = useState<MantisPoint[]>([]);
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [scope, setScope] = useState<'selection' | 'map'>('selection');

    const [analysis, setAnalysis] = useState<AnalyzeResult | null>(null);
    const [batch, setBatch] = useState<ScreenResult | null>(null);
    const [prediction, setPrediction] = useState<PredictResult | null>(null);
    const [shapeResult, setShapeResult] = useState<ShapeResult | null>(null);
    const [shapeTarget, setShapeTarget] = useState<string>('');
    const [enrichResult, setEnrichResult] = useState<EnrichResult | null>(null);
    const [geneColumn, setGeneColumn] = useState('');

    const [labelColumn, setLabelColumn] = useState('');
    const [batchSize, setBatchSize] = useState(24);
    const [adaptive, setAdaptive] = useState(false);
    const [linkNeighbours, setLinkNeighbours] = useState(true);
    const [colourBy, setColourBy] = useState('drug_likeness');
    const [note, setNote] = useState<string | null>(null);
    const [conformer, setConformer] = useState<{
        atoms: ConformerAtom[]; bonds: ConformerBond[]; formula: string;
        restraintError: number;
    } | null>(null);
    const [space3d, setSpace3d] = useState<{
        points: SpacePoint[]; explainedVariance: number[]; colourBy: string;
    } | null>(null);

    const disposables = useRef<any[]>([]);
    const ownedBag = useRef<string | null>(null);

    // --- Map state, kept current by subscription rather than polling.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const active = await window.mantis.maps.getActive();
                if (cancelled) return;
                const id = active && active.mapId;
                setMapId(id ?? null);
                const list = await window.mantis.maps.getPoints(id);
                if (!cancelled) setPoints(Array.isArray(list) ? list : (list?.points ?? []));
                const selection = await window.mantis.selection.get(id);
                if (!cancelled) setSelectedIds(selection?.selected ?? []);

                disposables.current.push(await window.mantis.events.subscribe(
                    'selection.changed', {}, (payload: any) => {
                        setSelectedIds(payload?.selected ?? []);
                    }));

                // A map streams its points in, so the first read can land
                // before any have arrived and report an empty map. Re-read
                // when the set of maps changes, and poll briefly at the start
                // until points show up rather than reporting zero.
                disposables.current.push(await window.mantis.events.subscribe(
                    'maps.changed', {}, async () => {
                        const refreshed = await window.mantis.maps.getPoints(id);
                        const list = Array.isArray(refreshed)
                            ? refreshed : (refreshed?.points ?? []);
                        // An empty answer here means the map is still
                        // streaming, not that it has no points. Keeping the
                        // previous read is the difference between a panel that
                        // settles on 400 molecules and one that drops to zero.
                        if (!cancelled && list.length) setPoints(list);
                    }));

                for (let attempt = 0; attempt < 10; attempt += 1) {
                    if (cancelled) break;
                    const current = await window.mantis.maps.getPoints(id);
                    const list = Array.isArray(current) ? current : (current?.points ?? []);
                    if (list.length) {
                        setPoints(list);
                        break;
                    }
                    await new Promise(resolve => setTimeout(resolve, 1500));
                }
            } catch (e) {
                if (!cancelled) setError((e as Error).message);
            }
        })();
        return () => {
            cancelled = true;
            for (const d of disposables.current) d?.dispose?.();
            disposables.current = [];
        };
    }, []);

    const pointById = useMemo(() => {
        const map = new Map<string, MantisPoint>();
        for (const p of points) map.set(p.id, p);
        return map;
    }, [points]);

    // Molecules the current scope offers the backend.
    const molecules = useMemo(() => {
        const source = scope === 'selection' && selectedIds.length
            ? selectedIds.map(id => pointById.get(id)).filter(Boolean) as MantisPoint[]
            : points;
        const out: { id: string; smiles: string }[] = [];
        for (const point of source) {
            const smiles = smilesOf(point);
            if (smiles) out.push({ id: point.id, smiles });
        }
        return out;
    }, [points, pointById, scope, selectedIds]);

    const columns = useMemo(() => numericColumns(points), [points]);

    /** Metadata columns that look like they hold gene or protein identifiers. */
    const geneColumns = useMemo(() => {
        const seen = new Set<string>();
        for (const point of points) {
            for (const key of Object.keys(point.metadata ?? {})) {
                if (GENE_KEYS.includes(key.toLowerCase())) seen.add(key);
            }
        }
        return [...seen];
    }, [points]);

    /** Gene symbols on the points in scope, deduplicated. */
    const genes = useMemo(() => {
        const column = geneColumn || geneColumns[0];
        if (!column) return [];
        const source = scope === 'selection' && selectedIds.length
            ? selectedIds.map(id => pointById.get(id)).filter(Boolean) as MantisPoint[]
            : points;
        const found = new Set<string>();
        for (const point of source) {
            const raw = (point.metadata ?? {})[column];
            if (typeof raw !== 'string') continue;
            // One cell may carry several targets, so split on the usual
            // separators. Nothing is normalised here: the backend holds the
            // ontology and resolves names against it, and a second normaliser
            // on this side would be free to drift away from that one.
            for (const piece of raw.split(/[;,|/]+/)) {
                const symbol = piece.trim();
                if (symbol) found.add(symbol);
            }
        }
        return [...found];
    }, [geneColumn, geneColumns, points, pointById, scope, selectedIds]);

    /** Measured values for the molecules in scope, read from a metadata column. */
    const measured = useMemo(() => {
        const out: Record<string, number> = {};
        if (!labelColumn) return out;
        for (const { id } of molecules) {
            const meta = (pointById.get(id)?.metadata ?? {}) as Record<string, unknown>;
            const raw = meta[labelColumn];
            const value = typeof raw === 'number' ? raw : Number.parseFloat(String(raw));
            if (Number.isFinite(value)) out[id] = value;
        }
        return out;
    }, [labelColumn, molecules, pointById]);

    // --- Chemical neighbours as an edge overlay, driven by map clicks.
    useEffect(() => {
        if (!linkNeighbours || molecules.length < 2) return;
        let disposable: any = null;
        let cancelled = false;
        (async () => {
            disposable = await window.mantis.events.subscribe('point.clicked', {}, async (payload: any) => {
                const focusId = payload?.pointId ?? payload?.point?.id;
                if (!focusId || !molecules.some(m => m.id === focusId)) return;
                try {
                    const result = await window.mantis.backend.invoke('neighbors', {
                        molecules, focusId, k: 8,
                    });
                    const to = (result?.neighbors ?? []).map((n: any) => n.id);
                    if (to.length) {
                        await window.mantis.edges.set({ from: focusId, to, predicate: 'similar to' });
                        setNote(`Linked ${to.length} nearest neighbours of the clicked molecule.`);
                    }
                } catch (e) {
                    setNote(`Neighbour lookup failed: ${(e as Error).message}`);
                }
            });
            if (cancelled) disposable?.dispose?.();
        })();
        return () => {
            cancelled = true;
            disposable?.dispose?.();
            window.mantis.edges.clear().catch(() => undefined);
        };
    }, [linkNeighbours, molecules]);

    const run = useCallback(async () => {
        if (molecules.length === 0) return;
        setPhase('running');
        setError(null);
        setNote(null);
        try {
            if (mode === 'analyze') {
                const result = await window.mantis.backend.invoke('analyze', { molecules });
                setAnalysis(result);
                // The depth projection and a conformer are extra reads, not the
                // answer, so a failure in either leaves the analysis standing.
                window.mantis.backend.invoke('space3d', { molecules, colourBy })
                    .then(setSpace3d).catch(() => setSpace3d(null));
                const first = result?.molecules?.[0]?.smiles ?? molecules[0]?.smiles;
                if (first) {
                    window.mantis.backend.invoke('conformer', { smiles: first })
                        .then(setConformer).catch(() => setConformer(null));
                }
            } else if (mode === 'screen') {
                const result: ScreenResult = await window.mantis.backend.invoke('screen', {
                    molecules, measured, batchSize, strategy: 'ei',
                });
                setBatch(result);
                // The batch is the answer, so put it where the map shows it.
                if (result.batch.length) {
                    await window.mantis.selection.set(result.batch.map(b => b.id), mapId ?? undefined);
                    setNote(`Selected the ${result.batch.length} proposed molecules on the map.`);
                }
            } else if (mode === 'targets') {
                if (genes.length === 0) {
                    throw new Error('no gene or protein identifiers on the points in scope');
                }
                setEnrichResult(await window.mantis.backend.invoke('enrich',
                    { genes, alpha: 0.05 }));
            } else if (mode === 'shape') {
                // One molecule at a time: persistence over a conformer is the
                // most expensive thing this backend does, and the answer is
                // about a single structure rather than about the set.
                const chosen = shapeTarget || selectedIds
                    .map(id => pointById.get(id))
                    .map(point => (point ? smilesOf(point) : null))
                    .find(Boolean);
                // With nothing named, hand the backend the scope and let it
                // pick the most ring-rich structure, which is the one with
                // topology worth showing.
                setShapeResult(await window.mantis.backend.invoke('shape',
                    chosen ? { smiles: chosen } : { molecules }));
            } else {
                setPrediction(await window.mantis.backend.invoke('predict', {
                    molecules, measured, alpha: 0.1, adaptive,
                }));
            }
            setPhase('done');
        } catch (e) {
            setPhase('error');
            setError((e as Error).message);
        }
    }, [adaptive, batchSize, colourBy, genes, mapId, measured, mode, molecules,
        pointById, selectedIds, shapeTarget]);

    /** Puts one scaffold family on the map as a highlighted bag. */
    const highlightFamily = useCallback(async (family: { scaffold: string; pointIds: string[] }) => {
        const name = `${BAG_PREFIX}: ${family.scaffold.slice(0, 24)}`;
        try {
            if (ownedBag.current) {
                await window.mantis.bags.unhighlight(ownedBag.current, mapId ?? undefined)
                    .catch(() => undefined);
            }
            await window.mantis.bags.create(name, family.pointIds, mapId ?? undefined);
            await window.mantis.bags.highlight(name, mapId ?? undefined);
            ownedBag.current = name;
            setNote(`Highlighted ${family.pointIds.length} molecules sharing this scaffold.`);
        } catch (e) {
            setNote(`Could not highlight the family: ${(e as Error).message}`);
        }
    }, [mapId]);

    // --- Derived chart inputs.
    const chem = useMemo(() => {
        if (!analysis) return null;
        const values = (key: string) => analysis.molecules
            .map(m => m[key]).filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
        const scatter: ScatterDatum[] = analysis.molecules
            .filter(m => typeof m.mw === 'number' && typeof m.logp === 'number')
            .map(m => ({
                x: m.mw as number, y: m.logp as number, label: m.formula,
                colourValue: typeof m[colourBy] === 'number' ? m[colourBy] as number : undefined,
            }));
        const ro5 = analysis.molecules.filter(m => (m.lipinski_violations as number) <= 1).length;
        return {
            logp: values('logp'), tpsa: values('tpsa'), quality: values('drug_likeness'),
            scatter,
            ro5Rate: analysis.molecules.length ? ro5 / analysis.molecules.length : 0,
            veberRate: analysis.molecules.length
                ? analysis.molecules.filter(m => m.veberPass).length / analysis.molecules.length : 0,
            alertRate: analysis.molecules.length
                ? analysis.molecules.filter(m => (m.structural_alerts as number) === 0).length
                  / analysis.molecules.length : 0,
            meanQuality: values('drug_likeness').reduce((a, b) => a + b, 0)
                / Math.max(1, values('drug_likeness').length),
        };
    }, [analysis, colourBy]);

    const inputStyle = {
        background: theme.surfaceInset, borderColor: theme.line, color: theme.text,
    };
    const busy = phase === 'running' || molecules.length === 0;

    // Mode, map and result in one line: the verb, the thing it acts on, and a
    // note underneath. The middle phrase is what carries the gradient, so it
    // has to be the part worth reading.
    const headline: [string, string, string] = useMemo(() => {
        const n = molecules.length;
        if (mode === 'screen') {
            return ['Next batch of', `${Math.min(batchSize, n)} molecules`,
                'Chosen for chemical spread, then written back as the selection'];
        }
        if (mode === 'predict') {
            return ['Predicts', labelColumn || 'a measured column',
                'Conformal intervals at 90%, calibrated on the measured points'];
        }
        if (mode === 'shape') {
            return ['Shape of', (shapeTarget ? `${shapeTarget.slice(0, 22)}…` : 'the most ring-rich molecule'),
                'Persistent homology and Ricci curvature on one structure'];
        }
        if (mode === 'targets') {
            return ['Targets across', `${genes.length.toLocaleString()} symbols`,
                'Gene Ontology over-representation, BH-corrected at FDR 0.05'];
        }
        return ['Chemistry of', `${n.toLocaleString()} molecule${n === 1 ? '' : 's'}`,
            'Every point is a molecule · scaffolds, descriptors and neighbours'];
    }, [batchSize, genes.length, labelColumn, mode, molecules.length, shapeTarget]);

    return (
        <div className="nexus-root" style={{ background: theme.bg, color: theme.text }}>
            <header className="nexus-hero">
                <div className="nexus-head">
                    <span className="nexus-eyebrow"><i />NEXUS</span>
                    <span className="nexus-hero-pill">
                        {molecules.length.toLocaleString()} molecule{molecules.length === 1 ? '' : 's'}
                    </span>
                </div>
                <h1 className="nexus-title">
                    {headline[0]} <span className="nexus-title-key"
                        style={gradientText(theme)}>{headline[1]}</span>
                </h1>
                <p className="nexus-sub">{headline[2]}</p>
            </header>

            <div className="nexus-controls" style={{ borderBottom: `1px solid ${theme.line}`,
                background: theme.surface }}>
                <div className="nexus-tabs" style={{ background: theme.surfaceInset,
                    border: `1px solid ${theme.line}` }}>
                    {(['analyze', 'screen', 'predict', 'shape', 'targets'] as const).map(key => (
                        <button key={key} type="button" onClick={() => setMode(key)}
                            aria-pressed={mode === key}
                            style={mode === key
                                ? { background: theme.accent, color: '#fff' }
                                : { color: theme.text2 }}>
                            {key === 'analyze' ? 'Analyze' : key === 'screen' ? 'Batch'
                                : key === 'predict' ? 'Predict'
                                    : key === 'shape' ? 'Shape' : 'Targets'}
                        </button>
                    ))}
                </div>

                <div className="nexus-row">
                    <label style={{ color: theme.text3 }}>Scope</label>
                    <select value={scope} onChange={e => setScope(e.target.value as 'selection' | 'map')}
                        style={inputStyle}>
                        <option value="selection">selection ({selectedIds.length})</option>
                        <option value="map">whole map ({points.length})</option>
                    </select>
                </div>

                {(mode === 'screen' || mode === 'predict') && (
                    <div className="nexus-row">
                        <label style={{ color: theme.text3 }}>Measured</label>
                        <select value={labelColumn} onChange={e => setLabelColumn(e.target.value)}
                            style={inputStyle}>
                            <option value="">none</option>
                            {columns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        {mode === 'screen' && (
                            <input type="number" min={1} max={200} value={batchSize}
                                onChange={e => setBatchSize(Math.max(1, Number(e.target.value) || 1))}
                                style={{ ...inputStyle, width: '3.5rem' }} />
                        )}
                        {mode === 'predict' && (
                            <label className="nexus-check" style={{ color: theme.text3 }}>
                                <input type="checkbox" checked={adaptive}
                                    onChange={e => setAdaptive(e.target.checked)} />
                                adaptive width
                            </label>
                        )}
                    </div>
                )}

                {mode === 'targets' && (
                    <div className="nexus-row">
                        <label style={{ color: theme.text3 }}>Gene column</label>
                        <select value={geneColumn || geneColumns[0] || ''}
                            onChange={e => setGeneColumn(e.target.value)}
                            style={{ ...inputStyle, maxWidth: '12rem' }}>
                            {geneColumns.length === 0 && <option value="">none found</option>}
                            {geneColumns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        <span style={{ color: theme.text3 }}>{genes.length} symbols</span>
                    </div>
                )}

                {mode === 'shape' && (
                    <div className="nexus-row">
                        <label style={{ color: theme.text3 }}>Molecule</label>
                        <select value={shapeTarget} onChange={e => setShapeTarget(e.target.value)}
                            style={{ ...inputStyle, maxWidth: '14rem' }}>
                            <option value="">most ring-rich in scope</option>
                            {molecules.slice(0, 200).map(m => (
                                <option key={m.id} value={m.smiles}>{m.smiles.slice(0, 32)}</option>
                            ))}
                        </select>
                    </div>
                )}

                <label className="nexus-check" style={{ color: theme.text3 }}>
                    <input type="checkbox" checked={linkNeighbours}
                        onChange={e => setLinkNeighbours(e.target.checked)} />
                    draw neighbours as edges when a point is clicked
                </label>

                <button type="button" className="nexus-run" onClick={run} disabled={busy}
                    style={busy
                        ? { background: theme.surfaceInset, color: theme.text3,
                            border: `1px solid ${theme.line}` }
                        : { backgroundImage: theme.gradient, color: '#06111f' }}>
                    {phase === 'running' ? 'Working…'
                        : mode === 'analyze' ? `Analyze ${molecules.length}`
                            : mode === 'screen' ? `Propose ${batchSize} to screen`
                                : mode === 'predict' ? 'Predict unmeasured'
                                    : mode === 'shape' ? 'Analyse shape'
                                        : `Enrich ${genes.length} targets`}
                </button>

                {molecules.length === 0 && (
                    <p className="nexus-hint" style={{ color: theme.text3 }}>
                        No points in scope carry a structure. NEXUS looks for a
                        metadata field named {SMILES_KEYS.slice(0, 3).join(', ')}…
                    </p>
                )}
                {note && <p className="nexus-hint" style={{ color: theme.text2 }}>{note}</p>}
            </div>

            <div className="nexus-body">
                {phase === 'error' && error && (
                    <div className="nexus-error"
                        style={{ background: `${theme.crit}1a`, color: theme.crit,
                            border: `1px solid ${theme.crit}40` }}>
                        {error}
                    </div>
                )}

                {mode === 'analyze' && analysis && chem && (
                    <div className="nexus-stack nexus-enter">
                        <div className="nexus-tiles">
                            <StatTile label="Molecules" value={String(analysis.nAnalyzed)}
                                hint={analysis.nDropped > 0
                                    ? `${analysis.nDropped} unreadable or over the cap`
                                    : `parsed in ${analysis.elapsedSeconds}s`} theme={theme} />
                            <StatTile label="Scaffold families" value={String(analysis.nScaffolds)}
                                hint={`${(analysis.nScaffolds / Math.max(1, analysis.nAnalyzed)).toFixed(2)} per molecule`}
                                theme={theme} />
                            <StatTile label="Mean drug-likeness" value={fmt(chem.meanQuality)}
                                theme={theme} />
                            <StatTile label="Ro5 pass" value={`${(chem.ro5Rate * 100).toFixed(0)}%`}
                                tone={chem.ro5Rate >= 0.8 ? 'good' : chem.ro5Rate >= 0.5 ? 'warn' : 'crit'}
                                hint="at most one violation" theme={theme} />
                        </div>

                        <RuleDots theme={theme} items={[
                            { label: 'Veber', pass: chem.veberRate >= 0.8 },
                            { label: 'Alert-free', pass: chem.alertRate >= 0.8 },
                        ]} />

                        <div className="nexus-row">
                            <label style={{ color: theme.text3 }}>Shade by</label>
                            <select value={colourBy} onChange={e => setColourBy(e.target.value)}
                                style={inputStyle}>
                                {['drug_likeness', 'tpsa', 'logp', 'mw', 'synthetic_accessibility',
                                  'fraction_csp3'].map(k => <option key={k} value={k}>{k}</option>)}
                            </select>
                        </div>

                        {space3d && (
                            <ViewBoundary label="The depth projection" theme={theme}>
                            <Space3D points={space3d.points}
                                explained={space3d.explainedVariance}
                                colourBy={space3d.colourBy}
                                theme={theme}
                                onPick={id => {
                                    const picked = analysis.molecules.find(m => m.id === id);
                                    if (picked) {
                                        window.mantis.backend.invoke('conformer',
                                            { smiles: picked.smiles })
                                            .then(setConformer).catch(() => undefined);
                                    }
                                    window.mantis.selection.set([id], mapId ?? undefined)
                                        .catch(() => undefined);
                                }} />
                            </ViewBoundary>
                        )}

                        {conformer && (
                            <ViewBoundary label="The conformer view" theme={theme}>
                                <Molecule3D atoms={conformer.atoms} bonds={conformer.bonds}
                                    formula={conformer.formula}
                                    restraintError={conformer.restraintError} theme={theme} />
                            </ViewBoundary>
                        )}

                        <ScatterPlot title="Property space" data={chem.scatter}
                            xLabel="MW" yLabel="logP" colourLabel={colourBy} theme={theme} />
                        <Histogram title="logP" values={chem.logp} digits={2} theme={theme} />
                        <Histogram title="Polar surface area" values={chem.tpsa} unit=" Å²"
                            digits={0} theme={theme} />

                        <div>
                            <RankedBars title="Scaffold families"
                                note="click a bar to highlight it on the map"
                                items={analysis.scaffoldFamilies.slice(0, 8).map(f => ({
                                    label: f.scaffold, value: f.count,
                                    sublabel: `${f.count} molecules share this core`,
                                }))}
                                valueFormat={v => String(v)} theme={theme} />
                            <div className="nexus-families">
                                {analysis.scaffoldFamilies.slice(0, 6).map(family => (
                                    <button key={family.scaffold} type="button"
                                        onClick={() => highlightFamily(family)}
                                        style={{ background: theme.surfaceInset,
                                            border: `1px solid ${theme.line}`, color: theme.text2 }}>
                                        {family.scaffold.slice(0, 18)} · {family.count}
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {mode === 'screen' && batch && (
                    <div className="nexus-stack nexus-enter">
                        <div className="nexus-tiles">
                            <StatTile label="Proposed" value={String(batch.batch.length)}
                                hint="now selected on the map" theme={theme} />
                            <StatTile label="Already measured" value={String(batch.nMeasured)}
                                hint={`${batch.nPool} still unmeasured`} theme={theme} />
                        </div>
                        <p className="nexus-hint" style={{ color: theme.text3 }}>{batch.rationale}</p>
                        <RankedBars title="Acquisition score" note="higher is more informative"
                            items={batch.batch.slice(0, 14).map((b, i) => ({
                                label: `${i + 1}. ${b.smiles}`, value: b.score, sublabel: b.smiles,
                            }))}
                            valueFormat={v => v.toFixed(3)} theme={theme} />
                    </div>
                )}

                {mode === 'predict' && prediction && (
                    <div className="nexus-stack nexus-enter">
                        <div className="nexus-tiles">
                            <StatTile label="Trained on" value={String(prediction.nLabelled)}
                                unit="labelled" theme={theme} />
                            <StatTile label="Calibrated on" value={String(prediction.nCalibration)}
                                unit="held out" theme={theme} />
                        </div>
                        <CoverageBar nominal={prediction.nominalCoverage}
                            measured={prediction.calibrationCoverage} theme={theme} />
                        <p className="nexus-hint" style={{ color: theme.text3 }}>
                            {prediction.intervalMode === 'adaptive'
                                ? 'Widths scale with the model’s own uncertainty, so they vary per molecule.'
                                : 'Every interval has the same width; tick “adaptive width” to let it vary.'}
                            {prediction.meanIntervalWidth !== null
                                && ` Mean width ${prediction.meanIntervalWidth.toFixed(2)}.`}
                        </p>
                        <IntervalStrip title="Predictions, sorted"
                            note={`${prediction.predictions.length} molecules`}
                            data={prediction.predictions.map(p => ({
                                label: p.smiles, prediction: p.prediction,
                                low: p.low, high: p.high,
                            }))} theme={theme} />
                    </div>
                )}

                {mode === 'targets' && enrichResult && (
                    <div className="nexus-stack nexus-enter">
                        <div className="nexus-tiles">
                            <StatTile label="Genes matched"
                                value={String(enrichResult.nStudyGenes)}
                                hint={enrichResult.nUnmatched > 0
                                    ? `from ${enrichResult.nResolvedNames} of ${
                                        enrichResult.nResolvedNames + enrichResult.nUnmatched
                                      } names · ${enrichResult.nUnmatched} not in the ontology`
                                    : `from ${enrichResult.nResolvedNames} names, all resolved`}
                                tone={enrichResult.nUnmatched > enrichResult.nStudyGenes / 2
                                    ? 'warn' : 'default'}
                                theme={theme} />
                            <StatTile label="Enriched terms"
                                value={String(enrichResult.nSignificant)}
                                hint={`of ${enrichResult.nTested} tested · FDR ${enrichResult.alpha}`}
                                tone={enrichResult.nSignificant > 0 ? 'good' : 'default'}
                                theme={theme} />
                        </div>

                        {enrichResult.terms.length > 0 ? (
                            <ViewBoundary label="The enrichment chart" theme={theme}>
                                <RankedBars
                                    title="Over-represented processes"
                                    note="fold enrichment · BH-adjusted"
                                    items={enrichResult.terms.slice(0, 12).map(term => ({
                                        label: term.name,
                                        value: term.foldEnrichment,
                                        sublabel: `${term.id} · ${term.studyHits}/${term.termSize} genes · FDR ${term.fdr.toExponential(1)}`,
                                    }))}
                                    valueFormat={v => `${v.toFixed(1)}×`}
                                    theme={theme} />
                            </ViewBoundary>
                        ) : (
                            <p className="nexus-hint" style={{ color: theme.text3 }}>
                                Nothing was over-represented in these targets beyond what
                                chance explains.
                            </p>
                        )}

                        <div>
                            <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.14em]"
                                style={{ color: theme.text2 }}>Terms</div>
                            <table className="w-full table-fixed text-[10px]">
                                <thead>
                                    <tr className="text-left" style={{ color: theme.text3 }}>
                                        <th className="w-[54%] font-normal">process</th>
                                        <th className="font-normal">genes</th>
                                        <th className="font-normal">fold</th>
                                        <th className="font-normal">FDR</th>
                                    </tr>
                                </thead>
                                <tbody className="tabular-nums">
                                    {enrichResult.terms.slice(0, 25).map(term => (
                                        <tr key={term.id}
                                            style={{ borderTop: `1px solid ${theme.line}` }}>
                                            <td className="truncate pr-1"
                                                title={`${term.id}: ${term.genes.join(', ')}`}>
                                                {term.name}
                                            </td>
                                            <td>{term.studyHits}/{term.termSize}</td>
                                            <td>{term.foldEnrichment.toFixed(1)}×</td>
                                            <td style={{ color: term.fdr <= enrichResult.alpha
                                                ? theme.good : theme.text3 }}>
                                                {term.fdr.toExponential(1)}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>

                        <p className="nexus-hint" style={{ color: theme.text3 }}>
                            Hypergeometric over-representation with Benjamini–Hochberg
                            control, against {enrichResult.background} ({enrichResult.population}
                            {' '}genes). The shipped ontology is the biological-process aspect
                            restricted to terms annotated to {enrichResult.window[0]}–
                            {enrichResult.window[1]} genes, which is the window
                            over-representation analysis is defined on.
                            {enrichResult.nUnmatched > 0 && ` Unmatched: ${enrichResult.unmatched.slice(0, 6).join(', ')}.`}
                        </p>
                    </div>
                )}

                {mode === 'shape' && shapeResult && (
                    <div className="nexus-stack nexus-enter">
                        <div className="nexus-tiles">
                            <StatTile label="Loops (H1)"
                                value={String(shapeResult.features.filter(f => f.dim === 1).length)}
                                hint="long-lived features are rings" theme={theme} />
                            <StatTile label="Ring bonds" value={String(shapeResult.ringBonds)}
                                hint={shapeResult.formula} theme={theme} />
                            <StatTile label="Mean curvature"
                                value={shapeResult.curvature.mean.toFixed(3)}
                                tone={Math.abs(shapeResult.curvature.mean) < 1e-9 ? 'good' : 'default'}
                                hint="zero everywhere on a cycle" theme={theme} />
                            <StatTile label="Restraint error"
                                value={shapeResult.restraintError.toFixed(3)}
                                tone={shapeResult.restraintError < 0.05 ? 'good' : 'warn'}
                                hint="conformer embedding residual" theme={theme} />
                        </div>

                        <ViewBoundary label="The curvature view" theme={theme}>
                        <Molecule3D
                            atoms={shapeResult.atoms.map(a => ({
                                ...a, radius: 0.72, aromatic: false, hydrogens: 0,
                            }))}
                            bonds={shapeResult.bonds}
                            formula={shapeResult.formula}
                            bondValues={shapeResult.bonds.map(b => b.ollivier)}
                            bondMagnitude={shapeResult.curvature.magnitude}
                            bondValueLabel="Ollivier–Ricci curvature"
                            theme={theme} />
                        </ViewBoundary>

                        <ViewBoundary label="The persistence diagram" theme={theme}>
                            <PersistenceDiagram features={shapeResult.features} theme={theme} />
                        </ViewBoundary>

                        <p className="nexus-hint" style={{ color: theme.text3 }}>
                            Persistence is computed on the 3D conformer, curvature on the bond
                            graph with an exact transportation simplex. A cycle graph has
                            curvature exactly zero, so an unstrained ring reads flat and a fused
                            or branched position does not.
                        </p>
                    </div>
                )}

                {phase === 'idle' && !analysis && !batch && !prediction && !shapeResult
                    && !enrichResult && (
                    <div className="nexus-stack nexus-enter">
                        <p style={{ color: theme.text2, fontSize: 11, lineHeight: 1.6 }}>
                            NEXUS reads the molecules behind the points on this map. Results go back
                            onto the map itself: a scaffold family becomes a highlighted bag, a
                            proposed batch becomes the selection, and the nearest chemical
                            neighbours of a clicked molecule are drawn as edges.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
};

const container = document.getElementById('root');
if (container) createRoot(container).render(<NexusExtensionPanel />);
