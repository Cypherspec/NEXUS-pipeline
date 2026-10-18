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
    CoverageBar, Histogram, IntervalStrip, RankedBars, RuleDots, ScatterPlot,
    StatTile, useNumberFormat, type ScatterDatum,
} from '../../mantis/src/app/components/panels/nexus/charts';
import { gradientText, nexusTheme } from '../../mantis/src/app/components/panels/nexus/theme';

/** Metadata keys searched, in order, for a structure string. */
const SMILES_KEYS = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure', 'mol'];

/** Bag this panel writes scaffold families into, so it only clears its own. */
const BAG_PREFIX = 'NEXUS scaffold';

type Mode = 'analyze' | 'screen' | 'predict';
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

    const [labelColumn, setLabelColumn] = useState('');
    const [batchSize, setBatchSize] = useState(24);
    const [adaptive, setAdaptive] = useState(false);
    const [linkNeighbours, setLinkNeighbours] = useState(true);
    const [colourBy, setColourBy] = useState('drug_likeness');
    const [note, setNote] = useState<string | null>(null);

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
                setAnalysis(await window.mantis.backend.invoke('analyze', { molecules }));
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
    }, [adaptive, batchSize, mapId, measured, mode, molecules]);

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

    return (
        <div className="nexus-root" style={{ background: theme.bg, color: theme.text }}>
            <header style={{ borderBottom: `1px solid ${theme.line}`, background: theme.surface }}>
                <div className="nexus-head">
                    <span className="nexus-mark" style={gradientText(theme)}>NEXUS</span>
                    <span className="nexus-chip"
                        style={{ background: theme.surfaceInset, border: `1px solid ${theme.line}`,
                            color: theme.text3 }}>
                        extension
                    </span>
                    <span className="nexus-count" style={{ color: theme.text3 }}>
                        {molecules.length} molecule{molecules.length === 1 ? '' : 's'} in scope
                    </span>
                </div>
                <p className="nexus-sub" style={{ color: theme.text3 }}>
                    Non-equilibrium exploration with cross-modal unified synthesis
                </p>
            </header>

            <div className="nexus-controls" style={{ borderBottom: `1px solid ${theme.line}`,
                background: theme.surface }}>
                <div className="nexus-tabs" style={{ background: theme.surfaceInset,
                    border: `1px solid ${theme.line}` }}>
                    {(['analyze', 'screen', 'predict'] as const).map(key => (
                        <button key={key} type="button" onClick={() => setMode(key)}
                            aria-pressed={mode === key}
                            style={mode === key
                                ? { background: theme.accent, color: '#fff' }
                                : { color: theme.text2 }}>
                            {key === 'analyze' ? 'Analyze' : key === 'screen' ? 'Next batch' : 'Predict'}
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
                                : 'Predict unmeasured'}
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
                    <div className="nexus-stack">
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
                    <div className="nexus-stack">
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
                    <div className="nexus-stack">
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

                {phase === 'idle' && !analysis && !batch && !prediction && (
                    <div className="nexus-stack">
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
