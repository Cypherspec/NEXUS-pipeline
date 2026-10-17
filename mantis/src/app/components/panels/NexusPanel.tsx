import {
    AlertTriangle,
    Beaker,
    CheckCircle2,
    FlaskConical,
    Loader2,
    Palette,
    RefreshCw,
    Target,
} from 'lucide-react';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { usePointColorStore } from '@/app/components/map/points/pointColorStore';
import { useDataStore } from '@/app/components/stores/dataStores';
import { useSelectionStore } from '@/app/components/stores/selectionStores';
import { useDarkMode } from '@/app/context/DarkModeContext';
import {
    analyzeMolecules,
    fetchNexusHealth,
    groupByScaffold,
    nexusBaseUrl,
    predictWithIntervals,
    proposeScreeningBatch,
} from '@/app/services/nexusApi';
import type { DataPoint } from '@/app/types';
import {
    NEXUS_COLOUR_PROPERTIES,
    NEXUS_STRATEGIES,
    type NexusAnalyzeResponse,
    type NexusPredictResponse,
    type NexusScaffoldResponse,
    type NexusScreenResponse,
    type NexusStrategy,
} from '@/types/nexus';

/** Layer this panel writes point colours into, so it only clears its own. */
const COLOR_OWNER = 'nexus';

/** Metadata keys searched, in order, for a structure string. */
const SMILES_KEYS = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure', 'mol'];

/** Sequential blue ramp, light to dark; one hue, as a magnitude scale should be. */
const RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#2a78d6', '#1c5cab', '#104281'];

type Mode = 'analyze' | 'screen' | 'predict';
type Phase = 'idle' | 'running' | 'done' | 'error';

/**
 * Finds the structure string on a map point.
 *
 * @param point Map point whose metadata may carry a structure.
 * @returns The SMILES string, or null when the point has none.
 */
function smilesOf(point: DataPoint): string | null {
    const meta = (point.metadata ?? {}) as Record<string, unknown>;
    for (const key of SMILES_KEYS) {
        const value = meta[key];
        if (typeof value === 'string' && value.trim()) return value.trim();
    }
    return null;
}

/**
 * Maps a value in [min, max] onto the sequential ramp.
 *
 * @param value Value to colour.
 * @param min Lower bound of the scale.
 * @param max Upper bound of the scale.
 * @returns A hex colour.
 */
function rampColor(value: number, min: number, max: number): string {
    if (!Number.isFinite(value) || max <= min) return RAMP[0];
    const unit = Math.min(1, Math.max(0, (value - min) / (max - min)));
    return RAMP[Math.min(RAMP.length - 1, Math.round(unit * (RAMP.length - 1)))];
}

/**
 * Formats a number for the compact result tables.
 *
 * @param value Number to format, possibly null.
 * @param digits Decimal places.
 * @returns A display string.
 */
function fmt(value: number | null | undefined, digits = 2): string {
    return value === null || value === undefined || !Number.isFinite(value)
        ? '--' : value.toFixed(digits);
}

const NexusPanel: React.FC = () => {
    const { isDarkMode } = useDarkMode();
    const mapId = useDataStore(s => s.map_id);
    const selected = useSelectionStore(s => s.selected) as DataPoint[];

    const [mode, setMode] = useState<Mode>('analyze');
    const [phase, setPhase] = useState<Phase>('idle');
    const [error, setError] = useState<string | null>(null);
    const [serviceVersion, setServiceVersion] = useState<string | null>(null);

    const [analysis, setAnalysis] = useState<NexusAnalyzeResponse | null>(null);
    const [scaffolds, setScaffolds] = useState<NexusScaffoldResponse | null>(null);
    const [batch, setBatch] = useState<NexusScreenResponse | null>(null);
    const [prediction, setPrediction] = useState<NexusPredictResponse | null>(null);

    const [colourBy, setColourBy] = useState<string>('drug_likeness');
    const [strategy, setStrategy] = useState<NexusStrategy>('ei');
    const [batchSize, setBatchSize] = useState(24);
    const [labelColumn, setLabelColumn] = useState('');

    const colouredIds = useRef<Set<string>>(new Set());
    const inFlight = useRef<AbortController | null>(null);

    const muted = isDarkMode ? 'text-gray-400' : 'text-gray-500';
    const border = isDarkMode ? 'border-gray-700' : 'border-gray-200';
    const surface = isDarkMode ? 'bg-gray-900' : 'bg-white';
    const subtle = isDarkMode ? 'bg-gray-800' : 'bg-gray-50';

    // Points carrying a structure, and the id each SMILES came from, so results
    // can be mapped back onto the map.
    const { structures, pointBySmiles } = useMemo(() => {
        const list: string[] = [];
        const bySmiles = new Map<string, string[]>();
        for (const point of selected) {
            const smiles = smilesOf(point);
            if (!smiles) continue;
            list.push(smiles);
            const ids = bySmiles.get(smiles) ?? [];
            ids.push(point.id);
            bySmiles.set(smiles, ids);
        }
        return { structures: list, pointBySmiles: bySmiles };
    }, [selected]);

    // Numeric metadata columns, offered as the label source for prediction.
    const numericColumns = useMemo(() => {
        const counts = new Map<string, number>();
        for (const point of selected) {
            const meta = (point.metadata ?? {}) as Record<string, unknown>;
            for (const [key, value] of Object.entries(meta)) {
                if (SMILES_KEYS.includes(key)) continue;
                const numeric = typeof value === 'number'
                    ? value : Number.parseFloat(String(value));
                if (Number.isFinite(numeric)) counts.set(key, (counts.get(key) ?? 0) + 1);
            }
        }
        return [...counts.entries()]
            .filter(([, n]) => n >= 5)
            .sort((a, b) => b[1] - a[1])
            .map(([key]) => key);
    }, [selected]);

    useEffect(() => {
        const controller = new AbortController();
        fetchNexusHealth(controller.signal)
            .then(health => setServiceVersion(health.version))
            .catch(() => setServiceVersion(null));
        return () => controller.abort();
    }, []);

    // Clear this panel's colours when it unmounts, so closing the panel does
    // not leave the map permanently tinted.
    useEffect(() => () => {
        usePointColorStore.getState().removeColors(colouredIds.current, COLOR_OWNER);
    }, []);

    const clearColours = useCallback(() => {
        usePointColorStore.getState().removeColors(colouredIds.current, COLOR_OWNER);
        colouredIds.current = new Set();
    }, []);

    /** Paints the map by one descriptor of the current analysis. */
    const paint = useCallback((property: string, result: NexusAnalyzeResponse | null) => {
        clearColours();
        if (!result) return;
        const values: { ids: string[]; value: number }[] = [];
        for (const molecule of result.molecules) {
            const value = molecule.descriptors[property];
            const ids = pointBySmiles.get(molecule.smiles);
            if (value === null || value === undefined || !ids) continue;
            values.push({ ids, value });
        }
        if (values.length === 0) return;
        const numbers = values.map(v => v.value);
        const min = Math.min(...numbers);
        const max = Math.max(...numbers);
        const colours = new Map<string, string>();
        for (const { ids, value } of values) {
            const colour = rampColor(value, min, max);
            for (const id of ids) colours.set(id, colour);
        }
        usePointColorStore.getState().updateMultiplePointColors(colours, COLOR_OWNER);
        colouredIds.current = new Set(colours.keys());
    }, [clearColours, pointBySmiles]);

    const run = useCallback(async () => {
        if (structures.length === 0) return;
        inFlight.current?.abort();
        const controller = new AbortController();
        inFlight.current = controller;
        setPhase('running');
        setError(null);
        try {
            if (mode === 'analyze') {
                const [result, groups] = await Promise.all([
                    analyzeMolecules(structures, {}, controller.signal),
                    groupByScaffold(structures, false, controller.signal),
                ]);
                setAnalysis(result);
                setScaffolds(groups);
                paint(colourBy, result);
            } else if (mode === 'screen') {
                const measured: Record<string, number> = {};
                if (labelColumn) {
                    for (const point of selected) {
                        const smiles = smilesOf(point);
                        if (!smiles) continue;
                        const raw = ((point.metadata ?? {}) as Record<string, unknown>)[labelColumn];
                        const value = typeof raw === 'number' ? raw : Number.parseFloat(String(raw));
                        if (Number.isFinite(value)) measured[smiles] = value;
                    }
                }
                setBatch(await proposeScreeningBatch(structures, measured,
                    { batchSize, strategy, diversity: 0.3 }, controller.signal));
            } else {
                if (!labelColumn) throw new Error('Choose a column holding the measured values.');
                const labels = structures.map(smiles => {
                    const ids = pointBySmiles.get(smiles) ?? [];
                    const point = selected.find(p => ids.includes(p.id));
                    const raw = ((point?.metadata ?? {}) as Record<string, unknown>)[labelColumn];
                    const value = typeof raw === 'number' ? raw : Number.parseFloat(String(raw));
                    return Number.isFinite(value) ? value : null;
                });
                setPrediction(await predictWithIntervals(structures, labels, 0.1,
                    controller.signal));
            }
            setPhase('done');
        } catch (e) {
            if ((e as Error).name === 'AbortError') return;
            setPhase('error');
            setError((e as Error).message);
        }
    }, [batchSize, colourBy, labelColumn, mode, paint, pointBySmiles, selected,
        strategy, structures]);

    const onColourChange = useCallback((property: string) => {
        setColourBy(property);
        paint(property, analysis);
    }, [analysis, paint]);

    const disabled = phase === 'running' || structures.length === 0;

    return (
        <div className={`flex h-full flex-col overflow-hidden ${surface}`}>
            {/* Header */}
            <div className={`flex items-center gap-2 border-b px-3 py-2 ${border}`}>
                <FlaskConical className="h-4 w-4 text-blue-500" aria-hidden />
                <span className="text-sm font-semibold">NEXUS</span>
                <span className={`text-xs ${muted}`}>
                    {serviceVersion ? `service v${serviceVersion}` : 'service offline'}
                </span>
                <span className={`ml-auto text-xs ${muted}`}>
                    {structures.length} of {selected.length} selected point
                    {selected.length === 1 ? '' : 's'} have structures
                </span>
            </div>

            {/* Mode switch */}
            <div className={`flex gap-1 border-b px-3 py-2 ${border}`}>
                {([['analyze', 'Analyze', Beaker],
                   ['screen', 'Next batch', Target],
                   ['predict', 'Predict', CheckCircle2]] as const).map(([key, label, Icon]) => (
                    <button
                        key={key}
                        type="button"
                        onClick={() => setMode(key)}
                        className={`flex items-center gap-1 rounded px-2 py-1 text-xs transition-colors ${
                            mode === key
                                ? 'bg-blue-500 text-white'
                                : isDarkMode
                                    ? 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                    >
                        <Icon className="h-3 w-3" aria-hidden />
                        {label}
                    </button>
                ))}
            </div>

            {/* Controls */}
            <div className={`space-y-2 border-b px-3 py-2 ${border}`}>
                {mode === 'screen' && (
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                        <label className={muted} htmlFor="nexus-strategy">Strategy</label>
                        <select
                            id="nexus-strategy"
                            value={strategy}
                            onChange={e => setStrategy(e.target.value as NexusStrategy)}
                            className={`rounded border px-1 py-0.5 ${border} ${subtle}`}
                        >
                            {NEXUS_STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                        <label className={muted} htmlFor="nexus-batch">Batch</label>
                        <input
                            id="nexus-batch"
                            type="number"
                            min={1}
                            max={500}
                            value={batchSize}
                            onChange={e => setBatchSize(Math.max(1, Number(e.target.value) || 1))}
                            className={`w-16 rounded border px-1 py-0.5 ${border} ${subtle}`}
                        />
                    </div>
                )}
                {(mode === 'screen' || mode === 'predict') && (
                    <div className="flex flex-wrap items-center gap-2 text-xs">
                        <label className={muted} htmlFor="nexus-label">
                            {mode === 'predict' ? 'Measured column' : 'Known results'}
                        </label>
                        <select
                            id="nexus-label"
                            value={labelColumn}
                            onChange={e => setLabelColumn(e.target.value)}
                            className={`max-w-[12rem] rounded border px-1 py-0.5 ${border} ${subtle}`}
                        >
                            <option value="">{mode === 'predict' ? 'select a column' : 'none yet'}</option>
                            {numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                    </div>
                )}
                <button
                    type="button"
                    onClick={run}
                    disabled={disabled}
                    className={`flex w-full items-center justify-center gap-2 rounded px-3 py-1.5 text-xs font-medium transition-colors ${
                        disabled
                            ? 'cursor-not-allowed bg-gray-400 text-white opacity-60'
                            : 'bg-blue-500 text-white hover:bg-blue-600'}`}
                >
                    {phase === 'running'
                        ? <><Loader2 className="h-3 w-3 animate-spin" aria-hidden />Working…</>
                        : <><RefreshCw className="h-3 w-3" aria-hidden />
                            {mode === 'analyze' ? `Analyze ${structures.length || ''} molecule${structures.length === 1 ? '' : 's'}`
                                : mode === 'screen' ? `Propose ${batchSize} to screen`
                                    : 'Predict unmeasured'}</>}
                </button>
                {structures.length === 0 && (
                    <p className={`text-xs ${muted}`}>
                        Select points whose metadata carries a SMILES string
                        ({SMILES_KEYS.slice(0, 3).join(', ')}…).
                    </p>
                )}
            </div>

            {/* Results */}
            <div className="flex-1 overflow-auto px-3 py-2 text-xs">
                {phase === 'error' && error && (
                    <div className="flex items-start gap-2 rounded bg-red-500/10 p-2 text-red-500">
                        <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
                        <div>
                            <div>{error}</div>
                            <div className={muted}>Service URL: {nexusBaseUrl()}</div>
                        </div>
                    </div>
                )}

                {mode === 'analyze' && analysis && (
                    <div className="space-y-3">
                        <div className="flex items-center gap-2">
                            <Palette className="h-3 w-3 text-blue-500" aria-hidden />
                            <label className={muted} htmlFor="nexus-colour">Colour map by</label>
                            <select
                                id="nexus-colour"
                                value={colourBy}
                                onChange={e => onColourChange(e.target.value)}
                                className={`rounded border px-1 py-0.5 ${border} ${subtle}`}
                            >
                                {NEXUS_COLOUR_PROPERTIES.map(p => (
                                    <option key={p.key} value={p.key}>{p.label}</option>
                                ))}
                            </select>
                            <button type="button" onClick={clearColours}
                                className={`ml-auto underline ${muted}`}>
                                clear colours
                            </button>
                        </div>
                        <div className="flex items-center gap-1" aria-hidden>
                            {RAMP.map(c => (
                                <span key={c} className="h-2 flex-1 rounded-sm"
                                    style={{ backgroundColor: c }} />
                            ))}
                        </div>

                        <div className={`grid grid-cols-2 gap-2 rounded p-2 ${subtle}`}>
                            {['mw', 'logp', 'tpsa', 'drug_likeness'].map(key => (
                                <div key={key}>
                                    <div className={muted}>{key}</div>
                                    <div className="font-mono">
                                        {fmt(analysis.summary[key]?.mean)}
                                        <span className={`ml-1 ${muted}`}>
                                            ± {fmt(analysis.summary[key]?.std)}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>

                        <div className={muted}>
                            {analysis.n_analyzed} analyzed in {analysis.elapsed_seconds}s
                            {analysis.n_failed > 0 && `, ${analysis.n_failed} unparseable`}
                            {scaffolds && ` · ${scaffolds.n_scaffolds} scaffolds`}
                        </div>

                        <table className="w-full table-fixed">
                            <thead className={muted}>
                                <tr className="text-left">
                                    <th className="w-[44%] font-normal">molecule</th>
                                    <th className="font-normal">MW</th>
                                    <th className="font-normal">logP</th>
                                    <th className="font-normal">TPSA</th>
                                    <th className="font-normal">Ro5</th>
                                </tr>
                            </thead>
                            <tbody className="font-mono">
                                {analysis.molecules.slice(0, 200).map(m => (
                                    <tr key={m.smiles} className={`border-t ${border}`}>
                                        <td className="truncate pr-1" title={m.smiles}>{m.formula}</td>
                                        <td>{fmt(m.descriptors.mw, 1)}</td>
                                        <td>{fmt(m.descriptors.logp)}</td>
                                        <td>{fmt(m.descriptors.tpsa, 1)}</td>
                                        <td className={m.lipinski_violations > 1 ? 'text-amber-500' : ''}>
                                            {m.lipinski_violations}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {mode === 'screen' && batch && (
                    <div className="space-y-2">
                        <div className={muted}>
                            {batch.rationale} · {batch.n_measured} already measured
                        </div>
                        <ol className="space-y-1">
                            {batch.batch.map((item, rank) => (
                                <li key={item.smiles}
                                    className={`flex items-center gap-2 border-t pt-1 ${border}`}>
                                    <span className={`w-5 shrink-0 text-right ${muted}`}>{rank + 1}</span>
                                    <span className="truncate font-mono" title={item.smiles}>
                                        {item.smiles}
                                    </span>
                                    <span className={`ml-auto shrink-0 font-mono ${muted}`}>
                                        {item.score.toFixed(3)}
                                    </span>
                                </li>
                            ))}
                        </ol>
                    </div>
                )}

                {mode === 'predict' && prediction && (
                    <div className="space-y-2">
                        <div className={`rounded p-2 ${subtle}`}>
                            <div>
                                {prediction.task} · trained on {prediction.n_labelled} labelled
                                molecules, {prediction.n_calibration} held out to calibrate
                            </div>
                            <div className={muted}>
                                Intervals carry a {(prediction.nominal_coverage * 100).toFixed(0)}%
                                coverage guarantee; measured {(prediction.calibration_coverage * 100).toFixed(1)}%
                                on the calibration set.
                            </div>
                        </div>
                        <table className="w-full table-fixed">
                            <thead className={muted}>
                                <tr className="text-left">
                                    <th className="w-[52%] font-normal">molecule</th>
                                    <th className="font-normal">predicted</th>
                                    <th className="font-normal">interval</th>
                                </tr>
                            </thead>
                            <tbody className="font-mono">
                                {prediction.predictions.slice(0, 200).map(p => (
                                    <tr key={p.smiles} className={`border-t ${border}`}>
                                        <td className="truncate pr-1" title={p.smiles}>{p.smiles}</td>
                                        <td>{fmt(p.prediction)}</td>
                                        <td className={muted}>
                                            {p.interval_low === undefined
                                                ? (p.ambiguous ? 'ambiguous' : 'confident')
                                                : `[${fmt(p.interval_low)}, ${fmt(p.interval_high)}]`}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}

                {phase === 'idle' && !analysis && !batch && !prediction && (
                    <p className={muted}>
                        NEXUS analyzes the molecules behind your selected points: descriptors and
                        drug-likeness, scaffold families, which compounds to screen next, and
                        property predictions with calibrated uncertainty.
                        {!mapId && ' Open a map to begin.'}
                    </p>
                )}
            </div>
        </div>
    );
};

export default NexusPanel;
