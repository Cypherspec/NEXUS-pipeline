import {
    AlertTriangle,
    Atom,
    Beaker,
    CheckCircle2,
    Layers,
    Loader2,
    RefreshCw,
    Sparkles,
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

import {
    CoverageBar,
    Histogram,
    IntervalStrip,
    RampLegend,
    RankedBars,
    RuleDots,
    ScatterPlot,
    StatTile,
    useNumberFormat,
    type ScatterDatum,
} from './nexus/charts';
import { gradientText, nexusTheme, rampColor } from './nexus/theme';

/** Layer this panel writes point colours into, so it only clears its own. */
const COLOR_OWNER = 'nexus';

/** Metadata keys searched, in order, for a structure string. */
const SMILES_KEYS = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'structure', 'mol'];

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

const NexusPanel: React.FC = () => {
    const { isDarkMode } = useDarkMode();
    const theme = nexusTheme(isDarkMode);
    const fmt = useNumberFormat();
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
    const [adaptive, setAdaptive] = useState(false);

    const colouredIds = useRef<Set<string>>(new Set());
    const inFlight = useRef<AbortController | null>(null);

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
            const colour = rampColor(value, min, max, theme.ramp);
            for (const id of ids) colours.set(id, colour);
        }
        usePointColorStore.getState().updateMultiplePointColors(colours, COLOR_OWNER);
        colouredIds.current = new Set(colours.keys());
    }, [clearColours, pointBySmiles, theme.ramp]);

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
                    { adaptive }, controller.signal));
            }
            setPhase('done');
        } catch (e) {
            if ((e as Error).name === 'AbortError') return;
            setPhase('error');
            setError((e as Error).message);
        }
    }, [adaptive, batchSize, colourBy, labelColumn, mode, paint, pointBySmiles,
        selected, strategy, structures]);

    const onColourChange = useCallback((property: string) => {
        setColourBy(property);
        paint(property, analysis);
    }, [analysis, paint]);

    // Derived views of the analysis, used by the charts.
    const chem = useMemo(() => {
        if (!analysis) return null;
        const molecules = analysis.molecules;
        const value = (key: string) => molecules
            .map(m => m.descriptors[key])
            .filter((v): v is number => v !== null && Number.isFinite(v));
        const colourValues = molecules.map(m => m.descriptors[colourBy]);
        const finite = colourValues.filter((v): v is number => v !== null && Number.isFinite(v));
        const scatter: ScatterDatum[] = molecules
            .filter(m => m.descriptors.mw !== null && m.descriptors.logp !== null)
            .map(m => ({
                x: m.descriptors.mw as number,
                y: m.descriptors.logp as number,
                label: m.formula,
                colourValue: m.descriptors[colourBy] ?? undefined,
            }));
        const ro5Pass = molecules.filter(m => m.lipinski_violations <= 1).length;
        const veberPass = molecules.filter(m => m.veber_pass).length;
        const alertFree = molecules.filter(m => m.structural_alerts === 0).length;
        return {
            logp: value('logp'),
            tpsa: value('tpsa'),
            mw: value('mw'),
            quality: value('drug_likeness'),
            scatter,
            colourMin: finite.length ? Math.min(...finite) : Number.NaN,
            colourMax: finite.length ? Math.max(...finite) : Number.NaN,
            ro5Rate: molecules.length ? ro5Pass / molecules.length : 0,
            veberRate: molecules.length ? veberPass / molecules.length : 0,
            alertRate: molecules.length ? alertFree / molecules.length : 0,
        };
    }, [analysis, colourBy]);

    const scaffoldBars = useMemo(() => {
        if (!scaffolds) return [];
        return scaffolds.groups
            .slice()
            .sort((a, b) => b.count - a.count)
            .slice(0, 8)
            .map(g => ({
                label: g.scaffold || 'acyclic',
                value: g.count,
                sublabel: `${g.count} molecule${g.count === 1 ? '' : 's'} share this core`,
            }));
    }, [scaffolds]);

    const disabled = phase === 'running' || structures.length === 0;
    const live = serviceVersion !== null;

    const selectStyle = {
        background: theme.surfaceInset,
        borderColor: theme.line,
        color: theme.text,
    };

    return (
        <div className="flex h-full flex-col overflow-hidden"
            style={{ background: theme.bg, color: theme.text }}>

            {/* Header: wordmark, what the service is, and whether it is reachable. */}
            <div className="shrink-0 px-3 pb-2 pt-2.5"
                style={{ borderBottom: `1px solid ${theme.line}`, background: theme.surface }}>
                <div className="flex items-center gap-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-md"
                        style={{ background: theme.surfaceInset, border: `1px solid ${theme.line}` }}>
                        <Atom className="h-3.5 w-3.5" style={{ color: theme.accent }} aria-hidden />
                    </div>
                    <span className="text-[15px] font-semibold tracking-tight"
                        style={gradientText(theme)}>NEXUS</span>
                    <span className="flex items-center gap-1 rounded-full px-1.5 py-0.5 text-[9px] uppercase tracking-[0.12em]"
                        style={{
                            background: theme.surfaceInset,
                            color: live ? theme.good : theme.text3,
                            border: `1px solid ${theme.line}`,
                        }}>
                        <span className="inline-block h-1 w-1 rounded-full"
                            style={{ background: live ? theme.good : theme.text3 }} aria-hidden />
                        {live ? `v${serviceVersion}` : 'offline'}
                    </span>
                    <span className="ml-auto text-[10px] tabular-nums" style={{ color: theme.text3 }}>
                        {structures.length}/{selected.length} with structures
                    </span>
                </div>
                <p className="mt-1 text-[10px] leading-snug" style={{ color: theme.text3 }}>
                    Non-equilibrium exploration with cross-modal unified synthesis
                </p>
            </div>

            {/* Mode switch, as a segmented control. */}
            <div className="shrink-0 px-3 py-2" style={{ background: theme.surface }}>
                <div className="flex gap-0.5 rounded-lg p-0.5"
                    style={{ background: theme.surfaceInset, border: `1px solid ${theme.line}` }}>
                    {([['analyze', 'Analyze', Beaker],
                       ['screen', 'Next batch', Target],
                       ['predict', 'Predict', CheckCircle2]] as const).map(([key, label, Icon]) => (
                        <button
                            key={key}
                            type="button"
                            onClick={() => setMode(key)}
                            aria-pressed={mode === key}
                            className="flex flex-1 items-center justify-center gap-1 rounded-md py-1 text-[11px] font-medium transition-colors"
                            style={mode === key
                                ? { background: theme.accent, color: '#ffffff' }
                                : { color: theme.text2 }}
                        >
                            <Icon className="h-3 w-3" aria-hidden />
                            {label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Controls */}
            <div className="shrink-0 space-y-2 px-3 pb-2.5"
                style={{ borderBottom: `1px solid ${theme.line}`, background: theme.surface }}>
                {mode === 'screen' && (
                    <div className="flex flex-wrap items-center gap-2 text-[10px]">
                        <label className="uppercase tracking-[0.12em]"
                            style={{ color: theme.text3 }} htmlFor="nexus-strategy">Strategy</label>
                        <select
                            id="nexus-strategy"
                            value={strategy}
                            onChange={e => setStrategy(e.target.value as NexusStrategy)}
                            className="rounded-md border px-1.5 py-1 text-[10px]"
                            style={selectStyle}
                        >
                            {NEXUS_STRATEGIES.map(s => <option key={s} value={s}>{s}</option>)}
                        </select>
                        <label className="uppercase tracking-[0.12em]"
                            style={{ color: theme.text3 }} htmlFor="nexus-batch">Batch</label>
                        <input
                            id="nexus-batch"
                            type="number"
                            min={1}
                            max={500}
                            value={batchSize}
                            onChange={e => setBatchSize(Math.max(1, Number(e.target.value) || 1))}
                            className="w-14 rounded-md border px-1.5 py-1 text-[10px] tabular-nums"
                            style={selectStyle}
                        />
                    </div>
                )}
                {(mode === 'screen' || mode === 'predict') && (
                    <div className="flex flex-wrap items-center gap-2 text-[10px]">
                        <label className="uppercase tracking-[0.12em]" style={{ color: theme.text3 }}
                            htmlFor="nexus-label">
                            {mode === 'predict' ? 'Measured column' : 'Known results'}
                        </label>
                        <select
                            id="nexus-label"
                            value={labelColumn}
                            onChange={e => setLabelColumn(e.target.value)}
                            className="max-w-[12rem] rounded-md border px-1.5 py-1 text-[10px]"
                            style={selectStyle}
                        >
                            <option value="">{mode === 'predict' ? 'select a column' : 'none yet'}</option>
                            {numericColumns.map(c => <option key={c} value={c}>{c}</option>)}
                        </select>
                        {mode === 'predict' && (
                            <label className="flex items-center gap-1.5"
                                style={{ color: theme.text3 }}>
                                <input type="checkbox" checked={adaptive}
                                    onChange={e => setAdaptive(e.target.checked)}
                                    style={{ accentColor: theme.accent }} />
                                <span className="uppercase tracking-[0.12em]">Adaptive width</span>
                            </label>
                        )}
                    </div>
                )}
                <button
                    type="button"
                    onClick={run}
                    disabled={disabled}
                    className="flex w-full items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-[11px] font-semibold transition-opacity"
                    style={disabled
                        ? { background: theme.surfaceInset, color: theme.text3,
                            border: `1px solid ${theme.line}`, cursor: 'not-allowed' }
                        : { backgroundImage: theme.gradient, color: '#06111f' }}
                >
                    {phase === 'running'
                        ? <><Loader2 className="h-3 w-3 animate-spin" aria-hidden />Working…</>
                        : <><RefreshCw className="h-3 w-3" aria-hidden />
                            {mode === 'analyze'
                                ? `Analyze ${structures.length || ''} molecule${structures.length === 1 ? '' : 's'}`
                                : mode === 'screen' ? `Propose ${batchSize} to screen`
                                    : 'Predict unmeasured'}</>}
                </button>
                {structures.length === 0 && (
                    <p className="text-[10px] leading-snug" style={{ color: theme.text3 }}>
                        Select points whose metadata carries a SMILES string
                        ({SMILES_KEYS.slice(0, 3).join(', ')}…).
                    </p>
                )}
            </div>

            {/* Results */}
            <div className="flex-1 overflow-auto px-3 py-2.5">
                {phase === 'error' && error && (
                    <div className="flex items-start gap-2 rounded-lg p-2 text-[10px]"
                        style={{ background: `${theme.crit}1a`, color: theme.crit,
                            border: `1px solid ${theme.crit}40` }}>
                        <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden />
                        <div>
                            <div>{error}</div>
                            <div style={{ color: theme.text3 }}>Service URL: {nexusBaseUrl()}</div>
                        </div>
                    </div>
                )}

                {mode === 'analyze' && analysis && chem && (
                    <div className="space-y-3.5">
                        <div className="grid grid-cols-2 gap-1.5">
                            <StatTile label="Molecules" value={String(analysis.n_analyzed)}
                                hint={analysis.n_failed > 0
                                    ? `${analysis.n_failed} unparseable`
                                    : `parsed in ${analysis.elapsed_seconds}s`}
                                theme={theme} />
                            <StatTile label="Scaffold families"
                                value={scaffolds ? String(scaffolds.n_scaffolds) : '--'}
                                hint={scaffolds && analysis.n_analyzed
                                    ? `${(scaffolds.n_scaffolds / analysis.n_analyzed).toFixed(2)} per molecule`
                                    : undefined}
                                theme={theme} />
                            <StatTile label="Mean drug-likeness"
                                value={fmt(analysis.summary.drug_likeness?.mean)}
                                hint={`± ${fmt(analysis.summary.drug_likeness?.std)} sd`}
                                theme={theme} />
                            <StatTile label="Ro5 pass"
                                value={`${(chem.ro5Rate * 100).toFixed(0)}%`}
                                tone={chem.ro5Rate >= 0.8 ? 'good' : chem.ro5Rate >= 0.5 ? 'warn' : 'crit'}
                                hint="at most one violation"
                                theme={theme} />
                        </div>

                        <RuleDots theme={theme} items={[
                            { label: 'Veber', pass: chem.veberRate >= 0.8 },
                            { label: 'Alert-free', pass: chem.alertRate >= 0.8 },
                            { label: 'Valence', pass: analysis.n_failed === 0 },
                        ]} />

                        <div className="rounded-lg p-2"
                            style={{ background: theme.surface, border: `1px solid ${theme.line}` }}>
                            <div className="mb-2 flex items-center gap-2">
                                <Sparkles className="h-3 w-3" style={{ color: theme.accent }} aria-hidden />
                                <label className="text-[10px] uppercase tracking-[0.12em]"
                                    style={{ color: theme.text3 }} htmlFor="nexus-colour">
                                    Colour map by
                                </label>
                                <select
                                    id="nexus-colour"
                                    value={colourBy}
                                    onChange={e => onColourChange(e.target.value)}
                                    className="rounded-md border px-1.5 py-0.5 text-[10px]"
                                    style={selectStyle}
                                >
                                    {NEXUS_COLOUR_PROPERTIES.map(p => (
                                        <option key={p.key} value={p.key}>{p.label}</option>
                                    ))}
                                </select>
                                <button type="button" onClick={clearColours}
                                    className="ml-auto text-[10px] underline"
                                    style={{ color: theme.text3 }}>
                                    clear
                                </button>
                            </div>
                            <RampLegend label={colourBy} min={chem.colourMin} max={chem.colourMax}
                                theme={theme} />
                        </div>

                        <ScatterPlot title="Property space" data={chem.scatter}
                            xLabel="MW" yLabel="logP" colourLabel={colourBy} theme={theme} />

                        <Histogram title="logP" values={chem.logp} digits={2} theme={theme} />
                        <Histogram title="Polar surface area" values={chem.tpsa}
                            unit=" Å²" digits={0} theme={theme} />

                        {scaffoldBars.length > 0 && (
                            <RankedBars title="Largest scaffold families" items={scaffoldBars}
                                note={scaffolds ? `${scaffolds.n_scaffolds} in total` : undefined}
                                valueFormat={v => String(v)} theme={theme} />
                        )}

                        <div>
                            <div className="mb-1 text-[10px] font-medium uppercase tracking-[0.14em]"
                                style={{ color: theme.text2 }}>Molecules</div>
                            <table className="w-full table-fixed text-[10px]">
                                <thead>
                                    <tr className="text-left" style={{ color: theme.text3 }}>
                                        <th className="w-[40%] font-normal">formula</th>
                                        <th className="font-normal">MW</th>
                                        <th className="font-normal">logP</th>
                                        <th className="font-normal">TPSA</th>
                                        <th className="font-normal">Ro5</th>
                                    </tr>
                                </thead>
                                <tbody className="font-mono tabular-nums">
                                    {analysis.molecules.slice(0, 200).map(m => (
                                        <tr key={m.smiles} style={{ borderTop: `1px solid ${theme.line}` }}>
                                            <td className="truncate pr-1" title={m.smiles}>{m.formula}</td>
                                            <td>{fmt(m.descriptors.mw, 1)}</td>
                                            <td>{fmt(m.descriptors.logp)}</td>
                                            <td>{fmt(m.descriptors.tpsa, 1)}</td>
                                            <td style={{ color: m.lipinski_violations > 1 ? theme.warn : theme.text2 }}>
                                                {m.lipinski_violations}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}

                {mode === 'screen' && batch && (
                    <div className="space-y-3.5">
                        <div className="grid grid-cols-2 gap-1.5">
                            <StatTile label="Proposed" value={String(batch.batch.length)}
                                hint={`by ${strategy}`} theme={theme} />
                            <StatTile label="Already measured" value={String(batch.n_measured)}
                                hint={batch.n_measured === 0
                                    ? 'cold start: diversity only' : 'informs the model'}
                                theme={theme} />
                        </div>
                        <p className="text-[10px] leading-snug" style={{ color: theme.text3 }}>
                            {batch.rationale}
                        </p>
                        <RankedBars
                            title="Acquisition score"
                            note="higher is more informative"
                            items={batch.batch.slice(0, 14).map((item, rank) => ({
                                label: `${rank + 1}. ${item.smiles}`,
                                value: item.score,
                                sublabel: item.smiles,
                            }))}
                            valueFormat={v => v.toFixed(3)}
                            theme={theme}
                        />
                        <div>
                            <div className="mb-1 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.14em]"
                                style={{ color: theme.text2 }}>
                                <Layers className="h-3 w-3" aria-hidden />Full batch
                            </div>
                            <ol className="space-y-0.5">
                                {batch.batch.map((item, rank) => (
                                    <li key={item.smiles}
                                        className="flex items-center gap-2 py-0.5 text-[10px]"
                                        style={{ borderTop: `1px solid ${theme.line}` }}>
                                        <span className="w-5 shrink-0 text-right tabular-nums"
                                            style={{ color: theme.text3 }}>{rank + 1}</span>
                                        <span className="truncate font-mono" title={item.smiles}
                                            style={{ color: theme.text2 }}>
                                            {item.smiles}
                                        </span>
                                        <span className="ml-auto shrink-0 font-mono tabular-nums"
                                            style={{ color: theme.text3 }}>
                                            {item.score.toFixed(3)}
                                        </span>
                                    </li>
                                ))}
                            </ol>
                        </div>
                    </div>
                )}

                {mode === 'predict' && prediction && (
                    <div className="space-y-3.5">
                        <div className="grid grid-cols-2 gap-1.5">
                            <StatTile label="Trained on" value={String(prediction.n_labelled)}
                                unit="labelled" hint={`${prediction.task} task`} theme={theme} />
                            <StatTile label="Calibrated on" value={String(prediction.n_calibration)}
                                unit="held out" hint="never seen in training" theme={theme} />
                        </div>
                        <CoverageBar nominal={prediction.nominal_coverage}
                            measured={prediction.calibration_coverage} theme={theme} />
                        <p className="text-[10px] leading-snug" style={{ color: theme.text3 }}>
                            Split-conformal intervals: the guarantee is finite-sample and
                            distribution-free, so coverage holds whatever the model gets wrong.
                            {prediction.interval_mode === 'adaptive'
                                ? ' Widths are scaled by the model\u2019s own uncertainty, so they vary per molecule.'
                                : ' Every interval has the same width; tick "adaptive width" to let it vary.'}
                            {prediction.mean_interval_width !== null
                                && ` Mean width ${prediction.mean_interval_width.toFixed(2)}.`}
                        </p>
                        <IntervalStrip
                            title="Predictions, sorted"
                            note={`${prediction.predictions.length} molecules`}
                            data={prediction.predictions.map(p => ({
                                label: p.smiles,
                                prediction: p.prediction,
                                low: p.interval_low,
                                high: p.interval_high,
                            }))}
                            theme={theme}
                        />
                        <table className="w-full table-fixed text-[10px]">
                            <thead>
                                <tr className="text-left" style={{ color: theme.text3 }}>
                                    <th className="w-[50%] font-normal">molecule</th>
                                    <th className="font-normal">predicted</th>
                                    <th className="font-normal">interval</th>
                                </tr>
                            </thead>
                            <tbody className="font-mono tabular-nums">
                                {prediction.predictions.slice(0, 200).map(p => (
                                    <tr key={p.smiles} style={{ borderTop: `1px solid ${theme.line}` }}>
                                        <td className="truncate pr-1" title={p.smiles}>{p.smiles}</td>
                                        <td>{fmt(p.prediction)}</td>
                                        <td style={{ color: theme.text3 }}>
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
                    <div className="space-y-2.5 pt-1">
                        <p className="text-[11px] leading-relaxed" style={{ color: theme.text2 }}>
                            NEXUS reads the molecules behind your selected points: descriptors and
                            drug-likeness, scaffold families, which compounds to screen next, and
                            property predictions with calibrated uncertainty.
                        </p>
                        <div className="space-y-1.5">
                            {[['Analyze', 'descriptors, scaffolds, and a map coloured by any of them'],
                              ['Next batch', 'the most informative compounds to measure next'],
                              ['Predict', 'values with intervals that carry a coverage guarantee'],
                            ].map(([name, what]) => (
                                <div key={name} className="rounded-lg px-2.5 py-2"
                                    style={{ background: theme.surface, border: `1px solid ${theme.line}` }}>
                                    <div className="text-[10px] font-semibold" style={{ color: theme.text }}>
                                        {name}
                                    </div>
                                    <div className="text-[10px] leading-snug" style={{ color: theme.text3 }}>
                                        {what}
                                    </div>
                                </div>
                            ))}
                        </div>
                        {!mapId && (
                            <p className="text-[10px]" style={{ color: theme.text3 }}>
                                Open a map to begin.
                            </p>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default NexusPanel;
