/**
 * Chart primitives for the NEXUS panel.
 *
 * Plain inline SVG, no plotting dependency, so the panel adds nothing to the
 * Mantis bundle beyond its own source. Every chart carries a hover layer, uses
 * recessive grid and axis ink, and keeps values and labels in text tokens --
 * colour identifies a mark, never the text beside it.
 */

import React, { useCallback, useId, useMemo, useState } from 'react';

import { rampColor, type NexusTheme } from './theme';

/** Rounded data-end radius, in px, for bars anchored to a baseline. */
const END_RADIUS = 4;
/** Surface-coloured gap between adjacent fills, in px. */
const BAR_GAP = 2;

interface Hover {
    x: number;
    y: number;
    lines: string[];
}

/**
 * Floating tooltip positioned inside the chart box.
 *
 * @param hover Active hover payload, or null.
 * @param width Chart width, used to keep the tooltip in frame.
 * @param theme Active theme.
 * @returns The tooltip, or null when nothing is hovered.
 */
const Tooltip: React.FC<{ hover: Hover | null; width: number; theme: NexusTheme }> =
({ hover, width, theme }) => {
    if (!hover) return null;
    const estimated = Math.max(...hover.lines.map(l => l.length)) * 5.6 + 16;
    const left = Math.min(Math.max(hover.x - estimated / 2, 0), Math.max(width - estimated, 0));
    return (
        <div
            className="pointer-events-none absolute z-10 rounded-md px-2 py-1 text-[10px] leading-snug shadow-lg"
            style={{
                left, top: Math.max(hover.y - 8, 0), minWidth: estimated,
                background: theme.tooltipBg, color: theme.text,
                border: `1px solid ${theme.lineStrong}`, backdropFilter: 'blur(6px)',
                transform: 'translateY(-100%)',
            }}
            role="tooltip"
        >
            {hover.lines.map((line, i) => (
                <div key={line + i} style={{ color: i === 0 ? theme.text : theme.text2 }}>
                    {line}
                </div>
            ))}
        </div>
    );
};

/**
 * Caption shown above a chart: a name and an optional right-aligned note.
 */
const ChartHead: React.FC<{ title: string; note?: string; theme: NexusTheme }> =
({ title, note, theme }) => (
    <div className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[10px] font-medium uppercase tracking-[0.14em]"
            style={{ color: theme.text2 }}>{title}</span>
        {note && <span className="text-[10px] tabular-nums" style={{ color: theme.text3 }}>{note}</span>}
    </div>
);

/**
 * A single headline number with its label.
 *
 * Some questions are answered by one number, and a chart around it would only
 * add furniture.
 *
 * @param label What the number measures.
 * @param value The number, already formatted.
 * @param unit Optional trailing unit.
 * @param hint Optional secondary line.
 * @param tone Optional status tone for the value.
 * @param theme Active theme.
 */
export const StatTile: React.FC<{
    label: string; value: string; unit?: string; hint?: string;
    tone?: 'default' | 'good' | 'warn' | 'crit'; theme: NexusTheme;
}> = ({ label, value, unit, hint, tone = 'default', theme }) => {
    const colour = tone === 'good' ? theme.good
        : tone === 'warn' ? theme.warn
            : tone === 'crit' ? theme.crit : theme.text;
    return (
        <div className="rounded-lg px-2.5 py-2"
            style={{ background: theme.surfaceInset, border: `1px solid ${theme.line}` }}>
            <div className="text-[9.5px] uppercase tracking-[0.14em]" style={{ color: theme.text3 }}>
                {label}
            </div>
            <div className="mt-0.5 flex items-baseline gap-1">
                <span className="font-mono text-[15px] font-semibold tabular-nums leading-none"
                    style={{ color: colour }}>{value}</span>
                {unit && <span className="text-[10px]" style={{ color: theme.text3 }}>{unit}</span>}
            </div>
            {hint && <div className="mt-1 text-[9.5px] leading-tight" style={{ color: theme.text3 }}>{hint}</div>}
        </div>
    );
};

/**
 * Distribution of one descriptor as a histogram.
 *
 * @param values Raw values; non-finite entries are ignored.
 * @param title Chart caption.
 * @param unit Unit appended to hover readouts.
 * @param digits Decimals in hover readouts.
 * @param height Plot height in px.
 * @param theme Active theme.
 */
export const Histogram: React.FC<{
    values: number[]; title: string; unit?: string; digits?: number;
    height?: number; theme: NexusTheme;
}> = ({ values, title, unit = '', digits = 1, height = 64, theme }) => {
    const [hover, setHover] = useState<Hover | null>(null);
    const width = 300;
    const clean = useMemo(() => values.filter(Number.isFinite), [values]);

    const bins = useMemo(() => {
        if (clean.length === 0) return null;
        const min = Math.min(...clean);
        const max = Math.max(...clean);
        if (!(max > min)) return { edges: [min, max], counts: [clean.length], min, max };
        // Sturges, capped: enough shape to read, never so many that bars vanish.
        const n = Math.min(18, Math.max(6, Math.ceil(Math.log2(clean.length) + 1)));
        const counts = new Array<number>(n).fill(0);
        for (const v of clean) {
            const idx = Math.min(n - 1, Math.floor(((v - min) / (max - min)) * n));
            counts[idx] += 1;
        }
        const edges = Array.from({ length: n + 1 }, (_, i) => min + ((max - min) * i) / n);
        return { edges, counts, min, max };
    }, [clean]);

    if (!bins) {
        return (
            <div>
                <ChartHead title={title} theme={theme} />
                <div className="rounded-md px-2 py-3 text-[10px]"
                    style={{ background: theme.surfaceInset, color: theme.text3 }}>
                    no values
                </div>
            </div>
        );
    }

    const peak = Math.max(...bins.counts, 1);
    const barWidth = width / bins.counts.length;
    const mean = clean.reduce((a, b) => a + b, 0) / clean.length;
    const meanX = bins.max > bins.min
        ? ((mean - bins.min) / (bins.max - bins.min)) * width : width / 2;

    return (
        <div className="relative">
            <ChartHead title={title}
                note={`n=${clean.length} · ${bins.min.toFixed(digits)}–${bins.max.toFixed(digits)}${unit}`}
                theme={theme} />
            <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
                role="img" aria-label={`${title} distribution, ${clean.length} molecules`}
                onMouseLeave={() => setHover(null)}>
                {[0.5, 1].map(f => (
                    <line key={f} x1={0} x2={width} y1={height - height * f} y2={height - height * f}
                        stroke={theme.grid} strokeWidth={1} />
                ))}
                {bins.counts.map((count, i) => {
                    const h = (count / peak) * (height - 6);
                    const x = i * barWidth;
                    const w = Math.max(barWidth - BAR_GAP, 1);
                    return (
                        <g key={i}>
                            <rect
                                x={x} y={height - h} width={w} height={h}
                                rx={Math.min(END_RADIUS, w / 2)}
                                fill={theme.series[0]}
                            />
                            {/* Hit target spans the full column height, not just the bar. */}
                            <rect
                                x={x} y={0} width={barWidth} height={height} fill="transparent"
                                onMouseEnter={() => setHover({
                                    x: x + barWidth / 2, y: height - h, lines: [
                                        `${count} molecule${count === 1 ? '' : 's'}`,
                                        `${bins.edges[i].toFixed(digits)} – ${bins.edges[i + 1].toFixed(digits)}${unit}`,
                                    ],
                                })}
                            />
                        </g>
                    );
                })}
                <line x1={meanX} x2={meanX} y1={0} y2={height}
                    stroke={theme.text3} strokeWidth={1} strokeDasharray="2 2" />
            </svg>
            <div className="mt-0.5 flex justify-between text-[9px] tabular-nums"
                style={{ color: theme.text3 }}>
                <span>{bins.min.toFixed(digits)}</span>
                <span>mean {mean.toFixed(digits)}{unit}</span>
                <span>{bins.max.toFixed(digits)}</span>
            </div>
            <Tooltip hover={hover} width={width} theme={theme} />
        </div>
    );
};

export interface ScatterDatum {
    x: number;
    y: number;
    label: string;
    colourValue?: number;
}

/**
 * Two descriptors against each other, points shaded by a third.
 *
 * @param data Points to draw.
 * @param xLabel Axis caption.
 * @param yLabel Axis caption.
 * @param colourLabel What the shading encodes.
 * @param title Chart caption.
 * @param theme Active theme.
 */
export const ScatterPlot: React.FC<{
    data: ScatterDatum[]; xLabel: string; yLabel: string; colourLabel?: string;
    title: string; height?: number; theme: NexusTheme;
}> = ({ data, xLabel, yLabel, colourLabel, title, height = 150, theme }) => {
    const [hover, setHover] = useState<Hover | null>(null);
    const width = 300;
    const pad = { l: 26, r: 6, t: 6, b: 16 };

    const clean = useMemo(
        () => data.filter(d => Number.isFinite(d.x) && Number.isFinite(d.y)), [data]);

    const scales = useMemo(() => {
        if (clean.length === 0) return null;
        const xs = clean.map(d => d.x);
        const ys = clean.map(d => d.y);
        const cs = clean.map(d => d.colourValue ?? 0).filter(Number.isFinite);
        const span = (lo: number, hi: number) => (hi > lo ? { lo, hi } : { lo: lo - 1, hi: hi + 1 });
        return {
            x: span(Math.min(...xs), Math.max(...xs)),
            y: span(Math.min(...ys), Math.max(...ys)),
            c: span(Math.min(...cs), Math.max(...cs)),
        };
    }, [clean]);

    if (!scales) {
        return (
            <div>
                <ChartHead title={title} theme={theme} />
                <div className="rounded-md px-2 py-6 text-center text-[10px]"
                    style={{ background: theme.surfaceInset, color: theme.text3 }}>
                    no points to plot
                </div>
            </div>
        );
    }

    const px = (v: number) => pad.l + ((v - scales.x.lo) / (scales.x.hi - scales.x.lo)) * (width - pad.l - pad.r);
    const py = (v: number) => height - pad.b - ((v - scales.y.lo) / (scales.y.hi - scales.y.lo)) * (height - pad.t - pad.b);

    return (
        <div className="relative">
            <ChartHead title={title}
                note={`${yLabel} vs ${xLabel}${colourLabel ? ` · shaded by ${colourLabel}` : ''}`}
                theme={theme} />
            <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
                role="img" aria-label={`${yLabel} against ${xLabel}, ${clean.length} molecules`}
                onMouseLeave={() => setHover(null)}>
                <rect x={pad.l} y={pad.t} width={width - pad.l - pad.r}
                    height={height - pad.t - pad.b} fill={theme.surfaceInset} rx={6} />
                {[0.25, 0.5, 0.75].map(f => (
                    <g key={f}>
                        <line x1={pad.l} x2={width - pad.r}
                            y1={pad.t + (height - pad.t - pad.b) * f}
                            y2={pad.t + (height - pad.t - pad.b) * f}
                            stroke={theme.grid} strokeWidth={1} />
                        <line y1={pad.t} y2={height - pad.b}
                            x1={pad.l + (width - pad.l - pad.r) * f}
                            x2={pad.l + (width - pad.l - pad.r) * f}
                            stroke={theme.grid} strokeWidth={1} />
                    </g>
                ))}
                {clean.map((d, i) => (
                    <circle
                        key={`${d.label}-${i}`} cx={px(d.x)} cy={py(d.y)} r={4}
                        fill={d.colourValue === undefined
                            ? theme.series[0]
                            : rampColor(d.colourValue, scales.c.lo, scales.c.hi, theme.ramp)}
                        fillOpacity={0.85}
                        // 2px surface ring keeps overlapping marks countable.
                        stroke={theme.surface} strokeWidth={1.5}
                        onMouseEnter={() => setHover({
                            x: px(d.x), y: py(d.y), lines: [
                                d.label,
                                `${xLabel} ${d.x.toFixed(2)} · ${yLabel} ${d.y.toFixed(2)}`,
                                ...(d.colourValue !== undefined && colourLabel
                                    ? [`${colourLabel} ${d.colourValue.toFixed(2)}`] : []),
                            ],
                        })}
                    />
                ))}
                <text x={pad.l} y={height - 4} fontSize={9} fill={theme.text3}>
                    {scales.x.lo.toFixed(1)}
                </text>
                <text x={width - pad.r} y={height - 4} fontSize={9} fill={theme.text3}
                    textAnchor="end">
                    {xLabel} {scales.x.hi.toFixed(1)}
                </text>
                <text x={2} y={pad.t - 1} fontSize={9} fill={theme.text3}>{yLabel}</text>
                <text x={2} y={pad.t + 8} fontSize={9} fill={theme.text3}>
                    {scales.y.hi.toFixed(1)}
                </text>
                <text x={2} y={height - pad.b} fontSize={9} fill={theme.text3}>
                    {scales.y.lo.toFixed(1)}
                </text>
            </svg>
            <Tooltip hover={hover} width={width} theme={theme} />
        </div>
    );
};

export interface RankedItem {
    label: string;
    value: number;
    sublabel?: string;
}

/**
 * Horizontal bars for a ranked list -- scaffold families, acquisition scores.
 *
 * Magnitude read against a common baseline, which is what a bar does best.
 *
 * @param items Already-ranked items.
 * @param title Chart caption.
 * @param valueFormat Formats the value for the direct label.
 * @param max Optional fixed upper bound; defaults to the largest value.
 * @param theme Active theme.
 */
export const RankedBars: React.FC<{
    items: RankedItem[]; title: string; note?: string;
    valueFormat?: (v: number) => string; max?: number; theme: NexusTheme;
}> = ({ items, title, note, valueFormat = v => v.toFixed(2), max, theme }) => {
    const [hover, setHover] = useState<Hover | null>(null);
    const width = 300;
    const rowHeight = 18;
    const height = Math.max(items.length * rowHeight, rowHeight);
    const peak = max ?? Math.max(...items.map(i => i.value), 1e-9);
    const labelWidth = 104;

    if (items.length === 0) return null;

    return (
        <div className="relative">
            <ChartHead title={title} note={note} theme={theme} />
            <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
                role="img" aria-label={title} onMouseLeave={() => setHover(null)}>
                {items.map((item, i) => {
                    const y = i * rowHeight;
                    const barH = rowHeight - BAR_GAP * 2;
                    const full = width - labelWidth - 38;
                    const w = Math.max((item.value / peak) * full, 2);
                    return (
                        <g key={`${item.label}-${i}`}>
                            <text x={0} y={y + rowHeight / 2 + 3} fontSize={10} fill={theme.text2}>
                                {item.label.length > 17 ? `${item.label.slice(0, 16)}…` : item.label}
                            </text>
                            <rect x={labelWidth} y={y + BAR_GAP} width={full} height={barH}
                                rx={END_RADIUS} fill={theme.grid} />
                            <rect x={labelWidth} y={y + BAR_GAP} width={w} height={barH}
                                rx={Math.min(END_RADIUS, w / 2)}
                                fill={rampColor(item.value, 0, peak, theme.ramp)} />
                            <text x={width} y={y + rowHeight / 2 + 3} fontSize={10}
                                textAnchor="end" fill={theme.text2}
                                className="tabular-nums">
                                {valueFormat(item.value)}
                            </text>
                            <rect x={0} y={y} width={width} height={rowHeight} fill="transparent"
                                onMouseEnter={() => setHover({
                                    x: labelWidth + w, y: y, lines: [
                                        item.label,
                                        ...(item.sublabel ? [item.sublabel] : []),
                                        valueFormat(item.value),
                                    ],
                                })} />
                        </g>
                    );
                })}
            </svg>
            <Tooltip hover={hover} width={width} theme={theme} />
        </div>
    );
};

export interface IntervalDatum {
    label: string;
    prediction: number;
    low?: number;
    high?: number;
}

/**
 * Predictions with their conformal intervals, sorted by predicted value.
 *
 * The point is the interval *width*: a guarantee you cannot see the size of is
 * not much of a guarantee.
 *
 * @param data Predictions to draw.
 * @param title Chart caption.
 * @param note Right-aligned caption note.
 * @param theme Active theme.
 */
export const IntervalStrip: React.FC<{
    data: IntervalDatum[]; title: string; note?: string; theme: NexusTheme;
}> = ({ data, title, note, theme }) => {
    const [hover, setHover] = useState<Hover | null>(null);
    const width = 300;
    const pad = { l: 4, r: 4, t: 6, b: 14 };

    const sorted = useMemo(
        () => data.filter(d => Number.isFinite(d.prediction))
            .slice()
            .sort((a, b) => a.prediction - b.prediction),
        [data]);

    const bounds = useMemo(() => {
        if (sorted.length === 0) return null;
        const all: number[] = [];
        for (const d of sorted) {
            all.push(d.prediction);
            if (d.low !== undefined && Number.isFinite(d.low)) all.push(d.low);
            if (d.high !== undefined && Number.isFinite(d.high)) all.push(d.high);
        }
        const lo = Math.min(...all);
        const hi = Math.max(...all);
        return hi > lo ? { lo, hi } : { lo: lo - 1, hi: hi + 1 };
    }, [sorted]);

    if (!bounds) return null;

    const rowHeight = Math.max(3, Math.min(9, 190 / sorted.length));
    const height = sorted.length * rowHeight + pad.t + pad.b;
    const sx = (v: number) => pad.l + ((v - bounds.lo) / (bounds.hi - bounds.lo)) * (width - pad.l - pad.r);

    return (
        <div className="relative">
            <ChartHead title={title} note={note} theme={theme} />
            <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
                role="img"
                aria-label={`${sorted.length} predictions with calibrated intervals`}
                onMouseLeave={() => setHover(null)}>
                {sorted.map((d, i) => {
                    const y = pad.t + i * rowHeight + rowHeight / 2;
                    const hasInterval = d.low !== undefined && d.high !== undefined
                        && Number.isFinite(d.low) && Number.isFinite(d.high);
                    return (
                        <g key={`${d.label}-${i}`}>
                            {hasInterval && (
                                <line x1={sx(d.low as number)} x2={sx(d.high as number)}
                                    y1={y} y2={y} stroke={theme.series[0]} strokeOpacity={0.45}
                                    strokeWidth={Math.max(2, rowHeight - 3)} strokeLinecap="round" />
                            )}
                            <circle cx={sx(d.prediction)} cy={y} r={Math.max(1.6, rowHeight / 3.4)}
                                fill={theme.text} />
                            <rect x={0} y={pad.t + i * rowHeight} width={width} height={rowHeight}
                                fill="transparent"
                                onMouseEnter={() => setHover({
                                    x: sx(d.prediction), y: pad.t + i * rowHeight, lines: [
                                        d.label.length > 34 ? `${d.label.slice(0, 33)}…` : d.label,
                                        `predicted ${d.prediction.toFixed(2)}`,
                                        hasInterval
                                            ? `interval [${(d.low as number).toFixed(2)}, ${(d.high as number).toFixed(2)}] · width ${((d.high as number) - (d.low as number)).toFixed(2)}`
                                            : 'no interval',
                                    ],
                                })} />
                        </g>
                    );
                })}
                <line x1={pad.l} x2={width - pad.r} y1={height - pad.b + 3}
                    y2={height - pad.b + 3} stroke={theme.line} strokeWidth={1} />
                <text x={pad.l} y={height - 2} fontSize={9} fill={theme.text3}>
                    {bounds.lo.toFixed(2)}
                </text>
                <text x={width - pad.r} y={height - 2} fontSize={9} fill={theme.text3}
                    textAnchor="end">{bounds.hi.toFixed(2)}</text>
            </svg>
            <Tooltip hover={hover} width={width} theme={theme} />
        </div>
    );
};

/**
 * Measured coverage against its nominal guarantee, as a bullet bar.
 *
 * @param nominal Guaranteed coverage in [0, 1].
 * @param measured Observed coverage in [0, 1].
 * @param theme Active theme.
 */
export const CoverageBar: React.FC<{
    nominal: number; measured: number; theme: NexusTheme;
}> = ({ nominal, measured, theme }) => {
    const width = 300;
    const height = 26;
    const barY = 7;
    const barH = 9;
    const holds = measured + 1e-9 >= nominal;
    return (
        <div>
            <ChartHead title="Interval coverage"
                note={`${(measured * 100).toFixed(1)}% measured vs ${(nominal * 100).toFixed(0)}% guaranteed`}
                theme={theme} />
            <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height}
                role="img"
                aria-label={`measured coverage ${(measured * 100).toFixed(1)} percent against a ${(nominal * 100).toFixed(0)} percent guarantee`}>
                <rect x={0} y={barY} width={width} height={barH} rx={END_RADIUS}
                    fill={theme.grid} />
                <rect x={0} y={barY} width={Math.max(measured * width, 2)} height={barH}
                    rx={END_RADIUS} fill={holds ? theme.good : theme.crit} />
                <line x1={nominal * width} x2={nominal * width} y1={barY - 4}
                    y2={barY + barH + 4} stroke={theme.text} strokeWidth={2} />
                <text x={Math.min(nominal * width + 5, width - 60)} y={height - 1}
                    fontSize={9} fill={theme.text3}>
                    guarantee {(nominal * 100).toFixed(0)}%
                </text>
            </svg>
        </div>
    );
};

/**
 * Legend for the sequential ramp, with the range it spans.
 *
 * @param label What the ramp encodes.
 * @param min Low end of the scale.
 * @param max High end of the scale.
 * @param theme Active theme.
 */
export const RampLegend: React.FC<{
    label: string; min: number; max: number; theme: NexusTheme;
}> = ({ label, min, max, theme }) => {
    const id = useId();
    return (
        <div>
            <div className="mb-1 text-[9.5px] uppercase tracking-[0.14em]" style={{ color: theme.text3 }}>
                {label}
            </div>
            <svg viewBox="0 0 300 8" width="100%" height={8} role="img"
                aria-label={`${label} scale from ${min} to ${max}`}>
                <defs>
                    <linearGradient id={`ramp-${id}`} x1="0" x2="1">
                        {theme.ramp.map((c, i) => (
                            <stop key={c} offset={`${(i / (theme.ramp.length - 1)) * 100}%`}
                                stopColor={c} />
                        ))}
                    </linearGradient>
                </defs>
                <rect x={0} y={0} width={300} height={8} rx={4} fill={`url(#ramp-${id})`} />
            </svg>
            <div className="mt-0.5 flex justify-between text-[9px] tabular-nums"
                style={{ color: theme.text3 }}>
                <span>{Number.isFinite(min) ? min.toFixed(2) : '--'}</span>
                <span>{Number.isFinite(max) ? max.toFixed(2) : '--'}</span>
            </div>
        </div>
    );
};

/**
 * Compact pass/fail dots for a rule set, e.g. Lipinski.
 *
 * Identity is never colour alone: each dot is labelled.
 *
 * @param items Rule name and whether it passed.
 * @param theme Active theme.
 */
export const RuleDots: React.FC<{
    items: { label: string; pass: boolean }[]; theme: NexusTheme;
}> = ({ items, theme }) => (
    <div className="flex flex-wrap gap-x-3 gap-y-1">
        {items.map(item => (
            <span key={item.label} className="flex items-center gap-1 text-[10px]"
                style={{ color: theme.text2 }}>
                <span className="inline-block h-1.5 w-1.5 rounded-full"
                    style={{ background: item.pass ? theme.good : theme.warn }} aria-hidden />
                {item.label}
                <span style={{ color: theme.text3 }}>{item.pass ? 'ok' : 'off'}</span>
            </span>
        ))}
    </div>
);

/**
 * Hook returning a stable callback that formats numbers for display.
 *
 * @returns A formatter tolerant of null and non-finite input.
 */
export function useNumberFormat(): (v: number | null | undefined, digits?: number) => string {
    return useCallback((v, digits = 2) => (
        v === null || v === undefined || !Number.isFinite(v) ? '--' : v.toFixed(digits)
    ), []);
}
