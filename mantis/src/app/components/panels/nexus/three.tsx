/**
 * Animated 3D views, drawn as SVG rather than with a WebGL library.
 *
 * A molecule at this size is tens of atoms and a chemical space a few hundred
 * points, which is well inside what SVG draws at 60fps with a painter's-
 * algorithm sort. That keeps the panel bundle free of a 600KB renderer, which
 * matters for an extension package capped at 25MB and shipped as UTF-8 text.
 *
 * Both views share one convention: depth is encoded twice, by size and by
 * opacity, because a single cue reads as "small" rather than "far" on a flat
 * background.
 */

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { rampColor, type NexusTheme } from './theme';

/** Degrees turned per second when spinning unattended. */
const SPIN_RATE = 14;

/**
 * Element colours.
 *
 * This is the CPK convention a chemist already reads without a legend --
 * oxygen red, nitrogen blue, sulfur yellow -- stepped for a dark surface
 * rather than used at its textbook values. It identifies an element, so it is
 * categorical by convention, not a palette choice; every element is also
 * labelled in the legend beneath the view.
 */
const ELEMENT_COLOURS: Record<string, string> = {
    C: '#9aa3b2', N: '#5b9dff', O: '#ff7a7a', S: '#ffd166', P: '#ff9f5b',
    F: '#7ae0c3', Cl: '#6fd49a', Br: '#c98b6b', I: '#b48ce0', H: '#dfe3ea',
};

/** Colour for an element symbol, falling back to a neutral. */
function elementColour(symbol: string): string {
    return ELEMENT_COLOURS[symbol] ?? '#8d93a3';
}

interface Rotation {
    yaw: number;
    pitch: number;
}

/**
 * Projects a unit-box point under yaw/pitch with a weak perspective.
 *
 * @param point Coordinates in [-1, 1].
 * @param rotation Current yaw and pitch, in radians.
 * @returns Screen x and y in [-1, 1] plus a depth in [0, 1], 1 being nearest.
 */
function project(point: { x: number; y: number; z: number }, rotation: Rotation) {
    const cosY = Math.cos(rotation.yaw);
    const sinY = Math.sin(rotation.yaw);
    const cosP = Math.cos(rotation.pitch);
    const sinP = Math.sin(rotation.pitch);

    const x1 = point.x * cosY + point.z * sinY;
    const z1 = -point.x * sinY + point.z * cosY;
    const y1 = point.y * cosP - z1 * sinP;
    const z2 = point.y * sinP + z1 * cosP;

    // Weak perspective: enough to read as depth, not enough to distort a
    // structure a chemist is trying to recognise.
    const scale = 1 / (1 + 0.28 * z2);
    return { x: x1 * scale, y: y1 * scale, depth: (1 - z2) / 2 };
}

/**
 * Hook driving the shared rotation: it spins on its own and follows a drag.
 *
 * @param active Whether the animation should run.
 * @returns The current rotation and the handlers that let a pointer steer it.
 */
function useRotation(active: boolean) {
    const [rotation, setRotation] = useState<Rotation>({ yaw: 0.6, pitch: -0.3 });
    const dragging = useRef<{ x: number; y: number } | null>(null);
    const [spinning, setSpinning] = useState(true);

    useEffect(() => {
        if (!active || !spinning) return undefined;
        let frame = 0;
        let previous = performance.now();
        const step = (now: number) => {
            const delta = (now - previous) / 1000;
            previous = now;
            setRotation(current => ({
                ...current,
                yaw: current.yaw + (SPIN_RATE * Math.PI / 180) * delta,
            }));
            frame = requestAnimationFrame(step);
        };
        frame = requestAnimationFrame(step);
        return () => cancelAnimationFrame(frame);
    }, [active, spinning]);

    const onPointerDown = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
        dragging.current = { x: event.clientX, y: event.clientY };
        setSpinning(false);
        event.currentTarget.setPointerCapture(event.pointerId);
    }, []);

    const onPointerMove = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
        const from = dragging.current;
        if (!from) return;
        const dx = event.clientX - from.x;
        const dy = event.clientY - from.y;
        dragging.current = { x: event.clientX, y: event.clientY };
        setRotation(current => ({
            yaw: current.yaw + dx * 0.012,
            // Clamp pitch so the structure cannot be rolled past vertical,
            // which reads as a glitch rather than as a rotation.
            pitch: Math.max(-1.3, Math.min(1.3, current.pitch + dy * 0.012)),
        }));
    }, []);

    const onPointerUp = useCallback((event: React.PointerEvent<SVGSVGElement>) => {
        dragging.current = null;
        event.currentTarget.releasePointerCapture(event.pointerId);
    }, []);

    return { rotation, spinning, setSpinning, onPointerDown, onPointerMove, onPointerUp };
}

export interface ConformerAtom {
    symbol: string;
    radius: number;
    aromatic: boolean;
    hydrogens: number;
    x: number;
    y: number;
    z: number;
}

export interface ConformerBond {
    a: number;
    b: number;
    order: number;
    inRing: boolean;
}

/**
 * A rotating ball-and-stick view of one conformer.
 *
 * @param atoms Atoms in a unit box, as the backend returns them.
 * @param bonds Bonds by atom index.
 * @param formula Molecular formula, shown as the caption.
 * @param restraintError Residual distance-restraint error from the embedding.
 * @param theme Active theme.
 */
export const Molecule3D: React.FC<{
    atoms: ConformerAtom[]; bonds: ConformerBond[]; formula: string;
    restraintError?: number; height?: number; theme: NexusTheme;
}> = ({ atoms, bonds, formula, restraintError, height = 190, theme }) => {
    const { rotation, spinning, setSpinning, onPointerDown, onPointerMove, onPointerUp } =
        useRotation(atoms.length > 0);
    const size = 300;

    const projected = useMemo(
        () => atoms.map(atom => ({ atom, ...project(atom, rotation) })), [atoms, rotation]);

    // Painter's algorithm: everything is drawn back to front, bonds with the
    // depth of their nearer end so a bond never floats over the atom it joins.
    const marks = useMemo(() => {
        const atomMarks = projected.map((p, i) => ({ kind: 'atom' as const, index: i, depth: p.depth }));
        const bondMarks = bonds.map((bond, i) => ({
            kind: 'bond' as const, index: i,
            depth: Math.min(projected[bond.a]?.depth ?? 0, projected[bond.b]?.depth ?? 0),
        }));
        return [...atomMarks, ...bondMarks].sort((a, b) => a.depth - b.depth);
    }, [bonds, projected]);

    const present = useMemo(() => {
        const seen = new Map<string, number>();
        for (const atom of atoms) seen.set(atom.symbol, (seen.get(atom.symbol) ?? 0) + 1);
        return [...seen.entries()].sort((a, b) => b[1] - a[1]);
    }, [atoms]);

    if (atoms.length === 0) return null;
    const toScreen = (value: number) => size / 2 + value * (size / 2) * 0.76;
    const toScreenY = (value: number) => height / 2 - value * (height / 2) * 0.76;

    return (
        <div className="nexus-three">
            <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="text-[10px] font-medium uppercase tracking-[0.14em]"
                    style={{ color: theme.text2 }}>Conformer</span>
                <span className="text-[10px] tabular-nums" style={{ color: theme.text3 }}>
                    {formula} · {atoms.length} atoms
                    {restraintError !== undefined && ` · restraint error ${restraintError.toFixed(3)}`}
                </span>
            </div>
            <svg
                viewBox={`0 0 ${size} ${height}`} width="100%" height={height}
                role="img"
                aria-label={`Rotating three-dimensional conformer of ${formula}, ${atoms.length} atoms`}
                style={{ cursor: 'grab', touchAction: 'none', borderRadius: 12,
                    background: theme.surfaceInset, border: `1px solid ${theme.line}` }}
                onPointerDown={onPointerDown} onPointerMove={onPointerMove}
                onPointerUp={onPointerUp} onPointerLeave={onPointerUp}
            >
                <defs>
                    <radialGradient id="nexus-atom-sheen" cx="35%" cy="32%">
                        <stop offset="0%" stopColor="#ffffff" stopOpacity="0.55" />
                        <stop offset="100%" stopColor="#ffffff" stopOpacity="0" />
                    </radialGradient>
                </defs>
                {marks.map(mark => {
                    if (mark.kind === 'bond') {
                        const bond = bonds[mark.index];
                        const from = projected[bond.a];
                        const to = projected[bond.b];
                        if (!from || !to) return null;
                        const depth = 0.4 + 0.6 * mark.depth;
                        const x1 = toScreen(from.x), y1 = toScreenY(from.y);
                        const x2 = toScreen(to.x), y2 = toScreenY(to.y);
                        // Offset the strands of a multiple bond perpendicular
                        // to it, which is how a structure diagram shows order.
                        const dx = x2 - x1, dy = y2 - y1;
                        const length = Math.hypot(dx, dy) || 1;
                        const nx = (-dy / length) * 2.1, ny = (dx / length) * 2.1;
                        const strands = bond.order === 2 ? [-1, 1]
                            : bond.order === 3 ? [-1, 0, 1] : [0];
                        return (
                            <g key={`b${mark.index}`} opacity={depth}>
                                {strands.map(offset => (
                                    <line key={offset}
                                        x1={x1 + nx * offset} y1={y1 + ny * offset}
                                        x2={x2 + nx * offset} y2={y2 + ny * offset}
                                        stroke={bond.order === 4 ? theme.accent : theme.text2}
                                        strokeOpacity={bond.order === 4 ? 0.75 : 0.5}
                                        strokeWidth={2.2 * (0.6 + 0.4 * mark.depth)}
                                        strokeLinecap="round"
                                        strokeDasharray={bond.order === 4 ? '5 3' : undefined} />
                                ))}
                            </g>
                        );
                    }
                    const { atom, x, y, depth } = projected[mark.index];
                    const r = (3.6 + atom.radius * 4.2) * (0.62 + 0.38 * depth);
                    return (
                        <g key={`a${mark.index}`} opacity={0.5 + 0.5 * depth}>
                            <circle cx={toScreen(x)} cy={toScreenY(y)} r={r}
                                fill={elementColour(atom.symbol)}
                                stroke={theme.surfaceInset} strokeWidth={1.1} />
                            <circle cx={toScreen(x)} cy={toScreenY(y)} r={r}
                                fill="url(#nexus-atom-sheen)" />
                        </g>
                    );
                })}
            </svg>
            <div className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1">
                {present.map(([symbol, count]) => (
                    <span key={symbol} className="flex items-center gap-1 text-[9.5px]"
                        style={{ color: theme.text3 }}>
                        <span className="inline-block h-2 w-2 rounded-full"
                            style={{ background: elementColour(symbol) }} aria-hidden />
                        {symbol}<span style={{ opacity: 0.7 }}>{count}</span>
                    </span>
                ))}
                <button type="button" className="ml-auto text-[9.5px] underline"
                    style={{ color: theme.text3 }}
                    onClick={() => setSpinning(value => !value)}>
                    {spinning ? 'pause' : 'spin'}
                </button>
            </div>
        </div>
    );
};

export interface SpacePoint {
    id: string;
    smiles: string;
    x: number;
    y: number;
    z: number;
    value: number | null;
}

/**
 * The selection's chemical space, rotating in three components.
 *
 * The map itself is two-dimensional; this is the same construction carried one
 * axis further, so a cluster that looks tight on the map can be checked for
 * depth it was hiding.
 *
 * @param points Projected points in a unit box.
 * @param explained Fraction of variance on each component.
 * @param colourBy Descriptor the shading encodes.
 * @param theme Active theme.
 */
export const Space3D: React.FC<{
    points: SpacePoint[]; explained: number[]; colourBy: string;
    height?: number; theme: NexusTheme; onPick?: (id: string) => void;
}> = ({ points, explained, colourBy, height = 210, theme, onPick }) => {
    const { rotation, spinning, setSpinning, onPointerDown, onPointerMove, onPointerUp } =
        useRotation(points.length > 0);
    const size = 300;
    const [hovered, setHovered] = useState<SpacePoint | null>(null);

    const bounds = useMemo(() => {
        const values = points.map(p => p.value).filter((v): v is number => v !== null && Number.isFinite(v));
        return values.length ? { lo: Math.min(...values), hi: Math.max(...values) } : null;
    }, [points]);

    const drawn = useMemo(() => points
        .map(point => ({ point, ...project(point, rotation) }))
        .sort((a, b) => a.depth - b.depth), [points, rotation]);

    if (points.length === 0) return null;
    const toScreen = (v: number) => size / 2 + v * (size / 2) * 0.82;
    const toScreenY = (v: number) => height / 2 - v * (height / 2) * 0.82;

    return (
        <div className="nexus-three relative">
            <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="text-[10px] font-medium uppercase tracking-[0.14em]"
                    style={{ color: theme.text2 }}>Chemical space, in depth</span>
                <span className="text-[10px] tabular-nums" style={{ color: theme.text3 }}>
                    {points.length} · PC1–3 {explained.map(v => `${(v * 100).toFixed(1)}%`).join(' / ')}
                </span>
            </div>
            <svg
                viewBox={`0 0 ${size} ${height}`} width="100%" height={height}
                role="img"
                aria-label={`Rotating three-component projection of ${points.length} molecules`}
                style={{ cursor: 'grab', touchAction: 'none', borderRadius: 12,
                    background: theme.surfaceInset, border: `1px solid ${theme.line}` }}
                onPointerDown={onPointerDown} onPointerMove={onPointerMove}
                onPointerUp={onPointerUp} onPointerLeave={onPointerUp}
            >
                {[0.25, 0.5, 0.75].map(f => (
                    <line key={f} x1={0} x2={size} y1={height * f} y2={height * f}
                        stroke={theme.grid} strokeWidth={1} />
                ))}
                {drawn.map(({ point, x, y, depth }) => {
                    const r = (1.6 + 2.6 * depth);
                    const colour = point.value !== null && bounds
                        ? rampColor(point.value, bounds.lo, bounds.hi, theme.ramp)
                        : theme.series[0];
                    return (
                        <circle key={point.id} cx={toScreen(x)} cy={toScreenY(y)} r={r}
                            fill={colour} fillOpacity={0.35 + 0.65 * depth}
                            stroke={theme.surfaceInset} strokeWidth={0.8}
                            onMouseEnter={() => setHovered(point)}
                            onMouseLeave={() => setHovered(null)}
                            onClick={() => onPick?.(point.id)}
                            style={{ cursor: onPick ? 'pointer' : 'default' }} />
                    );
                })}
            </svg>
            {hovered && (
                <div className="pointer-events-none absolute left-2 top-7 rounded-md px-2 py-1 text-[10px]"
                    style={{ background: theme.tooltipBg, color: theme.text,
                        border: `1px solid ${theme.lineStrong}`, maxWidth: '90%' }}>
                    <div className="truncate font-mono">{hovered.smiles}</div>
                    {hovered.value !== null && (
                        <div style={{ color: theme.text2 }}>
                            {colourBy} {hovered.value.toFixed(2)}
                        </div>
                    )}
                </div>
            )}
            <div className="mt-1 flex items-center gap-2 text-[9.5px]" style={{ color: theme.text3 }}>
                <span>drag to rotate · shaded by {colourBy}</span>
                <button type="button" className="ml-auto underline" style={{ color: theme.text3 }}
                    onClick={() => setSpinning(value => !value)}>
                    {spinning ? 'pause' : 'spin'}
                </button>
            </div>
        </div>
    );
};
