/**
 * Minimal stand-ins for the four Mantis modules NexusPanel imports.
 *
 * Only these are stubbed. The panel itself, its service client and the NEXUS
 * backend are the real ones, so a screenshot taken through this harness shows
 * genuine output from the pipeline rather than mock data.
 */
import React from 'react';

import points from '../points.json';

export interface DataPoint {
    id: string;
    data_row?: string;
    title?: string;
    cluster?: string;
    embedding?: [number, number];
    metadata?: Record<string, unknown>;
}

// Only part of the set carries a measurement, which is the realistic case:
// some compounds have been assayed, the rest are candidates to screen next.
const N_MEASURED = 28;

const SELECTED: DataPoint[] = (points as { id: string; smiles: string; logS: number }[])
    .map((p, i) => ({
        id: p.id,
        title: p.smiles,
        metadata: i < N_MEASURED
            ? { smiles: p.smiles, measured_logS: p.logS }
            : { smiles: p.smiles },
    }));

/** Points the harness pretends the user has selected on a map. */
export function useSelectionStore<T>(selector: (s: { selected: DataPoint[] }) => T): T {
    return selector({ selected: SELECTED });
}

export function useDataStore<T>(selector: (s: { map_id: string; space_id: string }) => T): T {
    return selector({ map_id: 'harness-map', space_id: 'harness-space' });
}

export function useDarkMode(): { isDarkMode: boolean } {
    return { isDarkMode: document.documentElement.dataset.theme === 'dark' };
}

/** Records the colours the panel writes, and shows them as a strip. */
const colourState: { colors: Map<string, string> } = { colors: new Map() };
const listeners = new Set<() => void>();

export const usePointColorStore = {
    getState: () => ({
        updateMultiplePointColors: (colors: Map<string, string>) => {
            colourState.colors = new Map(colors);
            listeners.forEach(fn => fn());
        },
        removeColors: (ids: Iterable<string>) => {
            const next = new Map(colourState.colors);
            for (const id of ids) next.delete(id);
            colourState.colors = next;
            listeners.forEach(fn => fn());
        },
    }),
};

/**
 * A stand-in for the map.
 *
 * Point *positions* here are a deterministic layout derived from each id --
 * illustrative only, since the harness has no embedding. The point *colours*
 * are the real ones the panel wrote through the colour store, which is what
 * these screenshots are meant to show.
 */
export const MapPreview: React.FC = () => {
    const [, force] = React.useReducer((n: number) => n + 1, 0);
    React.useEffect(() => {
        listeners.add(force);
        return () => { listeners.delete(force); };
    }, []);
    const entries = [...colourState.colors.entries()];
    const coloured = new Map(entries);

    // Golden-angle spiral: stable, evenly spread, and obviously a layout rather
    // than a claim about chemical similarity.
    const layout = SELECTED.map((p, i) => {
        const angle = i * 2.399963;
        const radius = Math.sqrt((i + 0.5) / SELECTED.length);
        return {
            id: p.id,
            title: p.title ?? p.id,
            x: 50 + Math.cos(angle) * radius * 42,
            y: 50 + Math.sin(angle) * radius * 42,
        };
    });

    return (
        <div className="flex h-full flex-col gap-2 p-3"
            style={{ background: '#07090d', color: '#93a4ba' }}>
            <div className="flex items-baseline gap-2">
                <span className="text-[10px] font-medium uppercase tracking-[0.14em]">
                    Map stand-in
                </span>
                <span className="text-[10px]" style={{ color: '#5d6e84' }}>
                    {entries.length
                        ? `${entries.length} points coloured by the panel`
                        : 'no colours written yet'}
                </span>
            </div>
            <div className="relative flex-1 overflow-hidden rounded-lg"
                style={{ background: '#0a0d12', border: '1px solid rgba(255,255,255,0.07)' }}>
                <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"
                    className="h-full w-full" role="img" aria-label="map stand-in point cloud">
                    <defs>
                        <radialGradient id="glow">
                            <stop offset="0%" stopColor="#4f9dff" stopOpacity="0.16" />
                            <stop offset="100%" stopColor="#4f9dff" stopOpacity="0" />
                        </radialGradient>
                    </defs>
                    <circle cx={50} cy={50} r={46} fill="url(#glow)" />
                    {[20, 32, 44].map(r => (
                        <circle key={r} cx={50} cy={50} r={r} fill="none"
                            stroke="rgba(255,255,255,0.045)" strokeWidth={0.2} />
                    ))}
                    {layout.map(point => (
                        <circle key={point.id} cx={point.x} cy={point.y} r={1.25}
                            fill={coloured.get(point.id) ?? '#263241'}
                            stroke="#0a0d12" strokeWidth={0.35}>
                            <title>{point.title}</title>
                        </circle>
                    ))}
                </svg>
            </div>
            <div className="text-[9px]" style={{ color: '#5d6e84' }}>
                Positions are an illustrative layout; colours come from the panel.
            </div>
        </div>
    );
};
