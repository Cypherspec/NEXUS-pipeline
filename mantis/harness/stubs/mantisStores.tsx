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

/** A stand-in for the map: one swatch per point the panel coloured. */
export const MapPreview: React.FC = () => {
    const [, force] = React.useReducer((n: number) => n + 1, 0);
    React.useEffect(() => {
        listeners.add(force);
        return () => { listeners.delete(force); };
    }, []);
    const entries = [...colourState.colors.entries()];
    return (
        <div className="flex h-full flex-col gap-2 p-3">
            <div className="text-xs text-gray-500">
                Map stand-in — {entries.length
                    ? `${entries.length} points coloured by the panel`
                    : 'no colours written yet'}
            </div>
            <div className="grid flex-1 content-start gap-1"
                style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(18px, 1fr))' }}>
                {(entries.length ? entries : SELECTED.map(p => [p.id, '#e1e0d9'] as const))
                    .map(([id, colour]) => (
                        <div key={id} title={id}
                            className="aspect-square rounded-sm"
                            style={{ backgroundColor: colour }} />
                    ))}
            </div>
        </div>
    );
};
