/*
 * A stand-in for the Mantis panel SDK.
 *
 * Only the surface the panel calls is implemented, and every backend call is
 * forwarded to the real extension backend through bridge.py. The molecules
 * come from the same ESOL set the seeded Mantis space holds, so what the panel
 * renders here is what it renders there.
 */
(function () {
    const BRIDGE = 'http://127.0.0.1:8770/invoke';
    const points = window.__NEXUS_POINTS__ || [];
    const listeners = {};
    let selected = [];

    async function invoke(action, payload) {
        const response = await fetch(BRIDGE, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action, payload }),
        });
        const body = await response.json();
        if (!body.ok) throw new Error(body.error);
        return body.result;
    }

    function emit(name, payload) {
        for (const handler of listeners[name] || []) handler(payload);
    }

    window.mantis = {
        extension: { getManifest: async () => ({ id: 'nexus.drug-discovery', version: '0.2.0' }) },
        maps: {
            getActive: async () => ({ mapId: 'harness-map' }),
            list: async () => ({ maps: [{ mapId: 'harness-map' }], activeMapId: 'harness-map' }),
            getPoints: async () => points,
            open: async () => ({ ok: true }),
        },
        selection: {
            get: async () => ({ selected, currentSelection: selected[0] ?? null, selectedBags: [] }),
            set: async (ids) => { selected = ids; emit('selection.changed', { selected }); return { ok: true }; },
        },
        bags: {
            list: async () => [],
            create: async (name, ids) => { window.__NEXUS_BAG__ = { name, count: ids.length }; return { ok: true }; },
            highlight: async (name) => { window.__NEXUS_HIGHLIGHT__ = name; return { ok: true }; },
            unhighlight: async () => ({ ok: true }),
        },
        edges: {
            set: async (c) => { window.__NEXUS_EDGES__ = c; return { ok: true, matchedCount: c.to.length }; },
            clear: async () => ({ ok: true }),
        },
        panels: { open: async () => ({ ok: true }), close: async () => ({ ok: true }) },
        commands: { execute: async () => ({ ok: true }), registerCommand: async () => ({ dispose: async () => undefined }) },
        backend: { invoke },
        events: {
            subscribe: async (name, _options, handler) => {
                (listeners[name] = listeners[name] || []).push(handler);
                return { dispose: async () => { listeners[name] = (listeners[name] || []).filter(h => h !== handler); } };
            },
        },
        workspaceState: { get: async (_k, d) => d, update: async () => undefined },
        globalState: { get: async (_k, d) => d, update: async () => undefined },
    };
})();
