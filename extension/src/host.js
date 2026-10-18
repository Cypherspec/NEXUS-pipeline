/*
 * NEXUS extension host.
 *
 * The host is a Web Worker that outlives any single panel, so it owns the
 * things that should survive the panel being closed: the command other
 * surfaces can call, and the count of how often this space has been analysed.
 * The analysis itself belongs to the panel and the Python backend; the host
 * deliberately holds no chemistry.
 */

exports.activate = async function activate(context) {
    const runs = await context.workspaceState.get('analysisRuns', 0);

    context.subscriptions.push(
        await mantis.commands.registerCommand(
            'nexus.drug-discovery.analyzeSelection',
            async () => {
                // Resolve the map once and carry it, rather than letting each
                // call answer about whichever map happens to be first.
                const active = await mantis.maps.getActive();
                const mapId = active && active.mapId;
                const selection = await mantis.selection.get(mapId);
                const points = await mantis.maps.getPoints(mapId);

                const selected = new Set(selection && selection.selected ? selection.selected : []);
                const scope = selected.size
                    ? points.filter((point) => selected.has(point.id))
                    : points;

                const molecules = [];
                for (const point of scope) {
                    const metadata = point.metadata || {};
                    const smiles = metadata.smiles || metadata.SMILES || metadata.Smiles
                        || metadata.canonical_smiles || metadata.structure || metadata.mol;
                    if (typeof smiles === 'string' && smiles.trim()) {
                        molecules.push({ id: point.id, smiles: smiles.trim() });
                    }
                }

                if (molecules.length === 0) {
                    return { ok: false, reason: 'no points in scope carry a structure' };
                }

                const result = await mantis.backend.invoke('analyze', { molecules });
                const next = runs + 1;
                await context.workspaceState.update('analysisRuns', next);

                return {
                    ok: true,
                    analyzed: result.nAnalyzed,
                    scaffolds: result.nScaffolds,
                    dropped: result.nDropped,
                    analysisRuns: next,
                };
            },
        ),
    );
};

exports.deactivate = function deactivate() {};
