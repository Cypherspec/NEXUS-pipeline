/** Bundles the real NexusPanel with stubbed Mantis stores for screenshotting. */
import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');

await build({
    entryPoints: [path.join(here, 'main.tsx')],
    outfile: path.join(here, 'bundle.js'),
    bundle: true,
    format: 'iife',
    jsx: 'automatic',
    loader: { '.json': 'json' },
    // The panel lives in this repository, not inside the Mantis tree, so point
    // esbuild at Mantis's installed packages for react, lucide-react and so on.
    nodePaths: [path.join('/home/user/mantis', 'node_modules')],
    define: { 'process.env.NEXT_PUBLIC_NEXUS_URL': '"http://127.0.0.1:8765"' },
    alias: {
        // The four Mantis modules the panel depends on are stubbed; everything
        // else resolves to the real source in this repository.
        '@/app/components/map/points/pointColorStore': path.join(here, 'stubs/mantisStores.tsx'),
        '@/app/components/stores/dataStores': path.join(here, 'stubs/mantisStores.tsx'),
        '@/app/components/stores/selectionStores': path.join(here, 'stubs/mantisStores.tsx'),
        '@/app/context/DarkModeContext': path.join(here, 'stubs/mantisStores.tsx'),
        '@/app/types': path.join(here, 'stubs/mantisStores.tsx'),
        '@/app/services/nexusApi': path.join(repo, 'src/app/services/nexusApi.ts'),
        '@/types/nexus': path.join(repo, 'src/types/nexus.ts'),
        '@/app/components/panels/NexusPanel': path.join(repo, 'src/app/components/panels/NexusPanel.tsx'),
    },
    logLevel: 'info',
});
