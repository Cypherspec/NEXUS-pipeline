/**
 * Builds and packages the NEXUS Mantis extension.
 *
 * Three things happen here, in order:
 *   1. the panel is bundled to a single file with esbuild, because a Mantis
 *      panel is loaded as one script inside a sandboxed iframe;
 *   2. the `nexus` Python package is vendored under backend/, so the backend
 *      imports nothing from outside its own directory;
 *   3. the result is zipped with the manifest at the package root.
 *
 * The package must be UTF-8 text throughout -- Mantis decodes every file on
 * import and refuses binaries -- so the build asserts that before zipping
 * rather than letting the import fail with a less specific message.
 *
 *     node build.mjs
 */

import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const dist = path.join(here, 'dist');
const backendPackage = path.join(here, 'backend', 'nexus');

/** Mantis refuses any package file that is not valid UTF-8 text. */
const MAX_FILE_BYTES = 5 * 1024 * 1024;
const MAX_PACKAGE_BYTES = 25 * 1024 * 1024;

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });

// 1. The panel, bundled with React into one file.
await build({
    entryPoints: [path.join(here, 'src', 'panel.tsx')],
    outfile: path.join(dist, 'panel.js'),
    bundle: true,
    format: 'iife',
    jsx: 'automatic',
    minify: true,
    target: ['es2020'],
    // React and the shared chart module come from the Mantis checkout's
    // packages; nothing of Mantis's own source is modified or shipped beyond
    // the two chart files this repository wrote.
    nodePaths: ['/home/user/mantis/node_modules'],
    logLevel: 'warning',
});

cpSync(path.join(here, 'src', 'panel.css'), path.join(dist, 'panel.css'));
cpSync(path.join(here, 'src', 'host.js'), path.join(dist, 'extension.js'));

// 2. Vendor the Python package beside the backend entry point.
rmSync(backendPackage, { recursive: true, force: true });
cpSync(path.join(repo, 'nexus'), backendPackage, {
    recursive: true,
    filter: (src) => !src.includes('__pycache__') && !src.endsWith('.pyc'),
});

// 3. Check the limits Mantis enforces, then zip.
const decoder = new TextDecoder('utf-8', { fatal: true });
let total = 0;
const offenders = [];

/** Checks one file against the UTF-8 and size rules. */
function check(full) {
    const bytes = readFileSync(full);
    total += bytes.length;
    if (bytes.length > MAX_FILE_BYTES) {
        offenders.push(`${path.relative(here, full)} is ${bytes.length} bytes (cap ${MAX_FILE_BYTES})`);
    }
    try {
        decoder.decode(bytes);
    } catch {
        offenders.push(`${path.relative(here, full)} is not valid UTF-8 text`);
    }
}

/** Walks only what the package will contain, following no symlinks. */
function inspect(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isSymbolicLink()) continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) inspect(full);
        else check(full);
    }
}

for (const folder of ['dist', 'backend']) inspect(path.join(here, folder));
check(path.join(here, 'mantis.extension.json'));

if (offenders.length) {
    console.error('package would be refused by Mantis:');
    for (const line of offenders) console.error(`  ${line}`);
    process.exit(1);
}

const manifest = JSON.parse(readFileSync(path.join(here, 'mantis.extension.json'), 'utf8'));
const outFile = path.join(here, `nexus-drug-discovery-${manifest.version}.zip`);
rmSync(outFile, { force: true });
execFileSync('zip', ['-q', '-r', outFile, 'mantis.extension.json', 'dist', 'backend',
    '-x', '*__pycache__*', '*.pyc'], { cwd: here });

const packaged = statSync(outFile).size;
if (packaged > MAX_PACKAGE_BYTES) {
    console.error(`package is ${(packaged / 1e6).toFixed(1)} MB, over the 25 MB cap`);
    process.exit(1);
}

console.log(`panel bundle   ${(statSync(path.join(dist, 'panel.js')).size / 1024).toFixed(0)} KB`);
console.log(`package        ${path.basename(outFile)}  ${(packaged / 1024).toFixed(0)} KB`);
console.log(`manifest       ${manifest.id} v${manifest.version}, ${manifest.permissions.length} permissions, ${manifest.backend.actions.length} backend actions`);
