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

cpSync(path.join(here, 'src', 'utilities.css'), path.join(dist, 'utilities.css'));
cpSync(path.join(here, 'src', 'panel.css'), path.join(dist, 'panel.css'));
cpSync(path.join(here, 'src', 'host.js'), path.join(dist, 'extension.js'));

// 2. Vendor the Python package beside the backend entry point.
rmSync(backendPackage, { recursive: true, force: true });
cpSync(path.join(repo, 'nexus'), backendPackage, {
    recursive: true,
    filter: (src) => !src.includes('__pycache__') && !src.endsWith('.pyc'),
});

// 3. Drop anything Python left behind. Running the backend -- in the harness,
// or a stray import -- writes __pycache__ next to the source, and a .pyc is
// not UTF-8 text, so the package would be refused for a file the build itself
// caused.
function clean(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (entry.isSymbolicLink()) continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === '__pycache__') rmSync(full, { recursive: true, force: true });
            else clean(full);
        } else if (full.endsWith('.pyc')) {
            rmSync(full, { force: true });
        }
    }
}
clean(path.join(here, 'backend'));

// 4. Every utility class the panel references must be defined in the
// stylesheet we ship. In-tree the host's Tailwind supplies these; in an
// extension iframe nothing does, and an undefined class is silent -- the
// layout just collapses. Checking it here turns that into a build failure.
{
    const sources = [
        path.join(repo, 'mantis/src/app/components/panels/nexus/charts.tsx'),
        path.join(repo, 'mantis/src/app/components/panels/nexus/three.tsx'),
        path.join(here, 'src/panel.tsx'),
    ];
    const used = new Set();
    for (const file of sources) {
        const text = readFileSync(file, 'utf8');
        for (const match of text.matchAll(/className="([^"$]+)"/g)) {
            for (const token of match[1].split(/\s+/)) {
                if (token && !token.startsWith('nexus-')) used.add(token);
            }
        }
    }
    const stylesheet = readFileSync(path.join(here, 'src/utilities.css'), 'utf8');
    // Class selectors in the stylesheet, with CSS escapes removed so
    // `.text-\[10px\]` matches the `text-[10px]` written in the markup.
    const defined = new Set(
        [...stylesheet.matchAll(/^\.([^\s,{]+)/gm)].map(m => m[1].replace(/\\/g, '')));
    const missing = [...used].filter(name => !defined.has(name)).sort();
    if (missing.length) {
        console.error('utility classes used by the panel but not shipped:');
        for (const name of missing) console.error(`  ${name}`);
        process.exit(1);
    }
    console.log(`utilities      ${used.size} classes used, all defined`);
}

// 5. Check the limits Mantis enforces, then zip.
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
