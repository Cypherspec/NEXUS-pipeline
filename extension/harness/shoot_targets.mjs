/** Drives the Targets mode on the Davis kinase panel. */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');
const out = path.resolve(here, '../../paper/screenshots');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = createServer(async (req, res) => {
    const rel = (req.url || '/').split('?')[0];
    // Chromium asks for this unprompted; a 404 here is noise that hides real ones.
    if (rel === '/favicon.ico') { res.writeHead(204).end(); return; }
    // Serve the Davis fixture in place of the solubility one.
    const wanted = rel === '/harness/points.js' ? '/harness/points_davis.js' : rel;
    const file = path.join(root, wanted === '/' ? 'harness/index.html' : wanted);
    try {
        const body = await readFile(file);
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
        res.end(body);
    } catch { res.writeHead(404).end('not found'); }
});
await new Promise(r => server.listen(8774, r));

const bridge = spawn('python3', [path.join(here, 'bridge.py'), '--port', '8770'],
                     { stdio: ['ignore', 'pipe', 'pipe'] });
bridge.stderr.on('data', c => process.stderr.write(`  [bridge] ${c}`));
const stop = () => { try { bridge.kill('SIGTERM'); } catch { /* gone */ } };
process.on('exit', stop);
for (let i = 0; i < 40; i += 1) {
    try {
        const r = await fetch('http://127.0.0.1:8770/invoke', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'analyze', payload: { molecules: [{ id: 'p', smiles: 'CCO' }] } }),
        });
        if (r.ok) break;
    } catch { /* not up */ }
    await new Promise(r => setTimeout(r, 1000));
}

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
const missing = [];
const page = await browser.newPage({ viewport: { width: 420, height: 1000 },
                                     deviceScaleFactor: 2, colorScheme: 'dark' });
const problems = [];
page.on('response', r => { if (r.status() >= 400) missing.push(`${r.status()} ${new URL(r.url()).pathname}`); });
page.on('console', m => { if (m.type() === 'error') problems.push(m.text().slice(0, 150)); });
page.on('pageerror', e => problems.push(String(e).slice(0, 200)));

await page.goto('http://127.0.0.1:8774/', { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);
await page.locator('select').first().selectOption('map').catch(() => undefined);
await page.waitForTimeout(500);
await page.getByRole('button', { name: 'Targets', exact: true }).click();
await page.waitForTimeout(900);
console.log('control row:', (await page.locator('.nexus-controls').innerText()).replace(/\n+/g, ' | '));
await page.screenshot({ path: `${out}/ext_10_targets_idle.png` });

await page.locator('.nexus-run').click();
await page.waitForSelector('text=Enriched terms', { timeout: 120000 });
await page.waitForTimeout(2500);
await page.screenshot({ path: `${out}/ext_11_targets.png` });
const body = page.locator('.nexus-body');
await body.evaluate(el => { el.scrollTop = 420; });
await page.waitForTimeout(1200);
await page.screenshot({ path: `${out}/ext_12_targets_table.png` });
console.log('result:', (await body.innerText()).slice(0, 420).replace(/\n+/g, ' | '));
console.log(problems.length ? `console errors: ${problems.slice(0, 3).join(' / ')}` : 'no console errors');
console.log(missing.length ? `missing: ${[...new Set(missing)].join(' ')}` : 'no missing requests');
await browser.close(); server.close(); stop();
