/**
 * Drives the built extension panel and captures what it renders.
 *
 * The panel bundle, its stylesheet and the Python backend are the real ones;
 * only the Mantis SDK is a shim, so these images show the panel's own output.
 */
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
// Serve from the extension root: the browser normalises '../dist/x'
// to '/dist/x' before it asks, so a server rooted at the harness
// directory would never find it.
const root = path.resolve(here, '..');
const out = path.resolve(here, '../../paper/screenshots');
const PORT = 8771;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json' };

const server = createServer(async (req, res) => {
    const rel = (req.url || '/').split('?')[0];
    const file = path.join(root, rel === '/' ? 'harness/index.html' : rel);
    try {
        const body = await readFile(file);
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
        res.end(body);
    } catch { res.writeHead(404).end('not found'); }
});
await new Promise(resolve => server.listen(PORT, resolve));

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
// A Mantis dock is a narrow column; shoot at that width so the layout is the
// one the panel actually gets.
const page = await browser.newPage({ viewport: { width: 420, height: 1000 },
                                     deviceScaleFactor: 2, colorScheme: 'dark' });
const problems = [];
page.on('console', m => { if (m.type() === 'error') problems.push(m.text().slice(0, 160)); });
page.on('pageerror', e => problems.push(String(e).slice(0, 200)));

await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);
console.log('scope:', await page.locator('.nexus-count').innerText().catch(() => '?'));
await page.screenshot({ path: `${out}/ext_01_idle.png` });

await page.locator('select').first().selectOption('map').catch(() => undefined);
await page.waitForTimeout(600);
await page.getByRole('button', { name: /^Analyze \d+/ }).click();
await page.waitForSelector('text=/Scaffold families/', { timeout: 120000 });
// Let the 3D views arrive and spin a little before capturing.
await page.waitForTimeout(6000);
await page.screenshot({ path: `${out}/ext_02_analyze.png` });

const body = page.locator('.nexus-body');
await body.evaluate(el => { el.scrollTop = 300; });
await page.waitForTimeout(2500);
await page.screenshot({ path: `${out}/ext_03_space3d.png` });
await body.evaluate(el => { el.scrollTop = 560; });
await page.waitForTimeout(2500);
await page.screenshot({ path: `${out}/ext_04_conformer.png` });
await body.evaluate(el => { el.scrollTop = 1100; });
await page.waitForTimeout(1200);
await page.screenshot({ path: `${out}/ext_05_charts.png` });

// Next batch: active learning against the measured column.
await page.getByRole('button', { name: 'Next batch' }).click();
await page.waitForTimeout(800);
const selects = page.locator('.nexus-controls select');
await selects.nth(1).selectOption('measured_logS').catch(() => undefined);
await page.waitForTimeout(500);
await page.getByRole('button', { name: /Propose \d+ to screen/ }).click();
await page.waitForSelector('text=/Acquisition score/', { timeout: 120000 });
await page.waitForTimeout(1500);
await page.screenshot({ path: `${out}/ext_06_next_batch.png` });

// Predict with conformal intervals.
await page.getByRole('button', { name: 'Predict' }).click();
await page.waitForTimeout(800);
await page.locator('.nexus-controls select').nth(1).selectOption('measured_logS').catch(() => undefined);
await page.waitForTimeout(400);
await page.getByRole('button', { name: 'Predict unmeasured' }).click();
await page.waitForSelector('text=/measured vs/', { timeout: 120000 });
await page.waitForTimeout(1500);
await page.screenshot({ path: `${out}/ext_07_predict.png` });
await page.locator('.nexus-controls input[type="checkbox"]').first().check().catch(() => undefined);

console.log('bag written:', JSON.stringify(await page.evaluate(() => window.__NEXUS_BAG__ ?? null)));
console.log('selection size:', await page.evaluate(async () => (await window.mantis.selection.get()).selected.length));
console.log(problems.length ? `console errors (${problems.length}):\n  ${problems.slice(0, 4).join('\n  ')}` : 'no console errors');
await browser.close();
server.close();
