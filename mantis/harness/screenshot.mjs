/**
 * Drives the real NexusPanel in a browser and captures what it renders.
 *
 * The panel talks to the live NEXUS service, so these images show genuine
 * output: descriptors computed by the Python package, a batch chosen by the
 * active-learning loop, and conformal intervals from the calibrated model.
 */
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(here, '../../paper/screenshots');
const PORT = 8766;

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json' };

const server = createServer(async (req, res) => {
    const rel = (req.url || '/').split('?')[0];
    const file = path.join(here, rel === '/' ? 'index.html' : rel);
    try {
        const body = await readFile(file);
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
        res.end(body);
    } catch {
        res.writeHead(404).end('not found');
    }
});
await new Promise(resolve => server.listen(PORT, resolve));

const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
const page = await browser.newPage({ viewport: { width: 1280, height: 820 },
                                     deviceScaleFactor: 2 });
const problems = [];
page.on('console', m => { if (m.type() === 'error') problems.push(m.text()); });
page.on('pageerror', e => problems.push(String(e)));

await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: 'networkidle' });
await page.waitForTimeout(800);

/**
 * Captures the viewport.
 *
 * @param name File stem to write under paper/screenshots.
 */
async function shot(name) {
    await page.screenshot({ path: path.join(outDir, `${name}.png`) });
    console.log(`  captured ${name}`);
}

// The header reports the live service version when the backend is reachable.
const header = await page.locator('text=/service v|service offline/').first().textContent();
console.log(`  service status in panel: ${header}`);
await shot('01_panel_initial');

// --- Analyze: descriptors for every selected molecule, then colour the map.
await page.getByRole('button', { name: /Analyze \d+ molecules/ }).click();
await page.waitForSelector('text=/analyzed in/', { timeout: 60000 });
await page.waitForTimeout(400);
await shot('02_analyze_results');

await page.locator('#nexus-colour').selectOption('logp');
await page.waitForTimeout(500);
await shot('03_map_coloured_by_logp');

// --- Next batch: active learning picks what to screen next.
await page.getByRole('button', { name: 'Next batch' }).click();
await page.locator('#nexus-label').selectOption('measured_logS');
await page.getByRole('button', { name: /Propose \d+ to screen/ }).click();
await page.waitForSelector('text=/acquisition over/', { timeout: 60000 });
await page.waitForTimeout(400);
await shot('04_next_batch');

// --- Predict: conformal intervals on the unmeasured molecules.
await page.getByRole('button', { name: 'Predict' }).click();
await page.locator('#nexus-label').selectOption('measured_logS');
await page.getByRole('button', { name: 'Predict unmeasured' }).click();
await page.waitForSelector('text=/coverage guarantee/', { timeout: 60000 });
await page.waitForTimeout(400);
await shot('05_predict_with_intervals');

// --- Dark mode, which Mantis panels must support.
await page.evaluate(() => { document.documentElement.dataset.theme = 'dark'; });
await page.getByRole('button', { name: 'Analyze' }).click();
await page.waitForTimeout(600);
await shot('06_dark_mode');

console.log(problems.length
    ? `  console errors: ${problems.length}\n   ${problems.slice(0, 5).join('\n   ')}`
    : '  no console errors');

await browser.close();
server.close();
