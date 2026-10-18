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
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' };
const server = createServer(async (req, res) => {
    const rel = (req.url || '/').split('?')[0];
    try {
        const file = path.join(root, rel === '/' ? 'harness/index.html' : rel);
        const body = await readFile(file);
        res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] ?? 'application/octet-stream' });
        res.end(body);
    } catch (e) { res.writeHead(404).end('not found: ' + rel); }
});
await new Promise(r => server.listen(8772, r));
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH });
const page = await browser.newPage({ viewport: { width: 420, height: 900 } });
const logs = [];
page.on('console', m => logs.push(`[${m.type()}] ${m.text().slice(0, 200)}`));
page.on('pageerror', e => logs.push(`[pageerror] ${String(e).slice(0, 300)}`));
page.on('requestfailed', r => logs.push(`[404?] ${r.url()} ${r.failure()?.errorText}`));
await page.goto('http://127.0.0.1:8772/', { waitUntil: 'networkidle' });
await page.waitForTimeout(3000);
console.log('root html length:', await page.evaluate(() => document.getElementById('root')?.innerHTML.length ?? -1));
console.log('mantis shim:', await page.evaluate(() => typeof window.mantis));
console.log('points:', await page.evaluate(() => (window.__NEXUS_POINTS__ || []).length));
console.log('--- logs:\n' + logs.join('\n'));
await browser.close(); server.close();
