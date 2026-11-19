// Usage: node tests/test_exact_mean_report.cjs /absolute/path/to/synthetic-report.html
// Requires Playwright and Chrome; the Python integration test creates this fixture.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');

(async () => {
  const browser = await chromium.launch({ channel: 'chrome', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1360, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(process.argv[2]).href);
    const root = page.locator('#mean-evidence');
    const slides = root.locator('.evidence-slide');
    assert.equal(await slides.count(), 2);
    await slides.first().locator(':scope > summary').click();
    await root.locator('h3').first().waitFor({ state: 'visible' });
    const images = slides.first().locator('img');
    assert.equal(await images.count(), 3);
    for (const image of await images.all()) {
      assert(await image.evaluate(node => node.complete && node.naturalWidth > 0));
    }
    assert.equal(await slides.first().locator('svg rect').count(), 20);
    await slides.first().locator('details > summary').click();
    assert.equal(await slides.first().locator('details tbody tr').count(), 1);
    const search = root.locator('input[type=search]');
    await search.fill('case-2');
    assert.equal(await root.locator('.evidence-slide:visible').count(), 1);
    assert.match(await root.locator('#mean-evidence-count').innerText(), /1 \/ 2/);
    await search.fill('');
    assert.equal(await root.locator('.evidence-slide:visible').count(), 2);
    await slides.first().locator('a[href^="#evidence-crop"]').first().click();
    assert(await page.locator('figure:target').isVisible());
    await root.scrollIntoViewIfNeeded();
    if (process.env.EXACT_MEAN_SCREENSHOT) {
      await page.screenshot({ path: process.env.EXACT_MEAN_SCREENSHOT, fullPage: true });
    }
    assert.deepEqual(errors, []);
    console.log('Chrome: evidence cards, embedded images, histogram, regions, search, crop links passed.');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
