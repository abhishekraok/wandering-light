/**
 * Drive the explorer in a headless browser and save screenshots.
 *
 *   npx playwright install chromium        # one-off, ~115MB
 *   python -m wandering_light.webapp &     # serve the built app
 *   node scripts/screenshot.mjs /tmp/shots
 *
 * Rendering bugs do not fail a unit test: a graph with no edges, a control
 * panel stretched across the canvas and a blank minimap all passed jsdom and
 * every backend assertion. Looking at the page is the check.
 */
import { chromium } from "playwright";

const OUT = process.argv[2] ?? "/tmp/wl-shots";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });
const problems = [];
page.on("console", (m) => { if (m.type() === "error") problems.push(`console: ${m.text()}`); });
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("requestfailed", (r) => problems.push(`requestfailed: ${r.url()}`));

async function shot(name) {
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log(`shot ${name}`);
}

await page.goto("http://127.0.0.1:8765/", { waitUntil: "networkidle" });
await page.waitForTimeout(600);
await shot("01-initial");

// Open the edge picker via "add step".
await page.getByRole("button", { name: /add step/ }).click();
await page.waitForTimeout(500);
await shot("02-picker");

// Pick `double`, then add `inc` -> should reach the default target.
await page.locator(".picker .option", { hasText: /^\s*double/ }).first().click();
await page.waitForTimeout(500);
await page.getByRole("button", { name: /add step/ }).click();
await page.waitForTimeout(500);
await page.locator(".picker .option", { hasText: /^\s*inc/ }).first().click();
await page.waitForTimeout(600);
await shot("03-trajectory-hits-target");

// Expand the graph from the current state.
await page.getByRole("button", { name: /expand/ }).click();
await page.waitForTimeout(1500);
await shot("04-graph");

// Select a node a couple of layers out.
const nodes = page.locator(".react-flow__node");
console.log("graph nodes rendered:", await nodes.count());
if (await nodes.count() > 4) {
  await nodes.nth(4).click();
  await page.waitForTimeout(700);
  await shot("05-node-selected");
}

// Zoom in to check labels are legible at working zoom.
await page.locator(".react-flow__controls-zoomin").click();
await page.locator(".react-flow__controls-zoomin").click();
await page.waitForTimeout(600);
await page.screenshot({ path: `${OUT}/05b-graph-zoomed.png`, clip: { x: 340, y: 40, width: 900, height: 620 } });
console.log("shot 05b-graph-zoomed");

// Solver panel.
await page.getByRole("button", { name: /run solver/ }).click();
await page.waitForTimeout(2500);
await shot("06-solver");

// Corpus tab.
await page.getByRole("button", { name: "corpus", exact: true }).click();
await page.waitForTimeout(1500);
await shot("07-corpus");

// Basis tab.
await page.getByRole("button", { name: "basis", exact: true }).click();
await page.waitForTimeout(800);
await shot("08-basis");

console.log(problems.length ? "PROBLEMS:\n" + problems.join("\n") : "no console/page errors");
await browser.close();
