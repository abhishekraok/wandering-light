/**
 * Drive the explorer in a headless browser and save screenshots.
 *
 *   npm install --no-save playwright       # one-off
 *   npx playwright install chromium        # one-off, ~115MB
 *   python -m wandering_light.webapp &     # serve the built app
 *   node scripts/screenshot.mjs /tmp/shots
 *
 * Rendering bugs do not fail a unit test: a graph with no edges, a control
 * panel stretched across the canvas and a blank minimap all passed jsdom and
 * every backend assertion. Looking at the page is the check.
 */
// Not a dependency of this package: installing playwright pulls a browser
// download into every `npm install`, and this script is an occasional check.
// `npm install --no-save playwright` (or NODE_PATH) is enough to run it.
let chromium;
try {
  ({ chromium } = await import("playwright"));
} catch {
  console.error(
    "playwright is not installed. Run:\n" +
      "  npm install --no-save playwright && npx playwright install chromium",
  );
  process.exit(2);
}

const OUT = process.argv[2] ?? "/tmp/wl-shots";
// CSS.escape is a browser global; the ids here are JSON strings full of quotes.
const CSS = { escape: (value) => value.replace(/["\\]/g, "\\$&") };

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1600, height: 950 } });
const problems = [];
page.on("console", (m) => {
  if (m.type() === "error") problems.push(`console: ${m.text()}`);
});
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("requestfailed", (r) => problems.push(`requestfailed: ${r.url()}`));

async function shot(name) {
  await page.screenshot({ path: `${OUT}/${name}.png` });
  console.log(`shot ${name}`);
}

/**
 * Wait until the canvas holds at least this much, then report what it holds.
 *
 * Edges render only once nodes are measured and fitView animates, so a bare
 * count taken straight after an action reads a half-drawn graph. Falling
 * through on timeout keeps a real regression visible as a wrong number rather
 * than as a crash in the harness.
 */
async function counts(minNodes = 0, minEdges = 0) {
  await page
    .waitForFunction(
      ([nodes, edges]) =>
        document.querySelectorAll(".react-flow__node").length >= nodes &&
        document.querySelectorAll(".react-flow__edge").length >= edges,
      [minNodes, minEdges],
      { timeout: 6000 },
    )
    .catch(() => {});
  await page.waitForTimeout(200);
  return (
    `${await page.locator(".react-flow__node").count()} nodes, ` +
    `${await page.locator(".react-flow__edge").count()} edges`
  );
}

/** A node clear of the floating controls and minimap, so a drag reaches it. */
async function draggableNode() {
  const boxes = await page.locator(".react-flow__node").evaluateAll((nodes) =>
    nodes.map((node) => {
      const rect = node.getBoundingClientRect();
      return { id: node.getAttribute("data-id"), x: rect.x, y: rect.y, w: rect.width, h: rect.height };
    }),
  );
  const clear = boxes.find((box) => {
    const centre = { x: box.x + box.w / 2, y: box.y + box.h / 2 };
    const nearControls = centre.x < 420 && centre.y < 190;
    const nearMinimap = centre.x > 1000 && centre.y < 220;
    return !nearControls && !nearMinimap && centre.y > 200 && centre.y < 700;
  });
  if (!clear) throw new Error("no node clear of the canvas overlays");
  return page.locator(`.react-flow__node[data-id="${CSS.escape(clear.id)}"]`);
}

await page.goto("http://127.0.0.1:8765/", { waitUntil: "networkidle" });
await page.waitForTimeout(700);
console.log("on load:", await counts(1, 0));
await shot("01-initial");

// The edge picker, with a live preview per candidate function.
await page.getByRole("button", { name: /add step/ }).click();
await page.waitForTimeout(500);
await shot("02-picker");

// Build double -> inc, which reaches the default target.
await page.locator(".picker .option", { hasText: /^\s*double/ }).first().click();
await page.waitForTimeout(450);
await page.getByRole("button", { name: /add step/ }).click();
await page.waitForTimeout(450);
await page.locator(".picker .option", { hasText: /^\s*inc/ }).first().click();
await page.waitForTimeout(700);
console.log("after two steps:", await counts(3, 2));
await shot("03-trajectory-on-canvas");

// Expand from the current state with the default palette.
await page.getByRole("button", { name: /^expand/ }).click();
await page.waitForTimeout(2000);
console.log("after expand:", await counts(20, 20));
await shot("04-expanded");

// Self-loops and involutions should be visible in the drawing.
console.log("self-loop edges:", await page.locator(".edge-label.loop").count());

// Select a node, then drag it and check the position sticks. The locator is
// pinned by id, not by index: React Flow reorders a dragged node in the DOM, so
// an index locator silently measures a different node afterwards.
const node = await draggableNode();
await node.click();
await page.waitForTimeout(600);
await shot("05-node-selected");
/** Position in graph coordinates, which a later fitView does not change. */
const graphPosition = async (locator) => {
  const style = await locator.getAttribute("style");
  const match = /translate\((-?[\d.]+)px, *(-?[\d.]+)px\)/.exec(style ?? "");
  return match ? { x: Number(match[1]), y: Number(match[2]) } : null;
};

const box = await node.boundingBox();
const before = await graphPosition(node);
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.down();
await page.mouse.move(box.x + 140, box.y + 120, { steps: 12 });
await page.mouse.up();
await page.waitForTimeout(400);
const after = await graphPosition(node);
console.log(
  "drag moved node by",
  Math.round(after.x - before.x),
  Math.round(after.y - before.y),
);
await shot("06-node-dragged");

// Expanding again must keep the dragged node where it was put.
await page.getByRole("button", { name: /^expand/ }).click();
await page.waitForTimeout(1800);
const afterExpand = await graphPosition(node);
console.log(
  "second expand:",
  await counts(40, 40),
  "| dragged node stayed:",
  Math.abs(afterExpand.x - after.x) < 1 && Math.abs(afterExpand.y - after.y) < 1,
);
await shot("07-expanded-twice");

// Clear returns the canvas to the trajectory alone.
await page.getByRole("button", { name: /clear canvas/ }).click();
await page.waitForTimeout(900);
console.log("after clear:", await counts(3, 2));
await shot("08-cleared");

await page.getByRole("button", { name: /run solver/ }).click();
await page.waitForTimeout(2500);
await shot("09-solver");

await page.getByRole("button", { name: "corpus", exact: true }).click();
await page.waitForTimeout(1500);
await shot("10-corpus");

await page.getByRole("button", { name: "basis", exact: true }).click();
await page.waitForTimeout(800);
await shot("11-basis");

console.log(problems.length ? "PROBLEMS:\n" + problems.join("\n") : "no console/page errors");
await browser.close();
