// Exercise the debugger UI's render path against a real session payload,
// with a DOM stub. Catches runtime errors that a syntax check cannot.
import fs from "node:fs";
import vm from "node:vm";

const html = fs.readFileSync(process.argv[2], "utf8");
const js = html.match(/<script>([\s\S]*?)<\/script>/g).pop()
  .replace(/^<script>/, "").replace(/<\/script>$/, "");

const session = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const drive = JSON.parse(fs.readFileSync(process.argv[4], "utf8"));
const logits = JSON.parse(fs.readFileSync(process.argv[5], "utf8"));
const encoding = JSON.parse(fs.readFileSync(process.argv[6], "utf8"));

const made = new Map();
function element(id) {
  const node = {
    id, _html: "", _text: "", className: "", title: "",
    set innerHTML(v) { this._html = String(v); },
    get innerHTML() { return this._html; },
    set textContent(v) { this._text = String(v); },
    get textContent() { return this._text; },
    querySelectorAll: () => [],
    querySelector: () => null,
    scrollIntoView: () => {},
  };
  made.set(id, node);
  return node;
}

const handlers = [];
const sandbox = {
  console,
  document: {
    getElementById: (id) => made.get(id) || element(id),
    addEventListener: (_type, fn) => handlers.push(fn),
    querySelectorAll: () => [],
  },
  location: { search: "?distance=3&seed=11&beam_width=8" },
  history: { replaceState: () => {} },
  URLSearchParams,
  setInterval: () => 0,
  clearInterval: () => {},
  Math, JSON, Number, String, Object, Array, Boolean, Error, RegExp,
  fetch: async (url) => {
    const path = url.split("?")[0];
    const body = path === "/api/session" ? session
      : path === "/api/drive" ? drive
      : path === "/api/logits" ? logits
      : path === "/api/encoding" ? encoding
      : {};
    return { ok: true, status: 200, json: async () => body };
  },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;

vm.createContext(sandbox);
vm.runInContext(js, sandbox, { filename: "ui.js" });

const press = async (key) => {
  for (const fn of handlers) await fn({ key, preventDefault() {}, metaKey: false });
};

const report = [];
const check = (name, condition, detail) => {
  report.push(`${condition ? "ok  " : "FAIL"} ${name}${detail ? " — " + detail : ""}`);
  if (!condition) process.exitCode = 1;
};

await new Promise((r) => setTimeout(r, 60));

const beam = () => made.get("beam")._html;
const tape = () => made.get("tape")._html;
const insp = () => made.get("insp")._html;

check("session rendered a beam", beam().includes("node"), beam().slice(0, 40));
check("task line filled", made.get("taskline")._html.includes("certified"));
check("target shown", made.get("out-val")._text.length > 0);
check("expansion tape rendered", tape().includes("exp"));
check("inspector shows normalized probability", insp().includes("type-normalized"));
check("inspector shows raw score section", insp().includes("raw policy score"));
if (session.map) {
  check("PCA map rendered",
    insp().includes("policy's own space") && insp().includes("<svg"));
}

await press("ArrowRight");
check("step forward kept the tape", tape().includes("exp"));
await press("ArrowDown");
check("layer change kept the beam", beam().includes("node"));
await press("2");
check("beam node selection marks one selected", beam().includes("sel"));
await press("v");
check("logits panel rendered", insp().includes("raw logits"));
await press("e");
check("encoding panel rendered", insp().includes("exact model input"));
check("encoding shows tensor shapes", insp().includes("sequences"));
await press("q");
check("q returns to actions", insp().includes("type-normalized"));
await press("t");
check("witness toggle hides it", made.get("witness")._html.includes("hidden"));
await press("t");
await press("m");
check("drive mode entered", made.get("r-mode")._text.startsWith("drive"));
check("drive shows keycaps", insp().includes("keycap") && insp().includes("press the key"));
check("drive lists a first action", insp().includes(drive.actions[0].function));
await press("a");
check("drive apply did not throw", true);
await press("w");
check("drive undo did not throw", true);
await press("m");
check("returned to replay", made.get("r-mode")._text.startsWith("replay"));
await press(" ");
await press("r");
check("reset returned to first layer", beam().includes("node"));

console.log(report.join("\n"));
