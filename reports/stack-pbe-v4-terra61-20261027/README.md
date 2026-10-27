# GPT-6.1 Terra on the fixed v4 40-task sample

Tracking: [terra61-sample40 (#191)](https://github.com/abhishekraok/wandering-light/issues/191).

Prepared October 27, 2026 on base `839159cb392dbacb82d05debb6a59282a487d190`.
Codex CLI: `0.163.0` before launch; no update performed. Python 3.12.12,
environment created with `uv sync --extra dev --extra experiments`.

## Results

Run October 27, 2026, Codex CLI `0.163.0` before and after; no update.
Every arm completed once with exit 0; no retries or additional probe calls.

| effort | solved /40 | memorised | illegal | mean output tokens | $/task | server identity confirmed /40 |
|---|---:|---:|---:|---:|---:|---:|
| low | 9 | 3 | 19 | 77.8 | 0.001689 | 0 |
| medium | 18 | 15 | 5 | 277.0 | 0.002685 | 0 |
| high | 20 | 10 | 6 | 308.9 | 0.002844 | 40 |
| xhigh | 21 | 15 | 3 | 504.8 | 0.003824 | 40 |

The best verified result is **21/40 at xhigh, $0.003824/task**. High gives
20/40 for $0.002844; xhigh costs 34% more for one additional solve, within
the sample's ±3-task noise. Both observed scores sit below the existing plot's
22–27 shaded band. This is a small-sample cost comparison, not a reliable ranking.

**Identity limitation:** low/medium grades and usage reproduce, but their saved
CLI output lacks a server model field. They remain provisional and are omitted
from the figure; the table preserves them with explicit identity counts.
High/xhigh have server-reported `gpt-6.1-terra` on all 80 calls. Those later
checks do not establish low/medium identity retrospectively.

High was seven tasks below the best hosted score (27), so xhigh was not
skipped. Ultra and Max were excluded under the owner's October 15 instruction
(message 1155).

## Protocol

Same seed-17 sample of 40 `stack_pbe_v4` eval tasks as the Sol 6.1 and Haiku 5.5
reports. The task indices were checked against the saved Sol 6.1 low arm.
`experiments/stack_pbe_baseline.py`, the prompt/parser in it, `wandering_light/solver.py`,
the stack machine, held-out grader and corpus manifest are unchanged from the
existing sweep baseline. One untooled `codex exec` call per task through the
subscription; no paid API route. `capture.py` executes that baseline unchanged
and saves the command and raw stdout/stderr from each Codex subprocess. It does
not change the prompt, model command, output parser, grader or number of calls.

Inference used preparation commit `9c538a7f7b8ce1ec763b98993250da59d3e2c66d`.
The inference source was clean at preparation; subsequent edits were confined
to report analysis/documentation and the plot. `validation.json` records source
and corpus-manifest digests, task indices, commit and CLI provenance.

The adapter's summary parser omits server identity. Low used ordinary captured
stdout/stderr; medium added `RUST_LOG=codex_api::sse::responses=trace`, which
showed websocket event names but no complete SSE events. High/xhigh added
`tungstenite::protocol=trace` to retain received websocket messages. This changed
logging only, without changing transport or inference settings.

`analyse.py` independently regrades each completed arm, reproduces its table,
and writes `validation.json` with per-call identity status. Unconfirmed identity
is reported as unconfirmed, never inferred from the command's requested model.
`cli-audit.json` retains every command and original CLI stdout, selected received
response identity fields, and SHA256 of each original stderr. `extract_audit.py`
extracts these fields from the locally retained transport traces; full traces
remain generated intermediates under ignored `raw/`, rather than adding 13 MB
of compressed transport/debug output to git. `analyse.py` checks all 160 saved
usage/tool/error summaries against the retained stdout. All 160 calls emit one
identical fail-closed Code Mode
availability warning because its host is disabled; none attempts a tool.
These CLI error events are retained in validation separately from generation
exceptions. All 160 candidates independently regrade, with usage for 160/160,
zero tool calls and zero generation exceptions. The full sweep does **not**
pass the every-call identity requirement: 80/160 are confirmed.

## Prices and comparison

Standard short-context prices checked October 27, 2026: $1 per million uncached
input tokens and $5 per million output tokens. Sources:
[OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[Terra 6.1 model page](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
[launch announcement](https://openai.com/index/introducing-gpt-6-1-terra/).
The plot estimate remains a fixed 1,300-token uncached prompt plus recorded output.

Historical saved-table comparisons, not new regrades: `gpt-5.6-terra` max scored
19/40 at $0.0633/task; `gpt-6.1-sol` medium scored 25/40 at $0.0070/task. The best
hosted saved score is 27/40; use the existing ±3-task comparison convention.

## Reproduce

From the sweep worktree root:

```bash
PYTHONPATH=. .venv/bin/python reports/stack-pbe-v4-terra61-20261027/analyse.py > reports/stack-pbe-v4-terra61-20261027/table.md
.venv/bin/python reports/stack-pbe-v4-effort-sweep-20260927/analyse.py > reports/stack-pbe-v4-effort-sweep-20260927/table.md
.venv/bin/python reports/stack-pbe-v4-effort-sweep-20260927/plot.py
```

To make a new measurement, use a fresh output directory (the runner refuses
to overwrite existing arms). These commands spend subscription calls:

```bash
export OUT=/path/to/fresh-output
export RUST_LOG=codex_api::sse::responses=trace,tungstenite::protocol=trace
EFFORTS='low medium high xhigh' bash reports/stack-pbe-v4-terra61-20261027/run.sh
.venv/bin/python reports/stack-pbe-v4-terra61-20261027/extract_audit.py --data-dir "$OUT" --out "$OUT/cli-audit.json"
PYTHONPATH=. .venv/bin/python reports/stack-pbe-v4-terra61-20261027/analyse.py --data-dir "$OUT"
```

Original arm completion times (PDT): low 09:31:40, medium 09:39:02,
high 09:53:31, xhigh 10:09:05. Each ran as a supervised user systemd unit with
MemoryMax=12G and MemorySwapMax=0. Console logs sit beside their saved JSON.
