# GPT-6.1 Terra on the fixed v4 40-task sample

Tracking: [terra61-sample40 (#191)](https://github.com/abhishekraok/wandering-light/issues/191).
Run October 27, 2026; Codex CLI 0.163.0 before/after every arm, no update.

## Results used in the plot

| effort | solved /40 | memorised | illegal | mean output tokens | $/task | server identity confirmed /40 | saved measurement |
|---|---:|---:|---:|---:|---:|---:|---|
| low | 9 | 6 | 17 | 124.6 | 0.001923 | 40 | identity rerun |
| medium | 20 | 11 | 6 | 286.4 | 0.002732 | 40 | identity rerun |
| high | 20 | 10 | 6 | 308.9 | 0.002844 | 40 | original |
| xhigh | 21 | 15 | 3 | 504.8 | 0.003824 | 40 | original |
| max | 24 | 14 | 0 | 1771.7 | 0.010158 | 40 | requested extension |

The best verified result is **24/40 at max, $0.010158/task**. It enters the
figure's existing 22–27 shaded band and is within three tasks of the best
hosted saved score, 27/40. Max costs 2.66 times xhigh for three more solves,
within this sample's ±3-task comparison convention. Medium gives 20/40 at
$0.002732/task. Sol 6.1 medium's saved 25/40 at $0.006976/task is both
higher-scoring and cheaper than Terra max; this small sample does not establish
a reliable model ranking.

### Authorized extensions and retained first runs

The first low/medium measurements lacked serving-model proof. Their candidates
and token usage reproduce, but later identity checks cannot validate them
retrospectively. They remain saved in this directory:

| first measurement | solved /40 | memorised | illegal | mean output tokens | $/task | identities confirmed /40 |
|---|---:|---:|---:|---:|---:|---:|
| low | 9 | 3 | 19 | 77.8 | 0.001689 | 0 |
| medium | 18 | 15 | 5 | 277.0 | 0.002685 | 0 |

Max was initially excluded under owner message 1155. The owner explicitly
requested it in message 2210 (“Try max as well.”). Operator message 2212 then
quoted the owner's authorization to rerun low and medium as well, so their
serving identities could be proved. These are explicit exceptions to the
no-rerun rule for missing identity evidence, not retries after machine failures
or attempts to improve scores.

The new low/medium runs are saved separately under `identity-rerun/`. The plot
uses those runs regardless of their scores: low stayed at 9/40 and medium
changed from the unconfirmed 18/40 to a verified 20/40. Neither original was
overwritten. No other arm was rerun. Ultra remains excluded.

## Protocol and serving-model audit

Same seed-17 sample of 40 `stack_pbe_v4` eval tasks as Sol 6.1 and Haiku 5.5.
Task indices match the saved Sol 6.1 low arm. The unchanged
`experiments/stack_pbe_baseline.py`, prompt/parser, `wandering_light/solver.py`,
stack machine, held-out grader and corpus manifest are used throughout.
One untooled subscription `codex exec` call per task; no paid API route.
`capture.py` observes commands and stdout/stderr without changing inference,
the parser, the grader, or the number of samples.

Before each new arm (max, low rerun, medium rerun), a separate low-effort
`Reply exactly OK.` preflight confirmed server-reported `gpt-6.1-terra`.
The three `preflight-before-*.json` files retain compact evidence.
These three probes are excluded from evaluation candidates and costs.
Every one of the 200 plotted evaluation calls independently confirms that same
served model. The adapter omits identity in its own summary, so incoming
websocket response fields, rather than the requested `-m`, establish it.

Original low used ordinary captured output. Original medium added SSE tracing,
which did not capture its websocket responses. High/xhigh and all new arms used
`RUST_LOG=codex_api::sse::responses=trace,tungstenite::protocol=trace`.
This changes logging only. `extract_audit.py` creates `cli-audit.json` from
the retained raw traces: original command/stdout, selected incoming model
fields, and SHA256 of original stderr. Full traces remain ignored local
intermediates under `raw/`; neither full traces nor authentication homes are
needed to reproduce grading from committed artifacts.

All **280 saved candidates** independently regrade and their usage, tools and
CLI error summaries agree with captured stdout. Root `validation.json`
covers the first four arms plus max (200 candidates, 120 confirmed identities).
`identity-rerun/validation.json` covers the two authorized reruns (80/80).
The 80 first low/medium calls remain identity-unconfirmed and excluded from the
plot. Every saved call has the same disabled-Code-Mode availability warning;
there are zero tool calls and zero candidate generation exceptions. Warning
events are retained separately from generation exceptions.

### Provenance and timing

| measurement | source commit | inference dirty state | finished (PDT) |
|---|---|---|---|
| original low | `9c538a7` | unknown; reconstructed metadata | 09:31:40 |
| original medium | `9c538a7` | unknown; reconstructed metadata | 09:39:02 |
| high | `9c538a7` | unknown; reconstructed metadata | 09:53:31 |
| xhigh | `9c538a7` | unknown; reconstructed metadata | 10:09:05 |
| max | `06fb3b0` | recorded clean at inference | 16:36:45 |
| low identity rerun | `d235970` | recorded clean at inference | 16:47:36 |
| medium identity rerun | `3e4594c` | recorded clean at inference | 17:01:12 |

Each arm finished with exit 0. Original metadata is explicitly reconstructed:
preparation was clean, but the actual inference-time dirty flags were not
recorded. New sidecars record actual HEAD/dirty state, arguments, CLI versions,
timestamps, source/manifest digests and final artifact SHA256. The inference
harness, parser, grader and manifest match for all new arms. The saved capture
wrapper digest differs from the current wrapper after review added environment
and cgroup-limit recording for future runs; this is surfaced as a caveat,
without editing the historical sidecars or rerunning inference. Analysis reads the selected
directory's sidecars and reports missing metadata or differences as caveats;
it does not substitute original-run constants for fresh measurements.
The seven small console logs are saved beside the corresponding JSON.
All new arms and all reproductions below use 12 GiB/no-swap memory scopes.
The new arms used the corrected scoped runner. Future sidecars additionally
record the actual cgroup's memory/swap limit and RUST_LOG; these fields were
not captured in the already-completed arms and have not been backfilled.
The runner now defaults to both websocket and SSE identity trace targets.

## Prices

Standard short-context list prices checked October 27, 2026: $1 per million
uncached input tokens, $5 per million output tokens.
[OpenAI pricing](https://developers.openai.com/api/docs/pricing),
[Terra 6.1 model page](https://developers.openai.com/api/docs/models/gpt-6.1-terra),
[launch announcement](https://openai.com/index/introducing-gpt-6-1-terra/).
The unchanged plot estimate is a fixed 1,300-token uncached prompt plus recorded
output; it excludes the CLI's residual tool-schema input overhead. It is an
API-list comparison, not the subscription's marginal bill.

## Reproduce without model calls

From the sweep worktree root, with the manifest-pinned corpus available:

```bash
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- env PYTHONPATH=. .venv/bin/python reports/stack-pbe-v4-terra61-20261027/analyse.py > reports/stack-pbe-v4-terra61-20261027/table.md
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- env PYTHONPATH=. .venv/bin/python reports/stack-pbe-v4-terra61-20261027/analyse.py --data-dir reports/stack-pbe-v4-terra61-20261027/identity-rerun > reports/stack-pbe-v4-terra61-20261027/identity-rerun/table.md
.venv/bin/python reports/stack-pbe-v4-effort-sweep-20260927/analyse.py > reports/stack-pbe-v4-effort-sweep-20260927/table.md
.venv/bin/python reports/stack-pbe-v4-effort-sweep-20260927/plot.py
```

Root `table.md` includes the original low/medium plus high/xhigh/max.
`identity-rerun/table.md` includes the verified replacement low/medium.
The unified analysis explicitly selects replacement low/medium and original
high/xhigh/max, reading validation from each selected artifact's directory.
Its summary records each Terra row's source artifact. All 57 earlier non-Terra
summary rows remain unchanged.

New measurements spend subscription calls. Use a fresh absolute `OUT` outside
the worktree and a separate preflight before each arm. For one new max arm:

```bash
export OUT=/absolute/path/to/fresh-output
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- env PYTHONPATH=. .venv/bin/python reports/stack-pbe-v4-terra61-20261027/preflight.py --out "$OUT/preflight-before-max.json" && EFFORTS=max bash reports/stack-pbe-v4-terra61-20261027/run.sh
```

`preflight.py` reproduces the same adapter command, lowest effort and probe
prompt as the three saved historical preflights, with portable output paths.
It retains incoming identity evidence, exits nonzero without confirmed Terra
identity/usage/zero tools, and refuses existing evidence paths. Both it and the
runner default to the trace targets above; an explicit RUST_LOG overrides them.
The runner's default efforts are low/medium/high/xhigh; callers must preflight
each arm separately when selecting multiple efforts. It refuses existing JSON or logs,
caps each baseline at 12 GiB/no swap, and stops on failure. A capped/incomplete
arm is never retried uncapped or scored as zero for missing output.
