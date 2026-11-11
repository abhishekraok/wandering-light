# Fable 5.2 low on the fixed v4 40-task sample

[fable52-sample40 (#197)](https://github.com/abhishekraok/wandering-light/issues/197).
Run November 10, 2026, with Claude Code 2.1.322; no CLI update by this desk.
The owner authorized **low effort only**, message 4946 (November 10):
“It must be too powerful. Try the lowest effort only.” No other effort is run.

| effort | solved /40 | memorised | illegal | out tok med / mean | $/task |
|---|---:|---:|---:|---:|---:|
| low | 22 | 17 | 1 | 714 / 2,085 | 0.09380 |

**22/40 at $0.093802/task** is below the plot's current 24–27 top band
(within three solves of the best hosted score, 27). It matches Fable 5.1
low's 22, with 10% higher cost ($0.085350 previously): 44% more recorded
output tokens offset the 20% token-price reduction. GPT-6.1 Sol low scores
23 at $0.006024/task, about 16 times cheaper. A one-task score difference
is within this sample's noise; these measurements support a cost comparison,
not a reliable ranking of model ability. The new point is outside the
measured hosted frontier and appears grey under the plot's existing rule.

All 40 calls were served by `claude-fable-5-2`, all 40 saved grades reproduce,
and all 40 original responses produce the same full Score after replaying
the parser. There were zero generation errors, tool/turn proxies or text
tool attempts. Shown fit is 39/40 (97.5%): 17 fit the shown examples and fail
the held-out examples. [Full table](table.md); [validation](validation.json).

Same seed-17 40-task eval sample, prompt, parser and eight held-out examples
as the [Haiku report](../stack-pbe-v4-haiku55-20261011/README.md) and README
plot. One generator call per task through the unchanged
`experiments/stack_pbe_baseline.py` and `ClaudeExecTokenGenerator`; tools,
MCP, settings and session persistence disabled. The observational
`capture.py` records original CLI commands, stdout, stderr and source/memory
provenance without changing the baseline protocol.

Before calls, all 40 selected indices matched the saved Haiku and Fable 5.1
low arms. Prompt, parser, solve function, complete Claude adapter, stack
machine and grading helpers match Haiku base `1aab85c`. The repository has
since added the search-only opt-in `grade(..., empty_is_identity=False)`;
the LLM baseline uses the unchanged false default. Both versions returned
identical full scores for all 200 saved Haiku candidates in an offline check.

A separate low-effort `Reply exactly OK.` preflight confirmed exactly
`claude-fable-5-2` in the CLI JSON's `modelUsage` at 17:55:56 PST, before
the arm. It is excluded from benchmark scores and task costs. Every
evaluation call was audited by the same served-model field. No arm retry,
additional sample or higher-effort call was made.

The plot retains its existing estimate: 1,300 uncached input tokens plus all
recorded output tokens, including thinking, at **$8 input / $40 output per
million tokens**. Source: [Anthropic list prices](https://platform.claude.com/docs/en/about-claude/pricing),
read November 10, 2026. Calls used the Claude subscription. The plot estimate
totals $3.752080 for 40 tasks. The actual recorded token mix, including
cache writes at $10/$16 per million for 5m/1h and reads at $0.20, has a
$4.168352 list equivalent; this exactly matches the CLI's reported total.
The plot total uses the fixed prompt and recorded output; the actual
token-mix total includes cache writes. The Claude Max 20x meter read 19% before the run
and at 18:14 PST, below the 60% start limit.

The one low arm ran November 10 from 17:56:17 to 18:13:52 PST in
`bench-fable52-20261110.service`, from clean preparation commit
`421b24a00d0fcc7ee83e11c2530ca2b4ed99d774`, exit 0. [Run log](run-status.txt)
and [inference provenance](fable52_low_v4_sample40.provenance.json) retain the
completion and source/memory evidence (12 GiB, zero swap). There are no
provenance caveats. All 62 prior aggregate rows are unchanged.

Reproduce from the repository root (use a fresh output directory):

```bash
export PYTHON=${PYTHON:-.venv/bin/python}
export OUT=/absolute/fresh-output/stack-pbe-v4-fable52-20261110
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. "$PYTHON" reports/stack-pbe-v4-fable52-20261110/preflight.py --out "$OUT/preflight.json"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHON="$PYTHON" OUT="$OUT" bash reports/stack-pbe-v4-fable52-20261110/run.sh
"$PYTHON" reports/stack-pbe-v4-fable52-20261110/extract_audit.py --data-dir "$OUT"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. "$PYTHON" reports/stack-pbe-v4-fable52-20261110/analyse.py --data-dir "$OUT" > "$OUT/table.md"
```

Inference-time source commit and digests are recorded in the arm's
provenance sidecar. Raw CLI intermediates
remain local under ignored `raw/`; no authentication files are collected.

`extract_audit.py` exports the 40 original commands and stdout records with
stdout/stderr digests into [cli-audit.json](cli-audit.json). The compact
evidence and saved arm/preflight/provenance JSONs are committed here. To
reproduce the table **without model calls or raw files**, from the repository root:

```bash
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. .venv/bin/python reports/stack-pbe-v4-fable52-20261110/analyse.py
.venv/bin/python reports/stack-pbe-v4-effort-sweep-20260927/analyse.py
.venv/bin/python reports/stack-pbe-v4-effort-sweep-20260927/plot.py
```

The report analysis prints the committed table and writes `summary.json` /
`validation.json`. It independently replays the original parser on each CLI
response, re-grades the saved candidate, checks the sample and every
usage/model field, and records full Score fields. Analysis with `--data-dir`
reads and writes that selected run's evidence instead.
