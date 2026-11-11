# Fable 5.2 low on the fixed v4 40-task sample

[fable52-sample40 (#197)](https://github.com/abhishekraok/wandering-light/issues/197).
Prepared November 10, 2026, with Claude Code 2.1.322; no CLI update by this desk.
The owner authorized **low effort only**, message 4946 (November 10):
“It must be too powerful. Try the lowest effort only.” No other effort is run.

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

A separate low-effort `Reply exactly OK.` preflight must identify
`claude-fable-5-2` in the CLI JSON's `modelUsage` before the arm starts.
It is excluded from all benchmark scores and task costs. Every evaluation
call will be audited by the same served-model field and independently
re-graded before publication. No rerun for score improvement.

The plot retains its existing estimate: 1,300 uncached input tokens plus all
recorded output tokens, including thinking, at **$8 input / $40 output per
million tokens**. Source: [Anthropic list prices](https://platform.claude.com/docs/en/about-claude/pricing),
read November 10, 2026. Calls use the Claude subscription. Using Fable 5.1
low's recorded tokens at these rates predicts about $2.78 list equivalent
for the 40 calls, roughly 0.23 plan points under the historical Opus
calibration; Fable's plan weighting may differ. The Claude Max 20x meter
read 19% before preparation, below the 60% start limit.

Status: preparation only. Results and independent validation will be added
when the one low arm finishes. Differences of at most three tasks are ties
at the resolution of this sample; one arm cannot establish a model ranking.

Reproduce from the repository root (use a fresh output directory):

```bash
export PYTHON=${PYTHON:-.venv/bin/python}
export OUT=/absolute/fresh-output/stack-pbe-v4-fable52-20261110
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. "$PYTHON" reports/stack-pbe-v4-fable52-20261110/preflight.py --out "$OUT/preflight.json"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHON="$PYTHON" OUT="$OUT" bash reports/stack-pbe-v4-fable52-20261110/run.sh
```

Inference-time source commit and digests are recorded in the arm's
provenance sidecar. Raw CLI intermediates
remain local under ignored `raw/`; no authentication files are collected.
