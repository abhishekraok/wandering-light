# GPT-6.1 Terra on the fixed v4 40-task sample

Tracking: [terra61-sample40 (#191)](https://github.com/abhishekraok/wandering-light/issues/191).

Prepared October 27, 2026 on base `839159cb392dbacb82d05debb6a59282a487d190`.
Codex CLI: `0.163.0` before launch; no update performed. Python 3.12.12,
environment created with `uv sync --extra dev --extra experiments`.

## Status

Prepared; no completed results yet. Run low first and inspect raw per-call
identity evidence before starting medium, high and xhigh. Ultra and Max are
excluded under the owner's October 15 effort instruction (message 1155).
No comparison claim is made before validation.

## Protocol

Same seed-17 sample of 40 `stack_pbe_v4` eval tasks as the Sol 6.1 and Haiku 5.5
reports. The task indices were checked against the saved Sol 6.1 low arm.
`experiments/stack_pbe_baseline.py`, the prompt/parser in it, `wandering_light/solver.py`,
the stack machine, held-out grader and corpus manifest are unchanged from the
existing sweep baseline. One untooled `codex exec` call per task through the
subscription; no paid API route. `capture.py` executes that baseline unchanged
and saves the command and raw stdout/stderr from each Codex subprocess. It does
not change the prompt, model command, output parser, grader or number of calls.

The Codex adapter's existing summary parser omits model identity. Raw evidence
will be inspected before any claim that the requested model served every call.
Independent candidate regrading and a reproducible table remain required.

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

## Run

From the sweep worktree root:

```bash
EFFORTS=low bash reports/stack-pbe-v4-terra61-20261027/run.sh
# After checking the low-arm audit evidence:
EFFORTS='medium high xhigh' bash reports/stack-pbe-v4-terra61-20261027/run.sh
```

`PYTHON` can select an environment. `OUT` can select the report output directory;
the supervised run writes into the clone's main report directory from launch.
The runner refuses to overwrite an existing arm. No arm has been rerun.
