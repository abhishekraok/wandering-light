# Sonnet 5.5 on the v4 40-task sample

[sonnet55-sample40 (#176)](https://github.com/abhishekraok/wandering-light/issues/176).
Released 2026-09-28; run started that day and finished 2026-09-29.
Same seed-17 sample, prompt, parser and
eight held-out examples as the [README plot](../stack-pbe-v4-effort-sweep-20260927/README.md).
One generator call per task at low / medium / high / xhigh / max; tools, MCP,
settings and session persistence disabled through `ClaudeExecTokenGenerator`.
The existing generator's per-call deadline is 30 minutes.

Xhigh reached 26/40 at $0.05870/task. The existing GPT-6 sol max run scores
26/40 at $0.02186/task under the same [estimate](../stack-pbe-v4-effort-sweep-20260927/table.md).

[Results](table.md). Raw candidates and per-call usage are in
`sonnet55_*_v4_sample40.json`; `validation.json` records sample indices,
source hashes, runtime versions and an independent re-grade of all candidates.
Generation failures count as unsolved without retry; their recorded output
tokens remain in the cost estimate. A cost point requires model identity and
nonzero output usage for all 40 calls; unusable metadata is recorded, not
treated as zero.

Max returned four `refusal` errors after 129–131k output tokens each, scored
as unsolved. Its eight reported `tool_calls` are the adapter's
`max(num_turns - 1, 0) + permission_denials` proxy, not an execution trace.
All other calls have zero proxy counts; no call emitted a text tool call.
Tools were disabled in the command; the full trajectory was not captured.

Cost uses the plot's existing estimate: a fixed 1,300-token prompt plus all
recorded output tokens, including thinking, at
[Sonnet 5.5's $2/$10 per million tokens](https://www.anthropic.com/claude-sonnet-5-5).
Calls ran on the Claude subscription; dollars estimate API list cost,
excluding Claude Code's input overhead and caching. One sample per task on
40 tasks supports a cost comparison, not a reliable model ranking. Like the
earlier curves, the scores include the sample's known `aligned_matches`
shortcut; finite held-out checks do not prove the intended behaviour.

Reproduce from the repository root (set `PYTHON` to a Python 3.12 environment
with the repository dependencies if `.venv/bin/python` is elsewhere):

```bash
export PYTHON=${PYTHON:-.venv/bin/python}
PYTHONPATH=. "$PYTHON" reports/stack-pbe-v4-sonnet55-20260928/preflight.py
bash reports/stack-pbe-v4-sonnet55-20260928/run.sh
PYTHONPATH=. "$PYTHON" reports/stack-pbe-v4-sonnet55-20260928/analyse.py > reports/stack-pbe-v4-sonnet55-20260928/table.md
"$PYTHON" reports/stack-pbe-v4-effort-sweep-20260927/analyse.py > reports/stack-pbe-v4-effort-sweep-20260927/table.md
"$PYTHON" reports/stack-pbe-v4-effort-sweep-20260927/plot.py
```
