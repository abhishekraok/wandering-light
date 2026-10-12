# Haiku 5.5 on the v4 40-task sample

[haiku55-sample40 (#181)](https://github.com/abhishekraok/wandering-light/issues/181).
Run 2026-10-11, Claude Code 2.1.296. Same seed-17 sample, prompt, parser and
eight held-out examples as the [README plot](../stack-pbe-v4-effort-sweep-20260927/README.md).
One generator call per task at low / medium / high / xhigh / max; tools, MCP,
settings and session persistence disabled through `ClaudeExecTokenGenerator`.

| effort | solved /40 | memorised | illegal | out tok med / mean | $/task |
|---|---:|---:|---:|---:|---:|
| low | 16 | 4 | 16 | 2,950 / 5,822 | 0.0030 |
| medium | 17 | 5 | 15 | 6,221 / 8,774 | 0.0045 |
| high | 18 | 9 | 10 | 4,474 / 12,819 | 0.0065 |
| xhigh | 16 | 8 | 12 | 8,402 / 15,516 | 0.0079 |
| max | 22 | 6 | 6 | 35,019 / 57,844 | 0.0291 |

Low to xhigh is flat at 16–18, below the 22–27 top band (±3-task noise on 40
tasks). Max reaches the band's lower edge, 22, at $0.029/task: about 5x the
cost of gpt-6.1-sol low (23, $0.006) and level with gpt-6-sol max (26, $0.022).
Haiku is not a cheaper top-band point. Its low arm (16, $0.003) is the best
score at that cost among hosted models; gpt-6-luna needs max for 15.

Haiku spends many more tokens than the larger Claude models at the same
effort (Sonnet 5.5 low: 1,112 mean), and 6–16 answers per arm do not run.
The max gain is in the 21–40-op bucket (7/12 against 2–4/12 below max).

All 200 calls were served by `claude-haiku-5-5` with no generation failure;
`validation.json` holds an independent re-grade of every candidate. Four max
calls report a turn proxy of 1 (`num_turns - 1`): each produced more than
128k output tokens and was continued for a second turn. Tools were disabled
in the command; no call emitted a text tool call. Full table in
[table.md](table.md).

Cost uses the plot's estimate: a fixed 1,300-token prompt plus all recorded
output tokens, including thinking, at $0.10/$0.50 per million tokens. Calls
ran on the Claude subscription. One sample per task on 40 tasks supports a
cost comparison, not a reliable model ranking.

Reproduce from the repository root:

```bash
export PYTHON=${PYTHON:-.venv/bin/python}
bash reports/stack-pbe-v4-haiku55-20261011/run.sh
PYTHONPATH=. "$PYTHON" reports/stack-pbe-v4-haiku55-20261011/analyse.py > reports/stack-pbe-v4-haiku55-20261011/table.md
"$PYTHON" reports/stack-pbe-v4-effort-sweep-20260927/analyse.py > reports/stack-pbe-v4-effort-sweep-20260927/table.md
"$PYTHON" reports/stack-pbe-v4-effort-sweep-20260927/plot.py
```
