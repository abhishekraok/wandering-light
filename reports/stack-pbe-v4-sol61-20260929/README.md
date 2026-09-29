# GPT-6.1 Sol on the v4 40-task sample

[sol61-sample40 (#177)](https://github.com/abhishekraok/wandering-light/issues/177).
Released and run 2026-09-29, Codex CLI 0.159.1 (0.155.1 did not list the model).
Same seed-17 sample, prompt, parser and held-out grader as the
[README plot](../stack-pbe-v4-effort-sweep-20260927/README.md); one untooled
`codex exec` call per task at low / medium / high / xhigh.

| effort | solved /40 | memorised | illegal | out tok med / mean | $/task |
|---|---:|---:|---:|---:|---:|
| low | 23 | 14 | 1 | 322 / 342 | 0.0060 |
| medium | 25 | 14 | 0 | 382 / 438 | 0.0070 |
| high | 24 | 16 | 0 | 500 / 756 | 0.0102 |
| xhigh | 24 | 15 | 0 | 860 / 1,133 | 0.0139 |

Effort is flat: every arm is in the 22–27 top band (±3-task noise on 40
tasks), so max was not run. Medium's 25 at $0.007/task is ~3x cheaper than
gpt-6-sol max (26, $0.022) and the cheapest point in the band. Cost uses the
plot's estimate at [$2/$10 per 1M input/output](https://openai.com/index/introducing-gpt-6-1-sol/)
(same as gpt-6-sol): 1.3k-token prompt plus recorded output, subscription calls priced at list.

Reproduce from the repository root (`PYTHON`, `OUT` optional):

```bash
bash reports/stack-pbe-v4-sol61-20260929/run.sh
python reports/stack-pbe-v4-effort-sweep-20260927/analyse.py > reports/stack-pbe-v4-effort-sweep-20260927/table.md
python reports/stack-pbe-v4-effort-sweep-20260927/plot.py
```
