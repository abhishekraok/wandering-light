# Gemini 4 Argon and 3.8 Flash on the fixed v4 40-task sample

Tracking: [gemini-sample40 (#196)](https://github.com/abhishekraok/wandering-light/issues/196).
Run 2026-11-06 (PST) through the Gemini REST API, at the three thinking levels
both models accept (low / medium / high). Same seed-17 sample, prompt, parser and
held-out grader as [the effort sweep](../stack-pbe-v4-effort-sweep-20260927/README.md).
The owner stopped the run early as clearly off the frontier, so Flash high is
partial and failed calls were not retried. Not plotted in the README figure.

| model | thinking | solved | memorised | timed out (900 s) | $ per completed call | mean thinking tokens |
|---|---|---:|---:|---:|---:|---:|
| gemini-4-argon | low | 12/40 | 0 | 1 | 0.0029 | 0 |
| gemini-4-argon | medium | 19/40 | 7 | 7 | 0.0232 | 2,042 |
| gemini-4-argon | high | 23/40 | 5 | 12 | 0.0517 | 4,890 |
| gemini-3.8-flash | low | 17/40 | 4 | 3 (+2 dropped) | 0.0185 | 4,664 |
| gemini-3.8-flash | medium | 19/40 | 2 | 10 | 0.0369 | 9,558 |
| gemini-3.8-flash | high | 9/36, partial | 3 | 20 (+1 503) | 0.0601 | 15,760 |

**Dominated at every level.** gpt-6.1-sol medium solves 25/40 at $0.007/task;
Argon high costs 5–7x that for 23. Best Gemini point is Argon high, one below
the 24–27 top band.

- **Cost** is list price x (1.3k-token prompt + recorded output, thinking
  included): Argon $2/$10 per 1M (intro price to 2026-11-30, then $4/$20),
  Flash $0.75/$3.75. Timed-out calls report no usage, so the figure is the
  mean over completed calls, as in earlier sweeps. Whether Google bills a
  call the client abandoned is unknown: counting those as free, Argon high is
  $0.036/task.
- **Timeouts** are scored unsolved, as in earlier sweeps. Some are service
  hangs (Argon low does not think and otherwise answers in under a minute;
  Flash high timed out on 12-op `min_pair`), the rest are probably runaway
  thinking. 24 shards ran at once, which may have added to the hangs.
- **Other failures**, also unsolved: 2 dropped connections (Flash low; the
  generator now retries those), one persistent 503 (Flash high), and one
  `PROHIBITED_CONTENT` safety block each on Argon medium and Flash low.
- Argon's server-counted prompt is ~566 tokens, Flash's ~544; the 1.3k
  convention overstates input cost slightly.

Files: `run.sh` (4 shards per arm; launched from commit `044a4ee`, see
`commit.txt`), `*_shard?.json` / `.log` per shard, `analyse.py` → `summary.json`,
`table.md`; `plot.py` draws the README figure with these rows added.
