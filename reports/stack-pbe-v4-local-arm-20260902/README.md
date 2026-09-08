# Local-arm alternatives on `stack_pbe_v4` (2026-09-03)

Question: can a free local model do better than the `qwen3:30b-a3b` arm's
6/40 on v4 ([baseline](../stack-pbe-v4-baseline-20260901/README.md))? Same
protocol as that row: seed 17, 40-task eval sample, one sample per task,
300 s per-request deadline, one RTX 4090. Runs sat in a two-hour GPU
window lent by the compute-optimal-policy-size (#80) campaign, so the run
is capped and partial.

## The 2507 thinking variant was already the baseline

Planned first: `qwen3:30b-a3b-thinking-2507-q4_K_M`, the July-2025 update
of the same MoE. Ollama's `qwen3:30b-a3b` tag already resolves to it —
both tags are digest `ad815644918f`, blob `sha256-58574f2e…`, the pull
downloaded nothing, and the library page for the tag describes the 2507
weights. So the v3 and v4 `qwen3:30b-a3b` rows in `docs/STATUS.md` are the
2507 thinking model. The run was stopped after one task
(`qwen3_thinking2507_v4_sample40.json`, 1/1, a replicate fragment, not a
result).

## `gpt-oss:20b`, reasoning effort high

Effort is passed as ollama's `think` level (new `--ollama-think`, off by
default so earlier runs are unchanged). Two runs, one protocol change
between them.

| solver | context | tasks | solved | timed out (300 s) | s/task |
|---|---:|---:|---:|---:|---:|
| `qwen3:30b-a3b` (baseline row) | 32k | 40 | 6 (15.0%) | 0 | 85 |
| `gpt-oss:20b`, effort high | 32k (server default) | 34 of 40 | 6 (17.6%) | 0, but 23 truncated | 178 |
| **`gpt-oss:20b`, effort high** | **131k** (`--ollama-num-ctx`) | 40 | **9 (22.5%)** | 21 | 222 |

**Run 1 was bounded by the context window, not the model.** The ollama
server's default 32,768-token window stopped **23 of 34** answers at
`n_tokens = 32767, truncated = 1` (mean 27,582 tokens generated); every
truncated answer is empty, the model was still thinking. Of the 11 that
finished, 6 solved. The run also hit a two-hour GPU-window cap after
task 34 (`"partial": true` in the JSON).

**Run 2 raised the window to the model's 131,072-token limit** (fits: 16.5 GB
resident on the 4090), so the protocol's own bound, the 300 s deadline,
is the only one. Now **21 of 40** requests time out instead, and the
model finishes on 19; of those it solves 9. Solves are all at ≤16 ops
(witnesses of 5, 5, 5, 5, 7, 12, 12 ops, plus the 39-op shortcut) and one of the
nine is the `aligned_matches` sampler shortcut named in the baseline
report (`drop uncons swap drop nil swap cons`, head of the first list),
so **8/40** computes the behaviour. 0 memorised.

| witness ops | tasks | run 2 solved | run 2 timed out |
|---:|---:|---:|---:|
| ≤7 | 8 | 6 | 0 |
| 8–16 | 6 | 2 | 2 |
| 22–39 | 12 | 1 (the shortcut) | 8 |
| 40+ | 14 | 0 | 11 |

Same-task disagreement between the two gpt-oss runs goes both ways:
`sum_negative` and `negate_each` solved in run 1 and failed in run 2 with
operand-order and word-order slips (`filter [ 0 lt? ]`,
`map [ 0 swap sub ]`); `keep_zeros`, `n_minus_each` and `aligned_matches`
went the other way. One sample per task at temperature 1 is noisy at
this size, and 6/34, 9/40 and qwen3's 6/40 are inside each other's noise.

**Reading.** On v4 the free local arm is still search-equivalent: gpt-oss
at high effort lands where qwen3 and a 100,000-node BFS land, at 2.5x
qwen3's wall clock, and its failure mode is the clock, not the substrate,
which is a different mode from qwen3's programs-that-do-not-run. Nothing
here moves a local model off the floor. What would: a longer deadline for
gpt-oss (21 timeouts at 300 s, and the finished ones solve at 47%), or
several samples per task, both of which cost GPU-hours this project is
not spending on the local arm.

The qwen 2507 fragment (1 task) is a replicate check, not a result.

## Reproduce

```bash
R=reports/stack-pbe-v4-local-arm-20260902
# run 1: server-default 32k context, capped to a 2 h GPU window
OLLAMA_BASE_URL=http://localhost:11434 \
timeout -s TERM -k 60 5916 .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider ollama \
  --model gpt-oss:20b --ollama-think --effort high --tasks 40 --split eval \
  --out $R/gptoss20b_high_v4_sample40.json
# run 2: 131k context, uncapped
OLLAMA_BASE_URL=http://localhost:11434 \
.venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider ollama \
  --model gpt-oss:20b --ollama-think --effort high --ollama-num-ctx 131072 --tasks 40 --split eval \
  --out $R/gptoss20b_high_ctx131k_v4_sample40.json
# run 1's truncation count, from the ollama journal over its window
journalctl -u ollama --since '2026-09-03 00:08' --until '2026-09-03 01:51' --no-pager | grep 'stop processing'
```

`run_gptoss.sh` and `run_gptoss_ctx131k.sh` are the exact invocations;
the `.log` beside each JSON has the per-task lines and the `ReadTimeout`
lines that mark the deadline hits. `.venv/bin/python` rather than
`uv run`: uv swallows the SIGTERM and the child writes nothing. The
script writes a partial file on SIGTERM (added for this run, marked
`"partial": true`). gpt-oss:20b is ollama digest `17052f91a42e`, MXFP4;
ollama 0.33.2; one RTX 4090; 2026-09-03, run 1 00:08–01:50, run 2
04:06–07:12.
