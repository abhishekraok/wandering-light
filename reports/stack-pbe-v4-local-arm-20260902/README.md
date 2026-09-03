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

| solver | tasks | solved | s/task |
|---|---:|---:|---:|
| `qwen3:30b-a3b` (baseline row) | 40 | 6 (15.0%) | 85 |
| **`gpt-oss:20b`, effort high** | 34 of 40 | **6 (17.6%)** | 178 |

Effort is passed as ollama's `think` level (new `--ollama-think`, off by
default so earlier runs are unchanged). The cap landed after task 34;
tasks 35–40 are unrun.

**The number is the context window's, not the model's.** The ollama
server runs its default 32,768-token window, and the journal shows
**23 of 34** answers stopped at `n_tokens = 32767, truncated = 1` (mean
27,582 tokens generated per task). Every truncated answer is empty — the
model was still thinking. Of the 11 that finished:

| finished thinking | solved | witness lengths solved |
|---|---:|---|
| yes (11) | **6** | 5, 5, 6, 12, 12, 16 |
| no, truncated (23) | 0 | — |

qwen3 on the same sample solved nothing at 16 ops or more; gpt-oss solved
`sum_negative` (16) and `drop_last` (12, twice) when it finished. Two of
the untruncated failures are near misses (`sum_negative` as
`[ 0 swap lt? ] filter 0 [ swap add ] fold`, later solved with the
operands the right way round). Truncation hit 5-op tasks too (`keep_zeros`
once truncated, once solved), so it is not only the long witnesses.

Two readings, not separated here: gpt-oss at high effort needs a window
above 32k for this prompt (the model allows 131k, and the KV fits in
24 GB alongside 12 GB of weights), or a lower effort finishes inside 32k.
Either is a rerun of about two GPU-hours, not done. Until then 6/34 is a
floor: the qwen3 arm's equal on a harder clock, with a different failure
mode (out of context, not out of ideas).

## Reproduce

```bash
R=reports/stack-pbe-v4-local-arm-20260902
OLLAMA_BASE_URL=http://localhost:11434 \
timeout -s TERM -k 60 5916 .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider ollama \
  --model gpt-oss:20b --ollama-think --effort high --tasks 40 --split eval \
  --out $R/gptoss20b_high_v4_sample40.json
# truncation count, from the ollama journal over the run's window
journalctl -u ollama --since '2026-09-03 00:08' --until '2026-09-03 01:51' --no-pager | grep 'stop processing'
```

`run_gptoss.sh` is the exact invocation. `.venv/bin/python` rather than
`uv run`: uv swallows the SIGTERM and the child writes nothing. The
script writes a partial file on SIGTERM (added for this run, marked
`"partial": true`). gpt-oss:20b is ollama digest `17052f91a42e`, MXFP4;
ollama 0.33.2.
