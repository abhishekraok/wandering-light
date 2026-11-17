# GPT-6.2 Astra on the fixed v4 40-task sample

**High solves 27/40 at $0.029503 per task**, tying the best hosted score.
The previous 27/40 point, Claude Fable 5 medium, costs $0.139904 per task.
Astra reaches the same score at 79% lower cost. GPT-6 Sol max solves
26/40 at $0.021856. High adds four solves over the preceding GPT-6 Astra
high arm, with zero losses on the paired tasks and 28% lower cost.

| effort | solved /40 | memorised | illegal | usage calls | mean output tokens | $/task | server identity confirmed /40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 21 | 11 | 3 | 40 | 141.3 | 0.016054 | 40 |
| medium | 23 | 15 | 2 | 40 | 238.9 | 0.019957 | 40 |
| high | 27 | 12 | 1 | 40 | 477.6 | 0.029503 | 40 |

High is the only Astra 6.2 arm in the 24–27 band. Output rose from 239
mean tokens at medium to 478 at high, with cost rising 48%. High already
ties the best hosted score, so we skipped xhigh under the top-effort rule.
The owner's effort preference excluded max and ultra. The band compares
scores within three solves of the best on this sample.

## Protocol and audit

Run November 17, 2026, with Codex CLI 0.169.0 before and after each arm.
The previous catalog check ran on 0.165.0. The sweep used the CLI already
installed on the machine.

Same seed-17 sample of 40 `stack_pbe_v4` eval tasks as the preceding
GPT-6 Astra and GPT-6.2 Sol. The preceding Astra arms use the same
40 indices, and all 120 Sol candidates regrade with the same results.
The harness, CLI adapter, capture, grader and corpus manifest match the
Sol run. Each task gets one `codex exec` invocation and one saved candidate
using `experiments/stack_pbe_baseline.py`. Tools are disabled. A separate
low-effort preflight before each arm confirmed `gpt-6.2-astra` and is
excluded from evaluation. [Preceding Astra sweep](../stack-pbe-v4-codex-exec-20260910/README.md)

All 120 server identities, parsed candidates, grades and token usages
agree. The residual CLI schema prompted one `exec` request at low, one at
medium and two at high. Each wire result says `code-mode host is disabled`.
No tool executed. The model continued in those same CLI turns and the cost
includes all recorded output. All 120 CLI turns completed and the 120
saved candidates have no generation exceptions.
[Compact CLI audit](cli-audit.json) · [Validation](validation.json)

The plot uses 1,300 uncached input tokens plus all recorded output tokens
at $8 input and $40 output per million for standard short contexts. The
subscription CLI inputs also include instructions and tool schema. Pricing
the recorded token mix, including cached input at $0.40 per million, gives
$1.770766, $2.161814 and $2.658301 total for low, medium and high. The
corresponding plot totals are $0.642160, $0.798280 and $1.180120. Prices
verified November 17, 2026. [OpenAI list prices](https://developers.openai.com/api/docs/pricing)

| effort | started PST | finished PST | clean inference commit |
|---|---|---|---|
| low | 14:34:34 | 14:44:10 | `ac84297` |
| medium | 14:53:14 | 15:05:21 | `ac84297` |
| high | 15:08:42 | 15:27:28 | `ac84297` |

Each arm exited 0 on its first run, with a 12 GiB cgroup and zero swap.
The sidecars retain exact revisions, source and artifact hashes, CLI versions
and memory scope at inference time.

## Reproduce

From the repository root, after syncing dependencies and fetching the
manifest-pinned corpus as in the main README, replay the committed results.
The compact audit contains the evidence for offline validation.

```bash
REPORT=reports/stack-pbe-v4-astra62-20261117
PLOT=reports/stack-pbe-v4-effort-sweep-20260927
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. .venv/bin/python "$REPORT/analyse.py" > "$REPORT/table.md"
.venv/bin/python "$PLOT/analyse.py" > "$PLOT/table.md"
.venv/bin/python "$PLOT/plot.py"
```

To run the benchmark again using the Codex subscription, choose a fresh
output directory. The runner makes and checks the low preflight for each
arm. The extractor takes the CLI commands, stdout, stderr hash and selected
wire evidence from the ignored `raw/` directory.

```bash
export REPORT=reports/stack-pbe-v4-astra62-20261117
export OUT=/absolute/fresh-output/stack-pbe-v4-astra62-20261117
for effort in low medium high; do
  bash "$REPORT/run.sh" "$effort"
done
.venv/bin/python "$REPORT/extract_audit.py" --data-dir "$OUT" --out "$OUT/cli-audit.json"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. .venv/bin/python "$REPORT/analyse.py" --data-dir "$OUT" > "$OUT/table.md"
```
