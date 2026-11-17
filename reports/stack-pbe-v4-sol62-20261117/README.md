# GPT-6.2 Sol on the fixed v4 40-task sample

**Low solves 24/40 at $0.0044875 per task**, the cheapest point in the
24–27 hosted band. The previous cheapest point in the band, GPT-6.1 Sol
medium, solves 25/40 at $0.0069755. The new low point costs 36% less and
solves one fewer task. High also solves 24/40 at 2.08× the cost of low.
The best hosted score remains 27/40. Differences of up to three solves
are within the plot's comparison band on this sample.

| effort | solved /40 | memorised | illegal | usage calls | mean output tokens | $/task | server identity confirmed /40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| low | 24 | 10 | 2 | 40 | 188.8 | 0.004488 | 40 |
| medium | 22 | 17 | 1 | 40 | 380.9 | 0.006409 | 40 |
| high | 24 | 14 | 2 | 40 | 674.8 | 0.009348 | 40 |

High reaches the band and costs more than low for the same score. The
GPT-6.1 Sol xhigh arm cost 37% more than its high arm on this sample, so
we skipped xhigh under the top-effort rule. The owner's effort preference
excluded max and ultra. [GPT-6.1 Sol comparison](../stack-pbe-v4-sol61-20260929/README.md)

## Protocol and audit

Run November 17, 2026, with Codex CLI 0.169.0 before and after each arm.
The previous catalog check ran on 0.165.0. Both GPT-6.2 models appear in
0.169.0, and this sweep used the CLI already installed on the machine.

Same seed-17 sample of 40 `stack_pbe_v4` eval tasks as GPT-6.1 Sol and
Haiku 5.5. All 40 indices match the predecessor's four saved arms and
its 160 grades reproduced before this sweep. The existing
`experiments/stack_pbe_baseline.py` provides the prompt, parser and
grader. Each task gets one `codex exec` invocation and one saved candidate.
The CLI runs with tools disabled. A separate low-effort identity preflight
before each arm confirmed `gpt-6.2-sol` and is excluded from evaluation.

All 120 calls confirm `gpt-6.2-sol` via server responses, and independent
replay matches each parsed candidate, grade and recorded usage. Each
CLI turn completed with zero generation exceptions. The CLI sends a
zero-output connection setup request. Its residual `exec` schema also
elicited four requests, two at medium and two at high. The outgoing
trace for each records `code-mode host is disabled`. The model continued
within those turns. Both responses count toward CLI usage, and their
output tokens enter the plot cost. The published audit records these rejected
requests separately from executed tools, of which there were zero.
[Compact CLI audit](cli-audit.json) · [Validation](validation.json)

The plot uses 1,300 uncached input tokens plus all recorded output tokens
at standard short-context list rates of $2 input and $10 output per
million. The subscription run's actual input includes CLI instructions
and tool schema. Pricing the recorded token mix instead gives $0.364564,
$0.477155 and $0.675158 in total for low, medium and high, respectively,
with cache reads at $0.10 per million. The plot equivalents are $0.179500,
$0.256340 and $0.373920. Price verified November 17, 2026.
[OpenAI list prices](https://developers.openai.com/api/docs/pricing)

| effort | started PST | finished PST | clean inference commit |
|---|---|---|---|
| low | 13:23:21 | 13:29:19 | `e483621` |
| medium | 13:37:51 | 13:52:30 | `e483621` |
| high | 13:56:32 | 14:10:32 | `dad0bc1` |

Each arm exited 0 on its first run, with a 12 GiB cgroup and zero swap.
The report's offline analysis changed between the inference commits.
Their source digests agree for the harness, CLI adapter, grader, stack
machine, manifest and observational capture script. Provenance sidecars
retain the full revisions and artifact hashes.

## Reproduce

From the repository root, after syncing dependencies and fetching the
manifest-pinned corpus as in the main README, replay the saved results.
This uses the committed compact audit and performs no model calls.

```bash
REPORT=reports/stack-pbe-v4-sol62-20261117
PLOT=reports/stack-pbe-v4-effort-sweep-20260927
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. .venv/bin/python "$REPORT/analyse.py" > "$REPORT/table.md"
.venv/bin/python "$PLOT/analyse.py" > "$PLOT/table.md"
.venv/bin/python "$PLOT/plot.py"
```

To run the benchmark again using the Codex subscription, use a fresh
output directory. The runner makes and checks the preflight for each arm.
Full traces are local under `raw/`. The extractor keeps the CLI commands,
stdout, stderr hashes and selected wire evidence in the compact audit.

```bash
export REPORT=reports/stack-pbe-v4-sol62-20261117
export OUT=/absolute/fresh-output/stack-pbe-v4-sol62-20261117
for effort in low medium high; do
  bash "$REPORT/run.sh" "$effort"
done
.venv/bin/python "$REPORT/extract_audit.py" --data-dir "$OUT" --out "$OUT/cli-audit.json"
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- \
  env PYTHONPATH=. .venv/bin/python "$REPORT/analyse.py" --data-dir "$OUT" > "$OUT/table.md"
```
