# Two saved runs on the same forty v4 tasks, paired and regraded

Date: 2026-09-13 · corpus `stack_pbe_v4` (Hub revision `00c2a999`), eval split,
seed-17 40-task sample · **no new inference**: both candidates per task come
from the saved JSONs, regraded per example with `wandering_light/stack_pbe_eval.py`.

| run | file | solved | memorised | failed (wrong / refused) | shown-only |
|---|---|---:|---:|---:|---:|
| `gpt-5.6-luna` high, OpenAI API, 2026-09-01 | `../stack-pbe-v4-baseline-20260901/luna_v4_sample40.json` | 15 | 3 | 22 (4 / 18) | 45.0% |
| `gpt-6-astra` high, `codex exec`, 2026-09-10 | `../stack-pbe-v4-codex-exec-20260910/astra_high_v4_sample40.json` | 23 | 15 | 2 (1 / 1) | 95.0% |

Interactive page: [`index.html`](index.html) (standalone; step through every
candidate op by op), also published at
<https://claude.ai/code/artifact/a1f97da6-9478-49bd-93ec-2df14683fa1f>.
Paired data with per-example verdicts and per-op traces: `paired.json.gz`.
Built by `experiments/stack_pbe_compare_runs.py` and `build_page.py`.

## Alignment, verified rather than assumed

Both files hold candidates in sample order. The luna file (older schema) has no
`task_index`, corpus, model, provider, effort or generation fields; it was
joined by re-drawing the seed-17 sample the way `stack_pbe_baseline.py` draws
it, and its behaviour sequence and witness sequence agree with the corpus on
all 40. The astra file carries `task_index` (agrees on all 40) and omits the
witness by design. Regrading every saved candidate with the current grader
reproduces every saved verdict, 40/40 in both files.

## Paired verdicts (rows luna, columns astra)

| | solved | memorised | failed |
|---|---:|---:|---:|
| solved | 15 | 0 | 0 |
| memorised | 0 | 3 | 0 |
| failed | **8** | **12** | 2 |

The 20 disagreements run one way: luna failed on all 20, and on 17 of them the
machine refused an op before the program finished. Where both memorise (#9
`adjacent_sign_change_count`, #14 `axis_point_count`) the two programs are
byte-identical; both "solve" #30 `aligned_matches` with the identical sampler
shortcut the baseline report flagged.

## Five cases

| # | task | luna | astra | what it shows |
|---|---|---|---|---|
| 1 | #25 `keep_below` (6-op witness) | refused: `swap` inside the filter body finds one value | solved: `quote [ lt? ] cat filter` | closure capture; a body sees only its element |
| 2 | #38 `prefix_sums` (25) | refused: fold body leaves 2 values | solved: list accumulator with the sum at its head, 18 tokens | the one-value fold contract |
| 3 | #3 `drop_last` (12) | refused at the first `fold` | memorised: `x > -3` threshold filter fits all 3 shown | shown draws can underdetermine the behaviour; on #32 astra wrote the witness |
| 4 | #8 `positive_majority_three` (39) | refused: folded three scalars as a list | memorised: "sum > 0" proxy, right on 12 of 13 | one-bit outputs make 8 held-out pairs a thin net |
| 5 | #5 `widest_pair` (84) | wrong: lookup table on list length, mistranscribes 11 as 10 | memorised: `3 × count(non-negative) + 2` | both hard-code; astra's fit passes a shown-only grader |

## What the examples suggest about failure modes

- **Luna loses the stack, not the algorithm.** Its plans are often right (case
  2) and its bookkeeping is not: bodies that expect outer-stack values, folds
  carrying two accumulators as two items, an input shape misread. 18 of 22
  failures are refusals. When it hard-codes it hard-codes badly.
- **Astra always fits, sometimes by any means.** 38/40 candidates run and fit
  the shown pairs; 15 are memorisations: threshold filters, sign proxies,
  affine functions of a count, arithmetic on the bounds that never reads the
  list (#28). Where it knows the idiom it writes short correct programs. The
  held-out set does all the grading: overstatement 37.5 points.
- **Corpus side.** Memorisation clusters on flag- and count-valued behaviours;
  some shown draws admit a simpler hypothesis (#3 vs #32); where both models
  memorise they converge on the same proxy, so the proxy is a property of the
  examples.

## What future runs should capture

1. `task_index`, corpus name, Hub revision and payload digest in every row.
2. A run header: model, provider, effort, prompt version, temperature and
   sampling parameters, timestamp, git SHA. The luna file has none of these.
3. Per-example verdicts: pass / wrong / refused per shown and held-out pair,
   the produced stack, and for a refusal the op index and the machine's reason.
4. The raw model text, not only the parsed candidate.
5. Generation cost (tokens, seconds) on API rows as on codex rows.
6. k ≥ 2 samples per task, or a sampling seed, so flips can be told from noise.
7. An output-shape tag per behaviour (flag / count / scalar / list), and a
   held-out count sized to it (already filed on the codex-exec issue).
8. A schema version on the run JSON.

## Reproduce

```bash
uv run python -c "from wandering_light.corpus_hub import fetch_corpus; fetch_corpus('wandering_light/evals/data/stack_pbe_v4/manifest.json')"
R=reports/stack-pbe-v4-run-comparison-20260913
uv run python experiments/stack_pbe_compare_runs.py \
  --a reports/stack-pbe-v4-baseline-20260901/luna_v4_sample40.json --label-a luna \
  --b reports/stack-pbe-v4-codex-exec-20260910/astra_high_v4_sample40.json --label-b astra \
  --out $R/paired.json
gzip -9 $R/paired.json               # the tracked copy is paired.json.gz
uv run python $R/build_page.py       # reads paired.json.gz, writes index.html
```

Caveats carried from the source reports: one sample per task at default
temperature (luna API vs luna via `codex exec` agree on 33/40); the astra arm
ran with the owner's `~/.codex` page in context; the sample is easier than
the split; 180 of 220 behaviours are `gpt-5.6-sol`-written, so this is corpus
calibration, not a model ranking.
