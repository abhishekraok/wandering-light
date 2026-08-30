# Status

**This file records state. Issues record argument. Never both.**

So they cannot contradict each other: if you want to know *what is true*, read
here; if you want to know *why*, follow the issue link. A claim that is not in
this file is not established, however confidently some comment states it.

Update this file as part of finishing work, not as a separate chore.

Last updated: 2026-08-28

## What is true now

Every row is merged to `main` and measured. The number is the point; the issue
holds the argument.

| | | issue |
|---|---|---|
| **Stack machine** | Named bases are `executable-26` (search default), `full-52` (legacy program/CLI default), and `mark-28` (corpus default since v3). Guards return a reason or `None`; illegal is an *absent transition*, not an exception. | [replace-typedlist-states (#42)](../../issues/42), [solver-build-quotations (#60)](../../issues/60) |
| **Identity is structural** | `value_key` builds nested tuples of str tags, ints and bools. No hash, no repr, no dict. Keys are byte-identical across `PYTHONHASHSEED`, pinned by a test running three seeds in subprocesses. | #42 |
| **Fuel** | Every first-order op costs exactly 1, so `fuel == len(program)` there. Combinators break it: `map` costs `1 + body x len(input)`. Totality rests on the guard, not on structure. | #42 |
| **Branching, stack machine** | first-order 18: `b_eff` 11.14, 3.464 bits/step. The bounded published protocol reproduces executable 26 at 13.38 / 3.742 / 7.3x growth, full 52 at 37.33 / 5.222 / 32.4x, and mark 28 at 23.11 / 4.530 / 22.5x. The older executable/full measurements were 13.32 / 37.28. | [replace-typedlist-states (#42)](../../issues/42), [solver-build-quotations (#60)](../../issues/60) |
| **`mark-28` reproduction** | Every requested measurement from solver-build-quotations (#60) reproduces: even-square sum 25 ops / fuel 106, its predicate 14 ops, recursive sum 33 ops / fuel 129. On the one-list frontier, quote-closed states are 721 / 1,842 at depth 3 and 6,298 / 37,312 at depth 4. This is implementation evidence, not an adoption result. | [report](../reports/mark-28-20260827/README.md), [solver-build-quotations (#60)](../../issues/60) |
| **`mark-28` corpus probe** | solver-build-quotations (#60) step 2, the gate on adopting it. **Not met.** Like-for-like at depth 5: combinator share moves **6.7% → 7.0%**, tasks running code at all do not move (12.4% both), for **10.5x the generation cost** (130 s → 1,364 s). At depths 2–4 the two corpora are *identical*. Quotation construction appears (0% → 4.3%) but is degenerate: bodies are 1–2 ops and **59 of 61 never execute what they built**. Cause: tasks are certified *shortest* and the families hand quotations over, so a built quotation can never beat a supplied one. **The lever is the generator, not the op set.** | [report](../reports/mark-28-corpus-20260828/README.md), [solver-build-quotations (#60)](../../issues/60) |
| **Stack corpus, v3** | `stack_pbe_v3.jsonl.gz` — 1,600 PBE tasks over **40 hand-written behaviours**, 1,160 train / 440 eval, **split on the behaviour** so none spans both halves. Generated behaviour-first: a person writes the problem and a witness, and the machine is asked to meet it. Inputs are numbers and lists of numbers drawn fresh per example — **15,550 distinct input stacks** against v2's 17, and **0%** of task inputs contain a quotation against v2's 36.6%. **87.5%** of witnesses build a quotation and use a combinator, against v2's 0.0% and 5.7%. Every task carries **8 hidden held-out examples**, so a candidate is graded on data it was never shown. Witness lengths 1–35, median 6.5. Witnesses are **authored, not certified shortest**; each is cross-checked against an independently written Python reference on 500 fresh inputs. **No solver has been run on it.** | [report](../reports/stack-pbe-v3-20260828/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **`mark-28` is forced once inputs carry no quotations** | Structural, and measured. No `executable-26` op puts an op *name* on the stack, so from a data-only root every reachable quotation is data — which `map` can only apply as the identity and which `filter` and `fold` reject. Depth-5 ball around four data-only roots: `executable-26` reaches 6,058 quotations, **0 executable**, and makes **33,468 legal applications** of `i`/`dip`/`map`/`filter`/`fold`/`ifte` of which **0 execute any op**; `mark-28` reaches 78,731 executable quotations and 2,289 working applications. This does not overturn #60's adoption gate, which measured the *harvesting* generator and whose own conclusion was "the lever is the generator, not the op set". | [report](../reports/stack-pbe-v3-20260828/README.md), [solver-build-quotations (#60)](../../issues/60) |
| **v3 tasks are not under-determined** | Exhaustive `mark-28` sweeps from each task's input stacks, with every fit re-checked on that task's held-out examples. 40-task sample, fixed seed: at depth 4 **1/40** tasks admit a fit, at depth 5 **7/40** — and at both depths **0** admit a program that fits the shown examples while failing the held-out ones. **0** have a fit shorter than the authored witness, and every fit found *is* that witness. All seven are the 1–5 op floor (`sum`, `product`, `tail`, `double_each`, `square_each`, `keep_negative`); nothing with a 6+ op witness is reached. So 28 minutes of exhaustive `mark-28` search buys the easiest 17.5% of the sample and nothing else — BFS is used here as an auditor, not a solver. `experiments/stack_pbe_search_audit.py`. | [report](../reports/stack-pbe-v3-20260828/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **BFS on `stack_pbe_v3`** | First solver numbers, and the first in this project graded on data the solver never saw. A backend sees only a task's shown examples; `wandering_light/stack_pbe_eval.py` grades what it returns on the 8 held-out ones. Forty eval tasks at three node budgets: **0/40 at 1,000**, **3/40 (7.5%) at 10,000**, **11/40 (27.5%) at 100,000** — 10x the budget for 10x the wall clock and ~3x the solves. At 100,000 BFS gets **every** 5-op behaviour in the sample and essentially nothing at 6+ ops (`maximum` 0/6, `abs_each` 0/4, `minimum` 0/2, `sum_both` 0/7): the same step function `deep-corpus-v1` shows, with the wall between 5 and 6 ops. A shown-only grader would have reported **+2.5 points** at every budget, from one task where `uncons drop i` returns the last element of a *two-element* list by accident. **No language model has been run**: no credentials on the machine. | [report](../reports/stack-pbe-baseline-20260829/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **A language model on `stack_pbe_v3`** | `gpt-5.6-terra` at reasoning effort high, one sample per task, whole eval split: **377/440 (85.7%)** against BFS's 39/440 (8.9%) at a 10,000-node budget and 11/40 (27.5%) at 100,000 on a sample. It does not see BFS's 5-to-6-op wall — `abs_each` 34/40 and `minimum` 29/40 are 16-op programs with a nested `ifte` inside a combinator body, roughly 10^20 nodes of search. **Its shown-only number is 2.7 points high against BFS's 0.7** on the same 440 tasks: the stronger solver overstates ~4x as much, so holding examples back is worth more as solvers improve. All 12 memorisations land on `maximum` (6), `minimum` (4) and `last` (2) — the branch-bearing behaviours. A local `qwen3:30b-a3b` scores 25.0% on the 40-task sample, level with a 100,000-node search and with the same early wall. 85.7% says the forty behaviours are **too easy to carry an LLM benchmark**, which is the number that should aim the next catalogue. | [report](../reports/stack-pbe-baseline-20260829/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **Branching, `wl-core-v1`** | 118 functions, 20.52 applicable, 4.292 bits/step. Type masking is worth ~3.9 bits, slightly better than #33 assumed. | #33 |
| **Substrate comparison** | Both bases generate new behaviour at ~the same rate under composition — 5.87x/op against 5.77x. The stack machine's win is price: 73% of choice bits become new behaviour against 59%. It pays a **median 2.00x in program length**. | #42 |
| **`deep-corpus-v1`** | 31,350 tasks, certified distances 1–8, 240 roots, split by root. BFS solve rate is a **step function**: 100% up to its depth cap, 0% beyond. Distance-8 answers cost it ~4.4M executions. | [#37](../../issues/37) |
| **`deep-corpus-v2`** | 313,438 fresh certified tasks from 2,400 roots, split 1,880 / 260 / 260 with zero root/task overlap across splits or with v1. All witnesses re-execute. The tracked 42 KB manifest carries corpus identity and pins per-root telemetry published separately at HF revision `36753a013f3a4e965892b33bc44a8cc0fd567d99`. | [#54](../../issues/54) |
| **Basis growth by MDL** | Degenerate. Accepted 400 words without converging, basis 118 → 518, corpus −36.4%, **search 42,631x more expensive at depth 6**. Every candidate clearing the bar is length 2. | [#41](../../issues/41) |
| **Policy probe, first action** | A lookup table on `(input type, output type)` reaches mean rank 3.18 and top-1 42.1%. Two byte-level encoders beat it by 10% on rank and tie on top-1. | [#33](../../issues/33) |
| **Policy probe, last action** | Learning pays here and barely does at the front: lookup 4.80 / 22.3%, conv encoder **3.43 / 42.1%** — 29% better rank, nearly double top-1. Masking is weaker at the back (25.88 functions produce a given output type against 9.87 consuming a given input type). Neither end degrades with distance. | #33 |
| **Value probe, distance-to-go** | Labels are free and exact: every witness prefix is an exactly labelled `(state, target, remaining)` point, 137,140 of them. A conv encoder reaches **MAE 0.978 steps** against a type-pair dict's 1.232 — learning pays 21% — but is off by **1.66 / 2.08 / 2.51 at remaining distance 6 / 7 / 8**, against a pre-stated bar of ~1. Train MAE 0.830 says representation-limited, not undertrained. As an expansion orderer `h` scores mean rank 3.35, *worse* than the policy lookup table's 3.18. | [#33](../../issues/33) |
| **Policy probe, inward** | The front/back asymmetry is a **first-step artifact**. `deep-corpus-v1` roots exclude `int`/`str`/`float`, so a task's input type is rare and masks to 9.50 of 118 functions; one op later states collapse onto `int`/`float`/`str` and the mask widens to 18.41, settling at 19.90. The back is flat at 25.87 → 26.87. So the dict matches the encoder at `k=0` only; at `k=1,2` on both ends the conv encoder beats it by 16–25% on rank and 23–88% on top-1. | [#33](../../issues/33) |
| **Policy probe, untruncated** | The published encoder cut `input + "\0" + output` at 256 bytes, overrunning on **46.0%** of tasks — and truncation eats the *target*, so it starved only the end that needs one. The identical model at 1024 bytes moves the back probes **+1.06–1.46 mean rank and +14.6–17.2 top-1** and the front probes not measurably (+0.03–0.06, inside seed spread). Encoding a task as its 3–16 **aligned example pairs** beats the widened encoder at all six probes by **+0.25–0.61 rank**, including the two where nothing was truncated; ablation puts 92–109% of that on the cross-item pooling and nothing on the type tag, at **90,374 parameters against the published 90,582**. Bits saved over the type mask: front **1.29–1.79**, back **3.08–3.25** — at d=7 that is ~500–5,900x against ~7.1M x. Best back is **1.79 rank / 71.0% top-1 / 1.35 bits/step**, the strongest policy number in the project. Witness labels, 3 seeds, paired. | [#33](../../issues/33) |
| **Learned solver, run 1** | Pretrained `facebook/opt-125m`, 3-epoch SFT: 14/96 at k=64 on held-out. Cheaper than BFS at distances 3–4; **0/12 at every distance 5–8**. | [#50](../../issues/50) |
| **Learned solver, run 2** | Pretrained `HuggingFaceTB/SmolLM2-360M`, same corpus and locked 3-epoch protocol: 4/96 at k=64, no cost win at any distance, and **0/12 at every distance 5–8**. A stronger pretrained base did not move the target number. | [#50](../../issues/50) |
| **Why run 2 regressed** | Measured from the saved generations, not inferred: **42.4% of SmolLM2 candidates are unparseable against opt-125m's 0.4%**, essentially all invented function names (`f_sqrt` for `f_abs_sqrt`, `str_to_int` for `int_to_str`, `list_sort`) because the prompt never states the 118-name vocabulary. But conditional on parsing it is still 3.5x worse (0.28% vs 0.98%), so vocabulary is not the whole gap. **Pretrained priors are a liability on an arbitrary closed vocabulary** — which is why #54 tests random initialisation rather than a better pretrained base. | #50 |
| **Scratch solver feasibility** | On one RTX 4090, the exact 10M and OPT-style 125M arms project to 2.258 h and 3.947 h for three pilots plus final. The required-checkpointed 360M arm fits VRAM but projects to **17.282 h**, above the 16 h hard limit. The three-arm study stopped before pilots, final training, generation evaluation, or test opening. | [#54](../../issues/54) |
| **Undirected expansion drains, `wl-core-v1`** | 300 uniform-random walks of length 40 from each of six root types land on **43–56 distinct endpoint states**, with a near-identical endpoint type mix whatever the walk started from. Against a depth-6 ball of 233,585 states, an 80-step walk lands **within 6 of the root 99% of the time, modal true distance 2** — the same distribution a 4-step walk gives. Mechanism: 37 of 118 ops output `int` and 27 output `bool`, and applicable ops span 3 (`bytearray`) to 29 (`str`), so motion runs downhill into low-out-degree scalar types. The modal endpoint `TL<int>([0,0,0,0,0])` has a depth-4 ball of 297 states against the root's 5,559. `experiments/walk_collapse.py`. | [#55](../../issues/55) |
| **Undirected expansion diffuses, `executable-26`** | The stack machine has the opposite pathology: **300/300 distinct endpoints** at walk length 40 and 80, every walk escapes the depth-5 ball (median first escape step 10–11), endpoint out-degree converges to ~12.9 = `b_eff`. Literals are always legal, giving ~0.36 items/step of stack growth; endpoint depth ~15 at length 40, ~29 at 80. Measured by the `reusing-solved-trajectories` session in a **separate implementation not in this repo**, sanity-checked against #41's pinned 9,503-state depth-5 empty-root ball. Not independently reproduced here. | [#60](../../issues/60) |
| **What that licenses** | Both substrates defeat *undirected* expansion past the BFS horizon, for opposite reasons — one cannot leave home, the other cannot come home. Both measurements use a uniform-random walk policy, so they bound undirected expansion only. **Neither shows that any directed mechanism reaches further**; that is assumed by the macro-op and learned-policy programmes, not established. | [#55](../../issues/55) |

## In flight

| what | who | where | question it answers |
|---|---|---|---|
| Theory discussion | `stack-machine-vs-register` session | [#52](../../issues/52) | — |

## Superseded

Claims that were stated confidently and are wrong. Recorded here so a
correction lands in one place instead of being chased across comment threads.

- **The committed stack corpus, `stack_pbe_v2.jsonl.gz`, is deleted** as of
  2026-08-28, along with `stack_corpus_scope.py`,
  `recertify_stack_machine_cost.py`, `stack_machine_baseline.py` and the tests
  that read it. It was 1,857 *programs* over **17 distinct input sets**, handed
  a quotation to the solver in 680 of them, and built none. Nothing derived from
  it is quotable, so keeping it invited accidental citation. Recover it with
  `git show 28e9518:wandering_light/evals/data/stack_pbe_v2.jsonl.gz` -- the
  history is intact, and `reports/stack-machine-basis/` and
  `reports/mark-28-corpus-20260828/` still describe what it contained. **The
  stack machine itself, its three bases, and every substrate measurement are
  unaffected** and stay. The replacement is `stack_pbe_v3.jsonl.gz`, generated
  behaviour-first; see the row above. Nothing measured on v2 carries over to
  it.

- **"BFS solves 75.2% of the stack corpus at a 10,000-node budget."** The
  number is real; what it measures is not what it was quoted for. That corpus
  builds no quotations, is 55.3% stack rearrangement, and draws on **17
  distinct input sets** across 1,857 tasks. So the figure describes search over
  an enumeration of programs from seventeen starting stacks, not PBE solve rate.
  Every stack-side solver number inherits this. Substrate measurements do
  **not** — branching, identity, fuel and the `wl-core-v1` cost comparison are
  measured by expanding from four fixed roots or from the ops themselves, and
  never read the corpus. See `reports/mark-28-corpus-20260828/` and
  [stack-corpus-no-higher-order (#59)](../../issues/59).

- **"Re-certifying the 101 loop tasks under Kt changed 0 labels, 0 bits."**
  Narrower than it sounds. `recertify_stack_machine_cost.py` selects
  `LOOP_FAMILIES = {with_map, with_fold, with_filter}` — three of the four
  families that **seed a quotation in the input**. So the tasks are loop-bearing
  because a quotation was handed to them, and they draw on about four input
  configurations. The mechanism argument survives (a label needs ~26x less fuel
  for one extra op, and four-element lists cannot vary that much); the evidence
  base does not support "101 tasks" as a general check. Re-run it on a corpus
  whose generator does not supply quotations before quoting it again.

- **"A richer basis will make the stack corpus more expressive."** The premise
  of adopting `mark-28`. Measured and false: at depth 5 the combinator share
  moves 6.7% -> 7.0% for 10.5x the generation cost, at depths 2-4 the corpora
  are *identical*, and 59 of the 61 quotation builders never execute what they
  built. Tasks are certified **shortest** and the families supply quotations, so
  a built quotation can never win — the generator selects against the
  construction the basis was extended to allow. **The lever is the generator,
  not the op set.** Scope note: this is a statement about the *harvesting*
  generator, and it does not license "`executable-26` is enough". Once the
  generator stops supplying quotations, `executable-26` reaches no executable
  quotation at all and `mark-28` becomes the floor rather than an upgrade —
  see the v3 rows above.

- **"Forward generation cannot produce problems that defeat forward search."**
  False. Generation pays `b^d` **once per root**, offline and amortised — 109
  minutes for all of `deep-corpus-v1`. Search pays it **once per task**. So a
  corpus can be generated at a depth per-task search cannot afford.
- **"BFS cannot solve 12.5% of `deep-corpus-v1`."** It is a depth cap, not a
  wall. The curve is a step function; the missed 12.5% is exactly the
  distance-8 tasks under a depth-7 search. What is true is that finding them
  costs a depth-8 sweep, ~6x the 60 s depth-7 failure.
- **"`wl-core-v1` is a near-trivial composition monoid"** (#42's premise).
  Measured, it generates new behaviour at 5.77x/op against the stack machine's
  5.87x. What the evidence supports is that its *corpus* was trivial.
- **"A lookup table on `(input type, output type)` is as good as a learned
  policy."** True only at the *first* action, and only because
  `deep-corpus-v1`'s roots exclude `int`/`str`/`float`, which makes the input
  type rare and the type mask unusually strong there. One op inward the mask
  widens from 9.50 to 18.41 of 118 and the encoder beats the dict by 16–25%
  on rank at both ends. Relatedly, **"type masking is worth ~3.9 bits"** is the
  first-step figure; the steady-state value is ~2.7 bits at the front and ~2.3
  at the back. See `reports/policy-probe-depth-20260825/`.
- **"The back is not uniquely learnable; it is *uniformly* learnable, and the
  front joins it one step in."** Measured with an encoder that could see the
  target, this is wrong. The published back numbers (top-1 41–43%) were held
  down by truncation, which starves the target end and so penalises only the
  probe that has to produce it. Un-truncated, the back reaches **60–71% top-1
  against the front's 32–48%** — a gap wider than the `k = 0` figure that was
  discounted as an artifact, not narrower. What survives is that the *mask-width*
  story is a first-step artifact; what does not is the inference that the two
  ends are comparably learnable. See `reports/policy-probe-untruncated-20260827/`.
- **"The autoregressive solver is finished on this substrate."** Overstated
  after run 1. A stronger pretrained SmolLM2-360M also remained 0/12 at every
  distance 5–8, which supports the bounded conclusion for this locked
  completion-only pretrained SFT protocol. It does not settle random-init
  training or other objectives; #54 is the scratch follow-up.
- **"Runs 1 and 2 used random initialization."** False. Both call
  `AutoModelForCausalLM.from_pretrained`; neither is scratch evidence.
- **"The feasible 10M and 125M scratch arms are pending while 360M is deferred
  separately."** False under #54's locked three-arm comparison: the 360M arm's
  failure of the predeclared hard gate stops all pilots and finals, so no
  partial two-arm result is pending.
- **"The TypedList eval harness is retired."** Reverted; nothing was deleted.
  `deep-corpus-v1` is the hard corpus and it lives on that harness.

## Where things live

- `wandering_light/stack_machine.py` — the machine
- `wandering_light/stack_machine_corpus.py` — PBE corpus generation, certified by BFS depth
- `docs/stack_machine_explorer.html` — keyboard-driven explorer, self-checks against the Python core
- `reports/` — every measurement, one directory per run
- `experiments/` — the scripts that produced them
