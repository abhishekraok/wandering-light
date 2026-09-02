# Status

**This file records state. Issues record argument. Never both.**

So they cannot contradict each other: if you want to know *what is true*, read
here; if you want to know *why*, follow the issue link. A claim that is not in
this file is not established, however confidently some comment states it.

Update this file as part of finishing work, not as a separate chore.

Last updated: 2026-08-31

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
| **Stack corpus, v3** | `stack_pbe_v3` (Hub, see the v4 row) — 1,600 PBE tasks over **40 hand-written behaviours**, 1,160 train / 440 eval, **split on the behaviour** so none spans both halves. Generated behaviour-first: a person writes the problem and a witness, and the machine is asked to meet it. Inputs are numbers and lists of numbers drawn fresh per example — **15,550 distinct input stacks** against v2's 17, and **0%** of task inputs contain a quotation against v2's 36.6%. **87.5%** of witnesses build a quotation and use a combinator, against v2's 0.0% and 5.7%. Every task carries **8 hidden held-out examples**, so a candidate is graded on data it was never shown. Witness lengths 1–35, median 6.5. Witnesses are **authored, not certified shortest**; each is cross-checked against an independently written Python reference on 500 fresh inputs. **No solver has been run on it.** | [report](../reports/stack-pbe-v3-20260828/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **`mark-28` is forced once inputs carry no quotations** | Structural, and measured. No `executable-26` op puts an op *name* on the stack, so from a data-only root every reachable quotation is data — which `map` can only apply as the identity and which `filter` and `fold` reject. Depth-5 ball around four data-only roots: `executable-26` reaches 6,058 quotations, **0 executable**, and makes **33,468 legal applications** of `i`/`dip`/`map`/`filter`/`fold`/`ifte` of which **0 execute any op**; `mark-28` reaches 78,731 executable quotations and 2,289 working applications. This does not overturn #60's adoption gate, which measured the *harvesting* generator and whose own conclusion was "the lever is the generator, not the op set". | [report](../reports/stack-pbe-v3-20260828/README.md), [solver-build-quotations (#60)](../../issues/60) |
| **v3 tasks are not under-determined** | Exhaustive `mark-28` sweeps from each task's input stacks, with every fit re-checked on that task's held-out examples. 40-task sample, fixed seed: at depth 4 **1/40** tasks admit a fit, at depth 5 **7/40** — and at both depths **0** admit a program that fits the shown examples while failing the held-out ones. **0** have a fit shorter than the authored witness, and every fit found *is* that witness. All seven are the 1–5 op floor (`sum`, `product`, `tail`, `double_each`, `square_each`, `keep_negative`); nothing with a 6+ op witness is reached. So 28 minutes of exhaustive `mark-28` search buys the easiest 17.5% of the sample and nothing else — BFS is used here as an auditor, not a solver. `experiments/stack_pbe_search_audit.py`. | [report](../reports/stack-pbe-v3-20260828/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **Guarding the open-quotation states** | Under an open `[`, `mark-28` captures every op as a symbol rather than applying it, so all 28 are legal where 13 were — the mechanism behind its `b_eff` of 23.11. `wandering_light/quotation_guard.py` reinstates a check at *search* level; the machine is unchanged, so no substrate measurement moves. **v3, 40 tasks, seed 17:** states per task fall **4.31x at depth 4** (41,917 → 9,730, 78 s → 39 s) and **9.21x at depth 5** (1,001,797 → 108,772, 1,851 s → 656 s), while per-task fits, held-out fits, spurious count and shortest fit stay **identical** to the unpruned run — still 1/40 and 7/40 tasks admitting a fit, still **0/40 spurious**. Reachability is established by replay rather than by the sweep, which only reaches 7 of 40 witnesses: **0 of 705** open-mark witness-prefix states pruned across v3's 40 behaviours and **0 of 26,013** across v4's 220. Reading the body as a program — the arm the issue argues for — strengthens with body *length*, not depth: alone it buys 1.55x / 2.15x / **2.96x** at depths 4/5/6, holding branching under a mark to **~20.8 against the machine's 28**, with **99.1% of length-16 bodies provably dead**. Bracket closability is a separate and larger arm (4.29x / 7.82x / 10.84x) that says nothing about whether a body makes sense, and is sound only because no v3 or v4 task carries a `MARK` in any input or output (0 of 1,600 and 0 of 8,800). **No solve rate here**: what a fixed *node* budget buys is [rerun-baselines-v4 (#85)](../../issues/85)'s measurement. | [report](../reports/mark-28-guard-20260901/README.md), [prune-open-mark-states (#70)](../../issues/70) |
| **Stack corpus, v4** | `stack_pbe_v4`: **8,800 tasks over 220 hand-written behaviours and 215 skeletons**, 7,000 train / 1,800 eval, **83,488 distinct input stacks** against v2's 17. Witness lengths 1–253 ops, median 43 against v3's 6.5; **76% carry a branch**. Every behaviour is its own inseparable group, so any split is possible; split is on the *skeleton*, and no behaviour or shape spans both halves. Shown examples may not hold list length fixed, which closed v3's most-repeated memorisation (`uncons drop i` on `last`: 1 of 40 → 0 of 40). Behaviours carry a tuple of witnesses, all verified. 180 of the 220 were written by `gpt-5.6-sol` via Codex against a brief, in two staged passes; every one is a literal entry checked against an independent Python reference on 500 fresh inputs, and an independent signature audit over 8,473 shape-compatible pairs found **zero** two behaviours computing the same function. Payload on the Hub: `abhishekraok/wandering-light-stack-pbe-v4`, revision `00c2a9996d32df8d1894b45f4246a66b6df46c25`. **No solver has been run on it.** | [report](../reports/stack-pbe-v3-20260828/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **Corpus inspection** | Hub viewers serve exactly the manifest-declared splits for stack PBE v3/v4 and deep corpora v1/v2; every split's row count matches its manifest. The v4 overview is a deterministic one-file report generated from SHA-verified pinned payloads, with corpus composition, split leakage, scaffolding and a filterable behaviour table. | [overview](../reports/stack-pbe-v4-20260830/index.html), [v3](https://huggingface.co/datasets/abhishekraok/wandering-light-stack-pbe-v3), [v4](https://huggingface.co/datasets/abhishekraok/wandering-light-stack-pbe-v4), [deep-v1](https://huggingface.co/datasets/abhishekraok/wandering-light-deep-corpus-v1), [deep-v2](https://huggingface.co/datasets/abhishekraok/wandering-light-deep-corpus-v2), [look-at-data (#78)](../../issues/78) |
| **Trajectory graph explorer** | The local FastAPI/Cytoscape app indexes corpora on disk, samples a task on startup, and explores bounded witness and compatible-function expansions without loading the whole corpus into memory. The canvas supports responsive pan/zoom, readable expanded layouts, path tracing, and selection inspection. | [trajectory-graph (#38)](../../issues/38) |
| **What the basis costs in shared code** | On `mark-28` a witness's median share of 8-op windows that *another* witness also contains is **79.8%** — 66.7% among the 40 behaviours written before this was measured, so it is the basis, not the catalogue. Twenty-eight ops, no local variables, every non-trivial program a fold around a hand-rolled accumulator. Eval witnesses share a median **76.8%** of their windows with train witnesses. This bounds what a held-out score on this corpus means: success is evidence of **composition**, not of invention. `experiments/stack_pbe_scope.py`. | [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **A wall in `mark-28`** | Five behaviours that resisted a first attempt do have witnesses — parameterized take/drop 50 ops, unequal-length interleave 53, ascending sort 118, stable sort over key/payload pairs 206, median 253. One does not: flat-point `bounding_box`, whose fold state needs a pair phase, a current coordinate and four extrema at once, and with no rotation, deep-pick or destructuring op every update becomes nested `cons`/`uncons` reconstruction. Recorded as a wall in this basis, **not** as proof of inexpressibility. Separately, stable sort on bare integers is *unobservable* rather than unwitnessable: with equal elements indistinguishable there is nothing an example can pin down. | [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **BFS on `stack_pbe_v3`** | First solver numbers, and the first in this project graded on data the solver never saw. A backend sees only a task's shown examples; `wandering_light/stack_pbe_eval.py` grades what it returns on the 8 held-out ones. Forty eval tasks at three node budgets: **0/40 at 1,000**, **3/40 (7.5%) at 10,000**, **11/40 (27.5%) at 100,000** — 10x the budget for 10x the wall clock and ~3x the solves. At 100,000 BFS gets **every** 5-op behaviour in the sample and essentially nothing at 6+ ops (`maximum` 0/6, `abs_each` 0/4, `minimum` 0/2, `sum_both` 0/7): the same step function `deep-corpus-v1` shows, with the wall between 5 and 6 ops. A shown-only grader would have reported **+2.5 points** at every budget, from one task where `uncons drop i` returns the last element of a *two-element* list by accident. **No language model has been run**: no credentials on the machine. | [report](../reports/stack-pbe-baseline-20260829/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **A language model on `stack_pbe_v3`** | `gpt-5.6-terra` at reasoning effort high, one sample per task, whole eval split: **377/440 (85.7%)** against BFS's 39/440 (8.9%) at a 10,000-node budget and 11/40 (27.5%) at 100,000 on a sample. It does not see BFS's 5-to-6-op wall — `abs_each` 34/40 and `minimum` 29/40 are 16-op programs with a nested `ifte` inside a combinator body, roughly 10^20 nodes of search. **Its shown-only number is 2.7 points high against BFS's 0.7** on the same 440 tasks: the stronger solver overstates ~4x as much, so holding examples back is worth more as solvers improve. All 12 memorisations land on `maximum` (6), `minimum` (4) and `last` (2) — the branch-bearing behaviours. A local `qwen3:30b-a3b` scores 25.0% on the 40-task sample, level with a 100,000-node search and with the same early wall. 85.7% says the forty behaviours are **too easy to carry an LLM benchmark**, which is the number that should aim the next catalogue. | [report](../reports/stack-pbe-baseline-20260829/README.md), [pbe-generator-behaviour-first (#67)](../../issues/67) |
| **The best free solver arm is search-equivalent** | Four local models on one RTX 4090 against the same forty v3 eval tasks. `qwen3:30b-a3b` (MoE, ~3B active) wins at **22.5%**, which sits on top of a **100,000-node BFS's 27.5%** and far below `gpt-5.6-terra`'s 82.5%. `gpt-oss:20b` and `qwen3-coder:30b` are worse; the dense `qwen3:32b` is unusable at **5.1 tok/s** because 20 GB of weights plus a 32k KV cache exceeds 24 GB of VRAM — on this card MoE is the only workable shape. So a free regression arm exists, and it measures roughly what search already measures. Separately: bound a local generation by **wall clock, not tokens** — a 4,096-token cap drops the same model to **5.0%** by truncating it mid-thought, while a 300 s deadline costs one task of forty at the same wall clock. | [report](../reports/stack-pbe-baseline-20260829/README.md) |
| **Branching, `wl-core-v1`** | 118 functions, 20.52 applicable, 4.292 bits/step. Type masking is worth ~3.9 bits, slightly better than #33 assumed. | #33 |
| **Comparable solver baselines** | One frozen cohort is shared across solvers and random seeds, and every ground-truth spec is validated before sampling; invalid specs fail with their source indices. The primary comparable cost is `wall_clock_seconds` — the timed solve call, the only unit that charges model inference — recorded with the evaluation host, and never comparable across hosts (seed aggregation refuses mixed hosts). `basis_function_executions` stays as the hardware-independent diagnostic: one per compatible BFS/rollout op tried, or the full length of each parse-valid random/LLM candidate. Solver-native budget units stay explicit. Random search samples uniformly from declared path lengths with per-task seeded RNG; its success-rate and execution-cost CIs bootstrap whole seed results. Deterministic/unseeded solvers report no seed CI. Ordinary `pytest` excludes the two `network` model-download tests. | [comparable-solver-baselines (#25)](../../issues/25) |
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
| **Materialized-state rollout** | A 90,374-parameter aligned-pair policy trained on every discovery witness prefix and selected beam 64 on validation. On the exact held-out cohort from [fine-tune-llm (#50)](../../issues/50), it solves **60/96 overall and 12/48 at d5–8**: 7/12 at d5, 5/12 at d6, then **0/12 at d7 and d8**. At d5–8 it averages 6,238 executions; BFS under the same mean capped spend solves 1/48 and needs **10.74x** the total executions to reach 12/48. In wall-clock — which charges the policy's inference, execution counting does not — the matched-solve advantage is **~4.5x**: rollout runs at ~13.0k executions/s including scoring (48 x 0.4797 s ≈ 23.0 s, `test/summary.json` beam-64 d5–8 `mean_seconds`) against BFS's ~30.7k executions/s (3,214,899 / 30,733 ≈ 104.6 s, `bfs_reference.json` totals; cross-run timing on the same host). The recorded d7–8 witness survives in 24/24 beams after depth 1, 2/24 after depth 5, and 0/24 after depth 6: materialization beats one-shot, while the remaining horizon is consistent with exposure bias. | [report](../reports/materialized-rollout-20260829/README.md), [solve-step-by-step (#55)](../../issues/55) |
| **Why run 2 regressed** | Measured from the saved generations, not inferred: **42.4% of SmolLM2 candidates are unparseable against opt-125m's 0.4%**, essentially all invented function names (`f_sqrt` for `f_abs_sqrt`, `str_to_int` for `int_to_str`, `list_sort`) because the prompt never states the 118-name vocabulary. But conditional on parsing it is still 3.5x worse (0.28% vs 0.98%), so vocabulary is not the whole gap. **Pretrained priors are a liability on an arbitrary closed vocabulary** — which is why #54 tests random initialisation rather than a better pretrained base. | #50 |
| **Scratch solver follow-up, `deep-corpus-v2`** | The two feasible arms **trained**, and the study still has **no result**. 10M and OPT-style 125M each completed three LR pilots and a three-epoch final, both selecting `3e-4`; final loss 0.755 and 0.686, token accuracy 0.752 and 0.761, 92,085 optimizer steps each. Eight W&B runs preserve **5.630 h** of telemetry (trainer-reported 4.860 h). Then it was lost: weights-only checkpoints went to staging under `/tmp`, `push_to_hub=false`, `hub_model_id=null`, and **neither final run has a model artifact**, so the weights are gone with the worktree. Validation never sealed — one distance-8 BFS task hit the fixed 20M ceiling and the validator refused. **Frozen-test access stayed at 0.** So there is no solve rate at any distance, no parse diagnostics, no capacity comparison and no scaling conclusion. The 360M arm remained deferred at **17.333 h** against the frozen 16 h cap. Per owner instruction the lost runs will not be retrained; resuming needs recreated weights and a freshly frozen protocol. | [report](../reports/deep-corpus-v2-scratch-20260827-incomplete/README.md), [build-deep-corpus-v2 (#54)](../../issues/54) |
| **Undirected expansion drains, `wl-core-v1`** | 300 uniform-random walks of length 40 from each of six root types land on **43–56 distinct endpoint states**, with a near-identical endpoint type mix whatever the walk started from. Against a depth-6 ball of 233,585 states, an 80-step walk lands **within 6 of the root 99% of the time, modal true distance 2** — the same distribution a 4-step walk gives. Mechanism: 37 of 118 ops output `int` and 27 output `bool`, and applicable ops span 3 (`bytearray`) to 29 (`str`), so motion runs downhill into low-out-degree scalar types. The modal endpoint `TL<int>([0,0,0,0,0])` has a depth-4 ball of 297 states against the root's 5,559. `experiments/walk_collapse.py`. | [#55](../../issues/55) |
| **Undirected expansion diffuses, `executable-26`** | The stack machine has the opposite pathology: **300/300 distinct endpoints** at walk length 40 and 80, every walk escapes the depth-5 ball (median first escape step 10–11), endpoint out-degree converges to ~12.9 = `b_eff`. Literals are always legal, giving ~0.36 items/step of stack growth; endpoint depth ~15 at length 40, ~29 at 80. Measured by the `reusing-solved-trajectories` session in a **separate implementation not in this repo**, sanity-checked against #41's pinned 9,503-state depth-5 empty-root ball. Not independently reproduced here. | [#60](../../issues/60) |
| **What that licenses** | Both substrates defeat *undirected* expansion past the BFS horizon, for opposite reasons — one cannot leave home, the other cannot come home. Both measurements use a uniform-random walk policy, so they bound undirected expansion only. **Neither shows that any directed mechanism reaches further**; that is assumed by the macro-op and learned-policy programmes, not established. | [#55](../../issues/55) |
| **Rollout debugger** | `wandering_light/rollout_debugger/`, keyboard-driven, serving the `deep-corpus-v1` test split against the private issue-55 checkpoint (Hub revision `009cb81`). It shows the **production** search: `RolloutPredictor.search_with_cost` takes an optional `RolloutObserver` that is reported to and never consulted, so an observed run and an unobserved one are the same run — asserted directly, and again against the real checkpoint under `-m network`. The encoding panel calls the policy's own `pair_batch`; nothing is reimplemented. A session is eager and ships whole (**456 KiB gzipped** at beam 64, depth 8), so stepping is a client-side index move; raw logit rows are excluded by default and fetched per node on demand (3–5 ms). Drive mode applies the policy's rank-1..9 compatible actions from `asdfghjkl` with `w` to undo, on the loaded task only. Two 2D views over the policy's own pooled 192-d `(state, target)` encoding: replay uses PCA anchored at φ(target, target); drive uses a fixed input→target axis, where x is fraction of the way to solved (PCA would move points as the fit grows). The page is driven in headless Chromium by `tests/test_rollout_debugger_browser.py`. | [interactive-rollout-debugger (#72)](../../issues/72) |
| **Execution and inference cost** | The basis function is **~15%** of a charged `basis_function_executions` unit (4.1 us of 27 us); the rest is `search_key()` and search bookkeeping. Policy scoring at beam 64 scales ~linearly on CPU (~80x from 90K to 125M params) and is **flat on CUDA** (8.2-10.3 ms across the same range), so the device crossover sits between **~1M and ~10M parameters** and inference latency does not bound model size on GPU. The published rollout run used `device=cuda` at 90,374 params, on the CPU side of that crossover. Load-sensitive; reproduce with `experiments/inference_cost.py`. | [report](../reports/inference-cost-20260830/README.md), [compute-optimal-policy-size (#80)](../../issues/80) |

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
- **"The 360M arm's failed gate stops the feasible 10M and 125M arms."**
  Superseded by a per-arm launch instruction on
  [build-deep-corpus-v2 (#54)](../../issues/54). Both feasible arms went on to
  complete their pilots and finals. Note this entry previously recorded the
  *opposite* correction; the three-arm lock was real when written and was then
  lifted.
- **"Completed 10M and 125M training means the scratch comparison is
  complete."** False, and it is the one that matters. The weights no longer
  exist locally or as W&B artifacts, validation never sealed, and the frozen
  test was never opened, so no solve rate exists at any distance. Training
  running is not the same as a result surviving.
- **"The deferred 360M arm fits by getting its own night."** False under the
  frozen protocol: its measured 17.333 h projection exceeds the hard 16 h cap.
  It needs a newly frozen LR-selection protocol with fewer pilots, or a changed
  compute limit; rescheduling the same workload changes neither.
- **"The TypedList eval harness is retired."** Reverted; nothing was deleted.
  `deep-corpus-v1` is the hard corpus and it lives on that harness.

## Where things live

- `docs/metrics.md` — how cost and success are measured: units, the AUC solve-cost metric, the Kt bridge, calibration constants
- `wandering_light/stack_machine.py` — the machine
- `wandering_light/stack_machine_corpus.py` — PBE corpus generation, certified by BFS depth
- `docs/stack_machine_explorer.html` — keyboard-driven explorer, self-checks against the Python core
- `wandering_light/rollout_debugger/` — the interactive rollout debugger; `cli.py serve` runs it
- `reports/` — every measurement, one directory per run
- `experiments/` — the scripts that produced them
