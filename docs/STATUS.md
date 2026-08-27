# Status

**This file records state. Issues record argument. Never both.**

So they cannot contradict each other: if you want to know *what is true*, read
here; if you want to know *why*, follow the issue link. A claim that is not in
this file is not established, however confidently some comment states it.

Update this file as part of finishing work, not as a separate chore.

Last updated: 2026-08-27

## What is true now

Every row is merged to `main` and measured. The number is the point; the issue
holds the argument.

| | | issue |
|---|---|---|
| **Stack machine** | 52 ops (26 executable + one symbol each). Guards return a reason or `None`; illegal is an *absent transition*, not an exception. | [#42](../../issues/42) |
| **Identity is structural** | `value_key` builds nested tuples of str tags, ints and bools. No hash, no repr, no dict. Keys are byte-identical across `PYTHONHASHSEED`, pinned by a test running three seeds in subprocesses. | #42 |
| **Fuel** | Every first-order op costs exactly 1, so `fuel == len(program)` there. Combinators break it: `map` costs `1 + body x len(input)`. Totality rests on the guard, not on structure. | #42 |
| **Branching, stack machine** | first-order 18: `b_eff` 11.14, 3.464 bits/step. executable 26: 13.32, 3.712. full 52: 37.28, 5.220. | #42 |
| **Branching, `wl-core-v1`** | 118 functions, 20.52 applicable, 4.292 bits/step. Type masking is worth ~3.9 bits, slightly better than #33 assumed. | #33 |
| **Substrate comparison** | Both bases generate new behaviour at ~the same rate under composition — 5.87x/op against 5.77x. The stack machine's win is price: 73% of choice bits become new behaviour against 59%. It pays a **median 2.00x in program length**. | #42 |
| **Stack corpus** | `stack_pbe_v2.jsonl.gz` — 1,857 PBE tasks certified shortest to depth 6 against `executable-26`, 1,502 train / 355 eval, 101 needing a loop. Split is a SHA-256 of the task, not a shuffle. | #42 |
| **Stack corpus composition** | **0 tasks construct a quotation.** 680 receive one in the input and 522 of those (76.8%) never execute it — they `drop`, `cons` or `cat` it. 219 (11.8%) run code at all; 1,026 (55.3%) use only literals, plumbing, structure and quotation ops — no arithmetic, no predicate, no combinator; 966 (52.0%) output a nested list. The BFS and solver numbers below are therefore measurements on stack rearrangement. | [#59](../../issues/59) |
| **Stack BFS baseline** | 75.2% at a 10,000-node budget; median 1,119 nodes per solved task. Node cost grows ~7.5x per unit of program length. | `reports/stack-machine-basis/` |
| **`deep-corpus-v1`** | 31,350 tasks, certified distances 1–8, 240 roots, split by root. BFS solve rate is a **step function**: 100% up to its depth cap, 0% beyond. Distance-8 answers cost it ~4.4M executions. | [#37](../../issues/37) |
| **Kt vs length** | Re-certifying the 101 loop tasks under `\|p\| + log2(fuel)` changed **0 labels, 0 bits**. A label needs ~26x less fuel for one extra op; four-element lists cannot vary that much. A fact about input sizes, not the metric. | [#45](../../issues/45) |
| **Basis growth by MDL** | Degenerate. Accepted 400 words without converging, basis 118 → 518, corpus −36.4%, **search 42,631x more expensive at depth 6**. Every candidate clearing the bar is length 2. | [#41](../../issues/41) |
| **Policy probe, first action** | A lookup table on `(input type, output type)` reaches mean rank 3.18 and top-1 42.1%. Two byte-level encoders beat it by 10% on rank and tie on top-1. | [#33](../../issues/33) |
| **Policy probe, last action** | Learning pays here and barely does at the front: lookup 4.80 / 22.3%, conv encoder **3.43 / 42.1%** — 29% better rank, nearly double top-1. Masking is weaker at the back (25.88 functions produce a given output type against 9.87 consuming a given input type). Neither end degrades with distance. | #33 |
| **Value probe, distance-to-go** | Labels are free and exact: every witness prefix is an exactly labelled `(state, target, remaining)` point, 137,140 of them. A conv encoder reaches **MAE 0.978 steps** against a type-pair dict's 1.232 — learning pays 21% — but is off by **1.66 / 2.08 / 2.51 at remaining distance 6 / 7 / 8**, against a pre-stated bar of ~1. Train MAE 0.830 says representation-limited, not undertrained. As an expansion orderer `h` scores mean rank 3.35, *worse* than the policy lookup table's 3.18. | [#33](../../issues/33) |
| **Policy probe, inward** | The front/back asymmetry is a **first-step artifact**. `deep-corpus-v1` roots exclude `int`/`str`/`float`, so a task's input type is rare and masks to 9.50 of 118 functions; one op later states collapse onto `int`/`float`/`str` and the mask widens to 18.41, settling at 19.90. The back is flat at 25.87 → 26.87. So the dict matches the encoder at `k=0` only; at `k=1,2` on both ends the conv encoder beats it by 16–25% on rank and 23–88% on top-1. | [#33](../../issues/33) |
| **Learned solver, run 1** | Pretrained `facebook/opt-125m`, 3-epoch SFT: 14/96 at k=64 on held-out. Cheaper than BFS at distances 3–4; **0/12 at every distance 5–8**. | [#50](../../issues/50) |
| **Learned solver, run 2** | Pretrained `HuggingFaceTB/SmolLM2-360M`, same corpus and locked 3-epoch protocol: 4/96 at k=64, no cost win at any distance, and **0/12 at every distance 5–8**. A stronger pretrained base did not move the target number. | [#50](../../issues/50) |
| **Why run 2 regressed** | Measured from the saved generations, not inferred: **42.4% of SmolLM2 candidates are unparseable against opt-125m's 0.4%**, essentially all invented function names (`f_sqrt` for `f_abs_sqrt`, `str_to_int` for `int_to_str`, `list_sort`) because the prompt never states the 118-name vocabulary. But conditional on parsing it is still 3.5x worse (0.28% vs 0.98%), so vocabulary is not the whole gap. **Pretrained priors are a liability on an arbitrary closed vocabulary** — which is why #54 tests random initialisation rather than a better pretrained base. | #50 |
| **`deep-corpus-v2`** | Built, published and verified: 313,438 certified records from 2,400 roots, distances 1–8, zero leakage across splits or against v1. Shared discovery-only tokenizer, vocab 4,196, basis names as atomic tokens. **No models trained against it yet.** | [#54](../../issues/54) |

## In flight

| what | who | where | question it answers |
|---|---|---|---|
| Scratch solver scaling | Codex agent | [#54](../../issues/54), PR #57 | Does a 10x corpus make random-init models viable? Corpus and tokenizer done. 10M (2.33 h) and 125M (3.72 h) arms are feasible and pending; 360M projects 17.02 h against a 16 h budget and is deferred to its own run. |
| Theory discussion | `stack-machine-vs-register` session | [#52](../../issues/52) | — |

## Superseded

Claims that were stated confidently and are wrong. Recorded here so a
correction lands in one place instead of being chased across comment threads.

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
- **"The autoregressive solver is finished on this substrate."** Overstated
  after run 1. A stronger pretrained SmolLM2-360M also remained 0/12 at every
  distance 5–8, which supports the bounded conclusion for this locked
  completion-only pretrained SFT protocol. It does not settle random-init
  training or other objectives; #54 is the scratch follow-up.
- **"Runs 1 and 2 used random initialization."** False. Both call
  `AutoModelForCausalLM.from_pretrained`; neither is scratch evidence.
- **"The TypedList eval harness is retired."** Reverted; nothing was deleted.
  `deep-corpus-v1` is the hard corpus and it lives on that harness.

## Where things live

- `wandering_light/stack_machine.py` — the machine
- `wandering_light/stack_machine_corpus.py` — PBE corpus generation, certified by BFS depth
- `docs/stack_machine_explorer.html` — keyboard-driven explorer, self-checks against the Python core
- `reports/` — every measurement, one directory per run
- `experiments/` — the scripts that produced them

## Open traps

- **PR #57's tracked manifest is 2.51 MB**, of which `expansion.roots` — 2,400
  blocks of per-root generation telemetry — is **98.5%**. Removing just that
  array leaves 42 KB with all identity, digests, histograms and `split_roots`
  (13 KB) intact. Fix it **before the PR merges**; afterwards it needs a
  history rewrite. #40's policy says "track the manifest" but sets no size
  rule, and manifests grow linearly with root count.
