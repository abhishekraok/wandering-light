# Deep-corpus-v2 scratch follow-up: incomplete recovered evidence

Issue [#54](https://github.com/abhishekraok/wandering-light/issues/54)
did not reach a solve-rate result. This directory preserves everything that
could be recovered durably from the eight W&B training runs after the local
`/tmp` worktree and its checkpoints were lost.

**The telemetry itself is not committed.** It lives where telemetry belongs —
the eight W&B runs linked throughout this report, under project
[`wandering-light-deep-corpus-v2-scratch`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch).
A recovery manifest holding the configurations, summaries, per-epoch
validation-loss records, file and artifact inventories and SHA-256 snapshots was
assembled at digest
`sha256:2c99e6c4ccdca4659f9da8e619ab447f31703abc4fcd7f862968a6e8fde3c3ff`; it is
a 2,248-line dump and the repository keeps the reading of it rather than the
dump, the same split `wandering_light/corpus_hub.py` applies to corpora. What is
below is that reading.

## What survived

All eight runs were created from source commit
`2318603cb3e279c4d91af9a1071b177c92123da2`. Both architectures record
`AutoModelForCausalLM.from_config`, `pretrained_weights_loaded=false`, seed
`54001`, and exact initial-state digests.

- Corpus: 313,438 witness-reexecuted records; manifest
  `sha256:37543f84d0bb376f285c1639ee6fe98bbdc693e41501741f5562424d3de1e4ee`;
  public payload revision `36753a013f3a4e965892b33bc44a8cc0fd567d99`.
- Tokenizer:
  `sha256:d4736f3a1623a89945daeaa8495250f28b8b47725d60f4d678fa9a0c1d8d5f66`.
- Basis: `wl-core-v1`,
  `sha256:ffc602fb24db249df7eb4f6b0d5ba38d5a5070b2c68c6c5f94c7f387de682494`.
- W&B recorded runtime for the six pilots and two finals: **20,267.804 s
  (5.630 h)**. Their trainer-reported training runtime totals **17,494.756 s
  (4.860 h)**. These totals exclude the feasibility work and the CPU BFS
  validation pass.

### Learning-rate pilots

Each pilot trained for one quarter epoch. Both arms selected `3e-4` by minimum
validation loss.

| arm | LR | validation loss | validation token accuracy | W&B run |
|---|---:|---:|---:|---|
| 10M | `1e-4` | 1.461842 | 0.586185 | [`piekprnm`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/piekprnm) |
| 10M | **`3e-4`** | **1.036373** | **0.641587** | [`yij1euys`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/yij1euys) |
| 10M | `1e-3` | 1.084216 | 0.609090 | [`hmre6vte`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/hmre6vte) |
| 125M | `1e-4` | 1.005916 | 0.649136 | [`rw6g30r8`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/rw6g30r8) |
| 125M | **`3e-4`** | **0.964090** | **0.659510** | [`9sluj815`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/9sluj815) |
| 125M | `1e-3` | 1.376255 | 0.530614 | [`hx40xwh8`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/hx40xwh8) |

### Three-epoch final training

| arm | parameters | initial-state digest | epoch 1 loss | epoch 2 loss | epoch 3 / best loss | final token accuracy | W&B run |
|---|---:|---|---:|---:|---:|---:|---|
| 10M | 10,080,896 | `sha256:32946ac2...0167` | 0.819534 | 0.762868 | **0.755485** | 0.752199 | [`denrj1ok`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/denrj1ok) |
| 125M | 125,265,408 | `sha256:244f4e50...63dc` | 0.772039 | 0.696329 | **0.686474** | 0.761236 | [`k6x0qelr`](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v2-scratch/runs/k6x0qelr) |

Both finals completed 92,085 optimizer steps and recorded 64,893,407 training
tokens. The 10M run reports 5,008.111 s of training time; the 125M run reports
8,664.803 s.

## What did not survive or never existed

The W&B configurations record weights-only saving to staging directories under
`/tmp/wandering-light-issue-54-per-arm`, `hub_model_id=null`, and
`push_to_hub=false`. Neither final W&B run has a logged artifact. The local
recovery search found no surviving model weights or selection locks. The two
W&B artifacts attached to pilot runs `piekprnm` and `yij1euys` are
`wandb-history` artifacts, not model artifacts.

The first validation attempt failed before generation because the evaluator
passed `token_type_ids` to a causal-LM path. The amended attempt completed all
96 candidate-generation tasks and all 96 matched BFS searches, but one
distance-8 BFS task reached the fixed 20M safety ceiling and the old validator
refused to seal the result. Its atomic staging directory was removed. No
candidate artifact, parse diagnostics, validation summary, or evaluation W&B
receipt survived.

Accordingly, none of the requested scientific outputs exists:

- no solve rate at any certified distance, including distances 5-8 or 7-8;
- no unparseable rate or parse-failure-reason distribution;
- no execution-correct-given-parseable rate;
- no matched-cost capacity comparison;
- no scratch scaling conclusion.

The frozen test was never model-scored; the recorded access count remained
zero and no opening ledger was created. The 360M arm remained deferred at a
17.333-hour projection, including three pilots, the three-epoch final, and the
15% margin. Per owner instruction, the lost 10M and 125M runs will not be
retrained.

This report is a recovery record of training evidence, not a substitute for
the protocol's validation/test result.
