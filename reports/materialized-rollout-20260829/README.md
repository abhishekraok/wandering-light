# Materialized-state rollout

Materializing every intermediate state helps, but does not cross the d7–8
horizon. The validation-selected beam 64 solves **12/48** held-out d5–8 tasks
(d5 7/12, d6 5/12, d7–8 0/24), where both one-shot models solved 0/12 at
each distance. It solves 60/96 overall, versus 14/96 and 4/96 for the two
one-shot k=64 runs in [fine-tune-llm (#50)](../../issues/50).

| beam | val d5–8 | test all | d5 | d6 | d7 | d8 | test d5–8 mean executions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2/48 | 25/96 | 2/12 | 0/12 | 0/12 | 0/12 | 140.3 |
| 8 | 2/48 | 46/96 | 3/12 | 2/12 | 0/12 | 0/12 | 968.8 |
| 10 | 2/48 | 47/96 | 3/12 | 3/12 | 0/12 | 0/12 | 1,187.7 |
| **64** | **7/48** | **60/96** | **7/12** | **5/12** | **0/12** | **0/12** | **6,238.1** |

At d5–8, beam 64 spends 299,429 executions. BFS under a uniform per-task cap
at or below the same mean spend solves 1/48; reaching 12/48 costs BFS
3,214,899 executions, **10.74x** rollout. There is no d7–8 cost win because
rollout solves none there.

The failure is consistent with exposure bias. On the 24 d7–8 tasks, the
recorded optimal witness remains in the beam for 24, 22, 17, 7, 2, and 0 tasks
after depths 1–6. This is path survival, not survival of every optimal action,
and structural deduplication can retain the same state through another path.
Still, every recorded witness is gone by depth 6 and all 24 searches fail.
A DAgger-style off-path pass is the direct follow-up, but labels remain cheap
only for visited states inside the certified expansion depth. A value head and
type-only lookup remain ruled out by prior measurements.

The 90,374-parameter aligned-pair policy trained for 12 epochs on all 107,049
discovery prefixes, selected on 15,046 validation prefixes, and materialized
zero test prefixes. Best validation CE was 1.39349 at epoch 12. The checkpoint
is a five-file weights-only artifact; weights SHA-256 is
`1a1e0d5815691ca9b1104caf44926c8ff746514fb953b24c48a3a3ad558b9bd6`.

Protocol: seed 20260816, 12 tasks per distance, depths 1–8, independent beams
1/8/10/64, depth cap 8, and 100,000 compatible-execution budget. Beam 64 was
[locked on validation](selection-lock.json) before the
[single-use test receipt](selection-lock.json.test-consumed.json). The exact
held-out cohort and [BFS table](bfs_reference.json) are hash-pinned.

Evidence: [training](https://wandb.ai/abhishekraok-na/wandering-light-rollout-policy/runs/4f4lqchn),
[validation](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v1/runs/vbfddpkl),
[test](https://wandb.ai/abhishekraok-na/wandering-light-deep-corpus-v1/runs/8gp1k5kg),
[validation summary](validation/summary.json), and [test summary](test/summary.json).
