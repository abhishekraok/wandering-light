# Learning-algorithm brainstorm, 2026-09-20: designs set aside

Exploratory and **unreviewed**. A design panel and a few throwaway probes,
none registered, none checked by a second agent. Each line gives the probe's
own scope; none is a general result. Raw designs, critiques and scripts are in
the main checkout's gitignored `.drafts/from-learning-algorithm-design-bd20f2/`
(UltraDesk only). What survived the panel became inner-frames (#160),
branch-and-check (#164) and meeting-pairs (#169).

| Design set aside | The probe, and its scope |
|---|---|
| A learned landing/option model of the executor | Beam split on 8 v4 eval tasks, 2m policy, CPU, 8 threads, beam 64 x depth 12: the scorer is 73.7% of wall-clock and the executor 19-26%. A model of the executor adds a forward pass to approximate the cheap part. One op per candidate; a design that runs whole programs per candidate inverts the split. |
| The persistent graph as retrieval memory | Uniform-random walks gave 300 distinct endpoints of 300 per tested root ([walk-collapse-stack](../walk-collapse-stack-20260908/README.md)). Reconvergence and retrieval under a directed policy were not measured here. |
| Training a value head on random rollouts | Argument, not a measurement: on a random chain the remaining length identifies the target, so the head can learn that shortcut. |
| Library-first on world-v3 | world-v3 was authored in that library, and 76-81% of macro-form tokens are still under an open mark. |
| Edit/repair search | Nearest train witness by edit distance: v4 eval 17 of 45 within 4 edits and 27 of 45 within 8; world-v3 eval 2 of 44 within 4, 4 of 44 within 8, 31 of 44 beyond 24. Of 40 v4 witnesses corrupted by 2 edits, 15 have any single edit that strictly raises the exact-match count; by 3 edits, 2 of 40. |
| Enumerating cycles as an action set | From one real v4 input, `[ 0 2 ]`: 326,875 legal `[body] C` cycles at body length 4 or less, reaching 278,641 distinct closed states. |

Also counted: 1,400 quotation bodies in v4's 175 programs and 15,060 in
world-v3's 482, the multiplier if bodies were treated as PBE sub-tasks.

Revisit any of these when new evidence or a changed design alters its premise.
