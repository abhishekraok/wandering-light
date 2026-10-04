VERDICT: CHANGES_REQUIRED

Review of frozen `59f7e7d6017e806719460fbe2c3df46af2d3a33c` for score-gap-feasibility (#180). The tree was clean before and after; nothing was written and no model, corpus, gate or declared entry point was touched. The design's core is sound and honestly labelled on the pass side. Four things need fixing before data: one would likely burn the single launch, one can flip the verdict, and two are wording/definition gaps.

## Blockers

**B1. The global attempt cap contradicts the per-cell caps (`DESIGN.md:75-77`).**
- A width-64, depth-48 cell can make 28 + 28·28 + 46·64·28 = 83,244 attempted-op callbacks (`stack_beam.py:416-431`; 28 ops confirmed from the frozen tree).
- Across 144 cells that is 11,987,136, against a cap of 1,000,000. Twelve cells that run to max depth exhaust it; an equal share is about 3.4 full layers per cell.
- Refused ops are charged only for the examples that stepped before the refusal (as few as zero), so the 200,000-execution budget does not bound attempts much below this.
- Non-fit cells are expected: support only requires 12 of 24 informative groups. The likely outcome is therefore resource-incomplete with no retry, which spends the launch for nothing.
- Fix: set the cap at or above the derived worst case, or record attempts compactly (op index, reason code, fuel, stepped, with successors re-derivable from parent state plus op) and re-derive the 2 GiB bound to match.
- Also state the stop mechanism: an exception raised from the observer, never a shortened `deadline` that production would report as `wall_clock_cap`.

**B2. The gate comparisons are not specified as exact, and floats flip the verdict (`DESIGN.md:142-143`).**
- With 20 informative groups, band 20/20 against prefix 18/20 gives `1.0 - 0.9 = 0.09999999999999998`, which fails `>= 0.10`. So does 19/20 against 17/20.
- My probe found 12 such disagreements for n in 12–24, all at n = 20; the protection threshold itself has none.
- Fix: compute protection, gain and slot means as integer or `Fraction` comparisons, in both the implementation and the verifier.

**B3. Definitions that must be pinned before data.**
- **Empty retained beam (`DESIGN.md:132-134`).** Production can score a layer and retain nothing (`stack_beam.py:479-481`). The cell then has a scored layer with Σ|C| = 0, which the "no scored layers → 1" rule does not cover. Define Σ|C| = 0 → 1 on both sides, counted.
- **Half assignment (`DESIGN.md:43-44`).** "Alternating positions" does not say which parity is selection. Also state ascending lowercase hex of UTF-8 bytes and minimum-digest choice at lines 39 and 43; #178's preregistration said "smallest", this one does not.
- **Minus-infinity (`DESIGN.md:121-122`).** "Explicit minus-infinity is retained in profiles" reads as the band keeping it. The #179 contract excludes −inf under a finite anchor. Reword to "recorded, excluded by any finite-anchor band", and say what happens when the labelled ancestor is −inf.
- **Comparison precision (`DESIGN.md:118-119`).** State the float64 comparison explicitly. Cumulative scores are float32 under the shared env's NumPy 2.4.4, and #179 casts to float64 only for the band test.
- **Legal mask (`DESIGN.md:98-99`).** A C state's own legal row exists only if it was fully expanded in the next layer. Define the field as null or partial for the last or interrupted layer, or record the parent's row.

**B4. The failure-side claim overreaches (`DESIGN.md:15-16`).**
- "A failed construction kills this route" is too broad. If the width-32 prefix already protects more than 0.90 of lineages in any seed, the +0.10 gain is arithmetically unattainable whatever the band does.
- In that case a band could still match protection at far fewer slots, so the route is not dead.
- That ceiling is the a priori likely case. The labelled lineage was selected by this same ranking at width 64, and expansion in beam order returns the lowest-ranked fitting parent (`stack_beam.py:416-445`).
- Fix: add a distinct terminal such as `NO_HEADROOM_OVER_PREFIX32`, and narrow the kill sentence to the registered conjunction. The pass side is adequately hedged and needs no change.

## What holds up

- **Lineage:** exact typed ancestors are recoverable from the unmodified `BeamObserver`. Retained nodes carry their programs, `parents[rank]` gives `(node, op)`, and seen-set dedup makes each prefix unique in its layer. No witness strings or state guessing are needed.
- **First fit:** the fit returns before scoring, so the terminal layer correctly has no scores. A program of length L has L−1 scored ancestor layers, so "at least two" means L ≥ 3.
- **Resource semantics:** the budget counts successful per-example executions, and an expansion is charged on parent entry before op 0. Both match production.
- **All-illegal rows:** these give NaN only at illegal positions, which the design ignores.
- **Feasibility:** nothing requires modifying production or calling the #179 candidate.
- **Pins:** `inputs.json` is identical to the three difference rows of #178's `inputs.json`. #178 found 134 eligible groups under the same rule, so 48 is attainable.
- **Statistics:** selection and confirmation halves, per-seed conjunction, no pooling and locking tau before confirmation are all sound. The band-versus-prefix comparison is a genuine Pareto test on the same snapshots.
- **Literature:** all seven source hashes match, and the cited lines check out except one (below).

## Nonblocking notes

- **Support is borderline.** Each half and seed needs at least 50% fits with L ≥ 3. The only reference figure I found is 161 of 300 tasks for capped/difference flat-48 on the held-out cohort, with no 10-second cap or observer overhead. `INSUFFICIENT_LINEAGE_SUPPORT` is a realistic outcome; say whether the descriptive rank/gap tables are still published in that case.
- **Timing-dependent support.** The 10-second deadline includes observer callbacks, so a fit versus `wall_clock_cap` can depend on host load and cannot be reproduced by the verifier. Count such cells separately, or let the execution budget be the only binding per-cell cap.
- **Granularity.** For 12–19 informative groups, protection ≥ 0.95 means zero misses; for 20–24 it allows one. State this in the design.
- **Smallest feasible tau** is the most fragile choice for confirmation. This is acceptable as registered.
- **Recorder.** Do not reuse the debugger's `TraceRecorder`: it rounds scores and stores text states. Copy `log_probs` and the beam, and store exact float bits.
- **Extra descriptives.** Add the full fixed-prefix curve (widths 1–64) and slot fractions restricted to informative cells.
- **Citation.** `LITERATURE.md` cites `variable_stream.py:464ff` for the sort; the file has 462 lines and the sort is at line 459.
- **Panel overlap.** The panel can overlap #178's 64 groups, whose per-behaviour losses are already known. Disclose it; it is not outcome selection.
- **Memory limit.** The enforcement mechanism for the 4 GiB limit is unspecified.

## Scope and what I did not check

The scope is narrow but worth running once B1–B4 are fixed. A pass only licenses the next preregistration; the descriptive rank/gap table is the durable output either way.

I did not hash the side file or open any data or weights. From #179 I read `DESIGN.md`, `AMENDMENT_A.md`, `GATE_A.md` and `experiments/score_band_search.py`, not the independent-verification or final-review reports in full.
