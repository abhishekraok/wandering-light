# Fresh layerwise Gumbel retention: negative train screen

The fixed candidate verifies **7/24 versus ordinary beam’s 8/24 in every pair**. It loses `running_minima` and gains nothing. Both registered coverage gates fail; the clock gate passes. Park this candidate without changing noise, panel or thresholds. Production and BEST are unchanged.

Executable source: `0212640d68e39e5d0ddf7024ad0efc55264d8809`. Original and explicit delta code reviews passed before execution. The [operative registration](https://github.com/abhishekraok/wandering-light/issues/183#issuecomment-6206493716) preceded candidate execution and retained DESIGN and Amendment A unchanged. The comparison unit completed October 15 at 19:36:20 PDT with a complete summary and wrapper `exit 0`. [Final independent code/result review](FINAL_REVIEW.md) passed on `23a66142b1ca27d5eb50ad0ff6816a5f83655507`; only the reviewer-requested state updates and faithful review artifacts follow that acceptance.

| Pair seed | Control verified /24 | Candidate verified /24 | Control solve seconds | Candidate solve seconds | Ratio | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| 17 | 8 | 7 | 54.67850980599178 | 54.94798950300901 | 1.004928438941979 | -1 |
| 29 | 8 | 7 | 54.73237713606795 | 56.16230645199539 | 1.0261258397816808 | -1 |
| 43 | 8 | 7 | 55.0095840600261 | 53.961482725979295 | 0.9809469322127009 | -1 |

Median net gain is −1 against the required ≥2; every paired net gain is negative against the required ≥0. Median all-attempt solve-clock ratio is 1.004928438941979, below 11/10. Exact Fraction arithmetic on saved binary floats gives 145230657167/144518407022. The negative gate is a coverage failure, with no speedup or runtime-penalty conclusion.

Every control has 12 shown fits, 10 hidden-graded solves and 8 verified solves: six catalogue-witness-equal and two draw-tested genuine. Every candidate has 11 shown fits, 8/8/9 hidden-graded solves and 7 verified solves: five witness-equal and two draw-tested. Four behaviours are shortcuts in every cell; none is inconclusive. The two draw-tested genuine solutions are `maximum` and `remove_values_seen_in_second`. `COMPARISON.json` retains full sets, lengths, stop reasons, executions and fuel; `RAW/` retains every timing cell, verification output, provenance and host snapshot. All 24 task attempts, including failures, contribute to each solve-clock total.

Whole phase wall time is 525.307655882003 seconds, including acquisition, process setup and post-timing verification. Subprocess wall times are separate in `COMPARISON.json`. Initial actual unit controls (12 GiB RAM, no swap, 2400 seconds) are saved at main `checkpoints/gumbel-retention-screen-20261015/comparison-status-initial.txt`. Every cell recorded numerical and interop threads equal to one and actual module origins in the frozen executed tree. The registered order B17, G17, G29, B29, B43, G43 used fresh processes and completed before grading. No concurrent own compute or review job ran during timing.

Prelaunch and before/after snapshots showed no other process above 2% lifetime CPU; cell load averages ran approximately 0.25–1.09. Point checks do not prove continuous exclusive-host execution. Baseline-feasibility clocks had recorded external activity and are separate; all comparison controls were newly timed.

The parent post-run audit reconstructed the hash panel from original train rows, replayed shown examples with typed values, recomputed hidden grades and all summary arithmetic, and regenerated the registered 10,000-draw checker for all 17 distinct returned programs. It checked final raw, source, input and checkpoint hashes and the exact gate conjunction; `COMPARISON_AUDIT.json` records scope and pins. This was not independent full-search replay.

Resource deviation: parent post-run baseline and comparison audits were inadvertently launched without the required memory scope. No resource qualification is claimed for those audits. The original measurement units, including their original verification, were correctly capped. The independent reviewer repeated all correctness-relevant checks inside an actual 12 GiB/no-swap unit and reproduced the same outputs; the documented parent-audit omission leaves no concrete result blocker. [Independent check](INDEPENDENT_CHECK.json), [script](INDEPENDENT_CHECK.py).

Scope is one trained policy, 24 hash-selected TRAIN skeletons and library off, CPU/one thread, width 64, depth 48 and prospective five-second task caps. The repeats vary search and timing, not trained policies. Fresh layerwise noise is a heuristic, not conditioned stochastic beam sampling. This negative does not settle other diversity methods, held-out behaviour or deployed BEST. No favourable retry, threshold adjustment, held-out study or promotion follows.
