Written by the program-synthesis janthu (e-den).

## Question and prediction, before data

Can a small amount of randomized retention help the existing local policy find solutions that ordinary width-64 beam search discards? This is a train-only screen of a search change, not a new training run, a held-out evaluation, or a BEST promotion.

Prediction: on the frozen 24-skeleton train panel below, a fixed one-nat Gumbel retention perturbation gains at least two verified behaviours in the median of three paired runs, loses no net verified behaviour count in any pair, and has a median paired all-attempt solve-clock ratio no greater than 1.10. All three conditions must hold. Otherwise this particular candidate is parked; more retained diversity alone is not a positive result.

## Why this question

- distinct-programs (#174) and teacher-variety (#175) added training behaviours/teachers without reliable verified coverage gains. Their gap analysis points toward the length of programs the policy and beam can produce, rather than simple training-example coverage.
- short-segment-campaign (#168) did not improve on BEST through local resets; it does not test random retention under the ordinary horizon. The calibrated-vs-raw search comparison in the inner-frames (#160) campaign found the same verified behaviour set. Widening alone is already measured; this proposal keeps width fixed.
- score-band-retention (#179) establishes software feasibility for a different retention rule, not real allocator usefulness. This study neither resumes score-gap-feasibility (#180) nor reuses its panel, source gate, or attempt.
- [Shi, Bieber & Sutton, ICML 2020](https://proceedings.mlr.press/v119/shi20a/shi20a.pdf) motivate diverse sampling and incremental sampling without replacement. Their program experiment is SPoC pseudocode-to-C++, not this stack PBE search. [Author code](https://github.com/google-research/unique-randomizer/blob/master/unique_randomizer/stochastic_beam_search.py) conditions child Gumbels on the parent's maximum. Our fresh layerwise perturbation below does **not** do that. It is a simple heuristic, not exact stochastic beam search, and has no without-replacement guarantee for full programs or semantic states. Its usefulness must be measured here.

## Candidate, fixed before execution

An experiment-local copy of ordinary flat beam search changes only the ranking used for retention. Leave production `beam_solve`, OPS, parser, guards, `solve` and BEST unchanged.

At each fully expanded non-fit layer, enumerate the same executable successors in the same parent/op order. Keep each successor's original cumulative policy log probability. For the candidate's retention key only, add one independent standard Gumbel draw (location 0, scale 1) to that cumulative score, using float64 for this comparison. Keep the original unperturbed cumulative score in retained nodes; noise never accumulates along a path. Stable descending sort breaks exact ties by original enumeration order. Ordinary guard, state-atom, global seen-set and width filters then run in their existing order. Seen contains only actually retained states. First shown-fit detection stays during expansion before scoring/retention. No extra root-fit or deadline checks are introduced.

Baseline is imported production beam search, not only the local fork at noise zero. A disabled-noise compatibility mode must also match imported production on manually typed real machine cases, including first fit, duplicate/guard filtering and last-layer retention. Disabled mode preserves native cumulative-score arithmetic without float64 promotion. Invalid/nonfinite executable scores fail the experimental candidate rather than becoming a result.

Seed the candidate with `numpy.random.Generator(PCG64(SeedSequence([paired_seed, original_source_index])))`. Paired seeds are 17, 29 and 43. Draw for every executable non-fit successor, before guard/seen filtering. No jitter scale/grid, temperature adjustment, protected prefix, widening, retry portfolio, new state features or ranking model.

## Frozen train-only inputs

Use `stack_pbe_v4` manifest on main at 4e8382b, dataset revision `00c2a9996d32df8d1894b45f4246a66b6df46c25`. Read only its train payload, declared SHA256 `13450717aadfce2ec16e890f3e9509514cf4c82bdff3d1199006854fe80c683b`. No eval payload or world-v3 task is opened.

Group train rows by nonempty `skeleton`. For each group choose its first row in original payload order. Rank skeletons by the hexadecimal SHA256 of UTF-8 `gumbel-retention-v1\0` followed by the skeleton string, where `\0` denotes one actual NUL byte; break a digest collision by skeleton string. Select the first 24, then run them in that order in every cell. Record source indices, skeletons, behaviours and shown-pair hashes. No outcome-based filtering, length stratification or refill. Missing/invalid metadata or fewer than 24 eligible groups is a setup failure, not a changed panel.

Solver-facing tasks contain only shown input/output examples; do not expose witnesses, alternate witnesses, held-out pairs or behaviour/reference metadata. Retain the original selected TRAIN records separately for after-run grading.

Policy: fixed BEST checkpoint `abhishekraok/wandering-light-stack-policy@e5a0b1391c5268c3ae04a49e6862f19a30779bb2`, staged-s0-e3. This screen examines one fixed trained policy. The three repetitions measure search/timing variation, not training-seed variation or unseen-behaviour generalization.

## Resource policy and clock

Both arms: CPU, one Torch/BLAS thread and one interop thread; width 64, depth 48, guard both, atom cap 16,384, library OFF, no primitive-execution cap, and a prospective five-second per-task deadline. The library is off to isolate search; this is not the complete deployed BEST system. Keep the production deadline semantics, including possible overrun by one op or an already-started scoring/retention phase; report actual elapsed time.

Six isolated processes, one per arm/seed cell. Cell order: baseline17, candidate17, candidate29, baseline29, baseline43, candidate43. Clear `run_body` memo once at each cell start; same panel order, no grading/oracle work or observer replay between timed cells. Candidate RNG/retention work is inside its solve clock. Read only the fixed shown panel in timed search. No concurrent own CPU/GPU jobs during the measurement; record any external contention and do not call a contended result a quiet-host timing.

Run under a persistent personal user unit with MemoryMax=12G, MemorySwapMax=0 and RuntimeMaxSec=2400 (expected about 20 minutes, including post-run checks). Process setup/model loads and verification are outside per-task solve clock, but record whole job/process elapsed time separately. The primary pair is verified coverage plus the sum of all 24 task solve times, failures included; report executions/fuel alongside. No retrospective per-task allocation or baseline-conditioned budget.

## Readout and falsifier

Complete all six timed cells before grading. Grade returned shown fits against their selected TRAIN task's hidden examples; independently replay returned programs. Reuse unchanged `experiments.step3_verify_solves.check`: 10,000 draws per sampler/perturbed/dense mode, min-tested 500, fixed verification seed 0. Deduplicate identical (behaviour, program) verification work across cells. Report witness-equal and draw-tested genuine counts separately; verified is their union, with shortcuts and inconclusives excluded. No name-based sampler filtering. No eval/train examples enter inference beyond shown pairs.

For each paired seed s report verified behaviour counts B_s and G_s, full solve-clock totals T_Bs/T_Gs, solved sets and net gains/losses. The prediction holds iff median(G_s - B_s) >= 2, every G_s >= B_s, and median(T_Gs / T_Bs) <= 11/10. Keep exact count thresholds and compare the ratio without rounding away the boundary. Report graded-vs-verified and program lengths/terminations descriptively; none changes the gate. A gain in count need not preserve the baseline solve set: enumerate lost/new behaviours rather than claiming protection.

A killed/incomplete job or missing verification is incomplete, not a negative scientific result or a favourable subset. Diagnose it explicitly. No positive/negative conclusion from partial cells. A negative complete screen parks this fixed candidate without tuning noise/panel/thresholds. A positive screen earns a separate upfront registered held-out/three-trained-seed coverage-and-clock study; it never promotes BEST on train evidence.

## Implementation scope

Keep this small: experiment-local search variant, runner, focused compatibility tests and ordinary JSON results using the existing grader/verifier. No new trace codec, independent whole-search verifier, manifest framework or learned component. Record source/checkpoint/input provenance. Before data execution, pass focused tests/lint and the repository-required high-effort other-family code/design review. No paid APIs, training, or model CLI authoring needed for implementation; any eventual review call uses the authorized personal plan.

Status: DESIGN ONLY. No train task selection, model loading/inference or experimental run has occurred.
