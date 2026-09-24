# Do short-body programs exist for the behaviours `BEST` cannot solve? (short-body-authoring #167)

**BETWEEN, as registered. Short-body programs exist, but `BEST` pays too much per op for any decomposition to reach them.**

- **They exist for all 25.** Every one of the 25 #101-cohort behaviours `BEST` does not solve has a verified program whose executed quotations are all at most 4 atoms, against a registered forecast of 15-18.
  - 22 of them pass #158's 10,000-draw test.
  - Three are unrolled to the sampler's longest input: `cartesian_pair_sums`, `stable_distinct_shared_values`, `widest_pair`.
- **They are long.** The median program is 67 ops (forecast 25-35 by the advisor, 45-60 by me), and 15 of 25 are longer than the authored witness. They break into a median 8 closed-state segments of at most 13 ops each; the authored witnesses have 2 segments of about 50.
- **No accounting clears them under `BEST`.**
  - Raw log-probabilities: 0 of 25 clear flat, with a budget reset at closed states, per body, or both. The authored witnesses also clear 0.
  - `BEST` pays 2.56 raw nats per op on the capped programs, and 1.74 on the authored ones.
  - A 13-op segment fits `ln 64` only below 0.32 per op. `BEST` pays about 0.35 per op on programs it finds itself (#165).
- **So the binding constraint is the policy's per-op cost on this style,** not body length and not how the search decomposes. Segment-reset search would pay off only for a policy about 8 times more fluent in short-body pipelines than `BEST` is on these behaviours. That is roughly as fluent as `BEST` already is on the ones it solves.

[Registration](https://github.com/abhishekraok/wandering-light/issues/167) posted before any program was authored.
- **Authors:** 9 Opus agents, 3 behaviours each (brief: `BRIEF.md`), and Codex as second author on the one Opus left at cap 8.
- **Checker:** `experiments/short_body/check.py`.
- **Scoring:** `experiments/short_body/score.py` gives `scores.json`.
- **Programs:** `programs_*.jsonl`.

**Status of these programs:** analysis artefacts only, never training data.

## Against the registration

| | advisor session | mine | measured |
|---|---:|---:|---:|
| cap 4 met | 18+ of 25 | about 15 | **25** (22 not unrolled, 22 genuine) |
| cap 8 met | 22+ | about 21 | 25 |
| median length, cap-4 programs | 25-35 | 45-60 | **67** |
| median longest segment | 6-8 | 8-12 | **13** |
| `BEST`, raw, clears flat | at most 3 | at most 2 | 0 |
| `BEST`, raw, clears with segment-reset | 10+ | about 6 | **0** |

**Reading.**
- It is not SEGMENTED_SEARCH: cap 4 was met (25 ≥ 15), but segment-reset adds 0 clears against the 5 needed.
- It is not NEEDS_LONG_BODIES: cap-8 programs exist for 25, above the 12 that reading needed.
- So it is **BETWEEN**. Both forecasts of segment-reset clears were wrong.

## Per behaviour

Medians over the 22 genuine programs, under `BEST`:

| | authored witness | capped program |
|---|---:|---:|
| ops | 69 | 63.5 |
| closed-state segments | 2 | 8 |
| longest segment, ops | 54.5 | 13 |
| longest body, ops | about 40 | 5 |
| raw nats per op | 1.74 | **2.56** |
| total raw nats | 109 | 163 |
| body share of deficit | 0.85 | 0.75 |

`scores.json` has every behaviour. A few:

| behaviour | authored ops / segments | capped ops / segments | raw nats, authored / capped |
|---|---:|---:|---:|
| `polyline_manhattan_length` | 188 / 7 | 53 / 6 | 286 / 93 |
| `positive_majority_three` | 39 / 2 | 22 / 3 | 64 / 18 |
| `median_floor` | 253 / 6 | 287 / 40 | 286 / 759 |
| `keep_ascending_pairs` | 80 / 2 | 75 / 9 | 39 / 263 |

## Caveats

- **Unrolled programs.** Three programs work only up to the sampler's longest input. Two of them, and `nearest_interval_endpoint_each`, fail #158's denser 10,000-draw test.
- **Code run through `i`.** Three programs meet the cap by building per-item code with `map` and running it with `i` inside a 4-atom fold body: `prefix_sums`, `axis_point_count`, `furthest_squared_radius`. Each executed quotation is within 4 atoms, but the work per fold step is larger.
- **One odd trick.** `replace_value` detects equality by dividing by 2^64. It is correct on every test and draw, but no author would write it.
- **One policy.** Every number is under `BEST`, which was trained on staged witnesses (#155) with a median longest body of 8. A policy trained on capped witnesses could pay far less on these programs; this check cannot say how much.

## What it means for the next step

- **Neither build is licensed on today's policy.** Decomposing search, by segments (#156, #164) or by skeleton (#163), cannot rescue programs whose every 13-op segment costs `BEST` about 33 raw nats.
- **The lever is the policy's fluency in a short-body style.** `BEST` pays 0.35 nats per op on programs it finds, but 2.56 on these. That points to training-side re-authoring under a hard body cap: #155's pipeline with a cap of 4, on train behaviours only.
- **These programs are a clean yardstick for that re-authoring.** Score them, analysis only, under the retrained policy. If per-op cost falls toward 0.35 and segments stay near 13 ops, segment-reset clears become reachable, and the search build becomes worth testing.

## Does the style transfer? (`transfer.py`, registered on #167)

**TRANSFERS, as registered.** Staged over fused, raw nats per op on the capped programs, is 0.54 at the median over seeds (0.43-0.55). The advisor forecast 0.75-0.85, and I forecast 0.85-0.95.

The checkpoints are #155's arms at epoch 3 and the world arm. Median raw nats per op over the 22 genuine programs of the unsolved behaviours, none of them trained on:

| checkpoint | authored witnesses | capped programs |
|---|---:|---:|
| `fused`, seeds 0/1/2 (v4 witnesses as authored) | 0.43 / 0.42 / 0.44 | 4.75 / 4.89 / 4.57 |
| world arm (v4 + world-v3, authored) | 0.37 | 3.78 |
| `refused` (staged algorithms, cycles merged back) | 1.30 / 1.20 / 1.20 | 2.55 / 2.45 / 2.45 |
| `staged` (`BEST` is s0) | 1.74 / 1.77 / 1.82 | 2.56 / 2.10 / 2.52 |

- **Each policy is fluent in the style it was trained on**, on behaviours it never saw. `fused` pays 0.43 per op on the authored witnesses of the unsolved behaviours, near the 0.32 a 13-op segment needs. `BEST`'s 2.56 on the capped programs is mostly style mismatch.
- **The authored witnesses still fail for their length.** Their segments are about 55 ops, so even at 0.4 per op one segment costs over 20 nats. `fused` clears 1 of 22 flat and 1 with segment-reset; the world arm 1 and 2.
- **The capped programs are further from `staged`'s training than `staged`'s own witnesses are.** Their cap is 4 atoms against 8, and some use unfamiliar idioms: closures run through `i`, unrolling. `staged` halves `fused`'s cost on them, but still pays 2.1-2.6 per op.
- **Where the cost sits,** pooled over ops under `staged`:

| | per op |
|---|---:|
| content ops | 3.2-3.7 |
| structural ops (`[`, `]`, consumers) | 1.5-1.7 |
| mid-segment ops | 2.6-3.0 |
| last op of a segment | 0.7-0.9 |

  The pattern is the same under every arm. In style it is much lower: `fused` on authored witnesses pays 0.9-1.0 per content op pooled, 0.4 at the median program.

**So re-authoring train behaviours under a hard body cap is the lever the data supports.** A policy fluent in that style could plausibly pay near the 0.4 per op `fused` pays in its own style. That is borderline for 13-op segments, which need 0.32. So the likely outcome is some clears with segment-reset search, not many.
