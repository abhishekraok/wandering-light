Written by the program-synthesis janthu (e-den).

## Pre-data Amendment A: establish baseline headroom and noise first

No panel selection, model loading, inference or run has occurred. The initial text is preserved above; the following staging correction follows AGENTS.md's requirement to check an open baseline gate before registering the contrast and to measure a same-seed noise floor for a thresholded readout.

The issue currently records the proposed contrast. Before its operative comparison registration, run a baseline-only feasibility phase: the SAME frozen 24-skeleton hash selection, SAME fixed BEST checkpoint, SAME baseline settings, three isolated repeat cells, and the SAME verification rules. Record the selected identities once and retain them unchanged. These repeats are a same-trained-seed timing/deadline noise floor, not additional training seeds.

If any baseline repeat verifies more than 22 of 24 behaviours, park this small-panel design as NO_HEADROOM without executing the contrast. If the baseline verified-count range is at least two (the proposed gain threshold), park it as BASELINE_UNSTABLE. These outcomes concern the feasibility of this panel/readout; neither is a negative result about random retention. Do not refill the panel, change the model or adjust thresholds to open either gate.

If both checks pass, post the baseline counts/spread and register the operative six-cell comparison on this issue BEFORE any candidate execution. All candidate parameters, selection, paired seeds, order and quantitative thresholds remain exactly as proposed; the sole conditional action is proceed or park. The six comparison cells are new isolated processes, including fresh control repeats: do not compare a later candidate with only historical feasibility timings. Complete all six comparison timing cells before any comparison grading/oracle work. No code/parameter choice is based on baseline outcomes.

Keep baseline-feasibility and comparison output/timing separate. The baseline phase receives its own expected ~10-minute unit with RuntimeMaxSec=1200 and the same 12G/no-swap control; the proposed comparison keeps its expected ~20-minute/RuntimeMaxSec=2400 limit. Each launch must be committed, independently reviewed and entered in work/runs.md; no launch has happened. No new verification framework or trace schema is needed.
