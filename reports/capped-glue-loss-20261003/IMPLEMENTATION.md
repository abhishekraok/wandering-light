# D1 + Amendment A implementation

Implementation validation used synthetic tests. Independent high-effort Claude
[pre-trial code/design review](pretrial-review.md) passed on exact frozen
`62f40764bbe7b4ccf52fc3708a0eda22d850828a`. The single real diagnostic has now
completed exit0 with a complete six-checkpoint manifest; its recorded negative
mid-span terminal passed separate independent complete-result verification;
other-family high-effort result/code review subsequently passed on frozen
`bcd20fe`; [full result/code review](result-review.md) retains its checks and limitations.
The reviewed execution source stayed frozen throughout the run.
The original preregistration and Amendment A govern population, thresholds,
resource limits, interpretation and the no-rerun rule.

Final-source validation, 2026-10-03 13:28 PDT: the full repository suite passed
2858 tests, with40 skipped,7 deselected,1 xfailed and8 existing fork warnings,
in162.40 seconds. This ran from development `e0a3752`, whose only difference
from frozen review source `62f4076` was `docs/STATUS.md`; source and tests are
identical. Full Ruff lint and formatting checks also passed. This supersedes
the earlier full-suite count obtained before the last source/zero-work guards,
but supplies no independent eligibility or measured-effect verdict. The
high-effort Claude pre-trial review subsequently passed on exact `62f4076`.

`verify_stage.py` independently reconstructs original-skeleton selection,
shown interpreter replay, lexical owners/spans, exact common support and counts,
and supported-only legal masks/refusal scope and typed encoder exposure. It uses
interpreter/encoder primitives but imports none of the diagnostic modules.
Tests challenge it with corrupted serialized synthetic artifacts.
`verify.py` adds independent successful-terminal and review/input/checkpoint
hash checks, train/shown-only loading, scalar stable log-sum-exp, every raw
metric/cell/behaviour and complete descriptive-summary recomputation, exact row
and checkpoint order, both-encoding gates and surplus-artifact rejection.
It rejects unsuccessful terminals before raw-logit access and never deserializes
a checkpoint. Numeric comparisons use absolute/relative1e-10 tolerance, with
exact structure/identities. The CLI has a900-second verification deadline and
requires the expected reviewed revision explicitly. No new trial is launched.
All34 independent Stage A/complete-result synthetic tests passed14.35s, including
rehashed corrupt artifacts and unsuccessful-terminal no-logit-access checks;
full lint/format/diff checks passed. The complete real verification subsequently
passed from frozen `5211680`, with no model deserialization; its retained receipt
is [independent-verification.json](independent-verification.json). Neither
synthetic passage, pre-trial PASS nor our verifier replaces result/code review.

The model-independent pieces are `experiments/capped_glue_loss.py` and
`capped_glue_stage_a.py`; `capped_glue_metrics.py` supplies pure encoder/loss
diagnostics, and `capped_glue_run.py` supplies the bounded one-shot controller.
The latter is invoked from a clean frozen reviewed source tree:

```sh
UV_PROJECT_ENVIRONMENT=/home/abhishekrao/repos/wandering-light/.venv UV_NO_SYNC=1 uv run python -m experiments.capped_glue_run
```

The CLI accepts no data, checkpoint, output, selection or limit overrides. It
uses this tree's committed `inputs.json`, the six checkpoint order there and
main `checkpoints/capped-glue-loss-20261003/trial/`. The launch is an external
user unit with RuntimeMaxSec1800 and MemoryMax4G; its log directory must exist
before shell redirection. The diagnostic uses CPU, one torch thread, batches32
and a900-second cooperative deadline, including all staging/scoring and checks.
A single call may overrun it; the external bound still applies.

Before launch, retain the independent high-effort Claude report and inspect its
verdict. Only an actual PASS permits writing the fixed
`checkpoints/capped-glue-loss-20261003/pretrial-review/receipt.json`, with:

```json
{
  "verdict": "PASS",
  "source_revision": "exact frozen reviewed commit",
  "issue": 178,
  "design": "D1+AmendmentA",
  "reviewer_family": "claude",
  "effort": "high",
  "report": {"path": "absolute retained report path", "sha256": "report bytes hash"}
}
```

This records reviewed evidence; a receipt is not a substitute for receiving the
review. A wrong/missing receipt refuses before reserving a trial or reading data.
The CLI checks clean source and exact receipt revision; no source edits/merges
are made in that execution tree after review.

`attempt.json` reserves the campaign before input/model work. An existing
attempt or diagnostic artifact forbids retry, including after a hard kill. All
selected shown replays finish before support, and support failure skips encoder,
alternative and model work. Supported Stage A includes encoder diagnostics and
reason-bearing legal masks, then is atomically sealed before any checkpoint
load. Input hashes are checked around reading; checkpoint pins around loading.

Each checkpoint is loaded once in fixed order. Raw rows are written in sealed
order with descriptor, ordinal, identity, full28 logits and recomputable
diagnostics. Each summary is recomputed from complete saved raw vectors. The
final check covers exact sealed rows times six, hashes, identities, finite
vectors, summaries and the registered both-encoding gate. `final.json` is
published atomically last.

Supported positive/negative or complete support-only states exit0 with
`complete=true`. Cooperative errors exit2 with `INVALID`, `complete=false`,
explicit phase/progress/error and no effect reading. A hard kill/OOM without a
final manifest and an `exit` log line is NO RESULT. Only exit0 plus the complete
validated manifest supplies a reading; partial artifacts supply none. No rerun
after INVALID/deadline/kill/OOM; a future attempt needs new preregistration and
review before any partial-logit inspection.

`validate_complete_manifest` rechecks persisted artifacts without loading a
model. This is the implementation's completeness check, not the independent
result verification/review still required before landing. Loss/deficit/token
shares and truncation/censor records are descriptive. The primary only concerns
exact-op matched mid-span emissions; boundary glue and causal runtime/frame
claims remain outside it. Training-row sample multiplicity is unavailable.

Required secondary reporting is in [RESULT.md](RESULT.md) and
[SECONDARY.json](SECONDARY.json). `publish_secondary.py` copies the complete
independently verified all/common label summaries, per-behaviour shares and
assembly-excluded results, with occurrence-weighted presentation tables and
sealed encoder-width histograms. It checks final/stage/summary hashes against the
retained verification receipt before reading them. It loads no models or new
inputs, changes no trial artifact and applies no new gate.

Final result/code validation, 2026-10-03 14:15 PDT: the full suite including both
independent verifier test files completed exit 0: 2892 passed, 40 skipped,
7 deselected, 1 xfailed, 8 existing fork warnings, 169.91 seconds. Whole-tree
Ruff lint/format and whitespace checks passed. Publication checks independently
compared all six copied source summaries and share denominators, label totals
and typed-width histograms against hash-pinned completed artifacts. This adds
no trial, inference or comparative-clock measurement and does not replace
independent other-family result/code review.
