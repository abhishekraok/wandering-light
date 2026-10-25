# Large-scale stack PBE data

Generate typed compositions and Codex-authored programs, compare independent Python references with stack witnesses, then merge distinct tested behaviors. Candidate counts, accepted fingerprints, semantic algorithm counts, and witness lengths are different quantities.

The author supervisor uses subscription-billed Codex with model API keys removed. It stops when weekly quota remaining is below 5% or above 95%, or at the configured Pacific deadline. It pauses launches when quota receipts are stale. Optional `author-notes.txt` in the run directory guides subsequent batches.

```bash
RUN=/home/abhishekrao/repos/wandering-light/checkpoints/pbe-scale-20261024
uv run python -m experiments.pbe_scale.campaign --output-dir "$RUN" --deadline 2026-10-25T00:00:00-07:00 --concurrency 8
```

`generate.py` builds compositions of typed stages; `generate_families.py` adds explicit sequence algorithms. Both emit JSONL candidates with an independent Python reference. `anchors.py` supplies simple witnesses so a long computation of a simple function can be replaced during merging.

Verification must run within the repository's execution cap:

```bash
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- uv run python -m experiments.pbe_scale.verify_batch --input "$RUN/structured-candidates.jsonl" "$RUN/family-candidates.jsonl" "$RUN/structured-long-candidates.jsonl" --output-dir "$RUN/v3-round-000" --validation-root "$RUN/validated-v3" --workers 8 --seed 20261024
```

Each round freezes its inputs. Later rounds use a new round directory and the same validation root. Verify anchors separately under the same probe policy, then merge only matching final-policy directories:

```bash
uv run python -m experiments.pbe_scale.merge "$RUN"/validated-v3/shard-* "$RUN/anchor-validation-v3" --output "$RUN/corpus-v3"
```

V3 uses 192 shared edge/random inputs plus 16 candidate-specific inputs for list tasks. Integer inputs are exhaustive over −10…10 (21 single integers or 441 integer pairs). Six shown examples vary in outputs; the rest remain hidden. Deduplication compares shared-probe output fingerprints and retains the shortest *verified* representative available. This is not a proof of minimal program length. Reference and witness execution have per-candidate CPU, wall-time, output, and memory bounds. Exports preserve journals, rejected candidates, source hashes, configuration, and provenance.

The corpus is training-only. Existing benchmark overlap has not been audited; no generalization claim follows from dataset construction. Program length is an authored upper bound, and macro expansion can dominate it.

Prepare bounded-memory training rows and train at fixed epochs:

```bash
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -- uv run python -m experiments.pbe_scale.train prepare --input "$RUN/corpus-v3/tasks_train.jsonl.gz" --output-dir "$RUN/packed32" --prefixes-per-task 32 --seed 0
uv run python -m experiments.pbe_scale.train train --data "$RUN/packed32" --output-dir "$RUN/policies/new-1m-s0" --arm 1m --epochs 3 --seed 0 --device cuda
```

Use `--prefixes-per-task 0` for every witness prefix. Preparation records state/target truncation and selected/full prefix counts. Training uses the existing policy, saves weights at every epoch, records personal W&B lineage, and reports final-epoch weights; its small training-row diagnostic does not select a checkpoint.
