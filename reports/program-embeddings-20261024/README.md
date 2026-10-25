# Program / I/O embeddings

![Held-out retrieval and executable I/O selection](metrics.png)

| Held-out test | Trained | Random | Lexical |
|---|---:|---:|---:|
| Alternate-text recall@1 / @10, 65 behaviors | **41.3% / 72.3%** | 22.3% / 45.9% | 17.9% / 41.0% |
| I/O hidden success, grouped holdout | **12/128** | 0/128 | — |
| I/O hidden success, all official v4 | **10/45** | 0/45 | — |
| I/O hidden success, eval-linked holdout | 0/4 | 0/4 | — |

Alternate-text ranking improves over both controls; I/O retrieval remains limited. Program metrics weight behaviors equally. I/O grades the first shown-passing top-10 candidate on hidden examples; no unknown outcomes. Tiny official/eval-linked alternate-text strata (3/2 behaviors) stay separate in [metrics.json](metrics.json).

On 47 behaviors with two independently authored implementations, top-10 recall is **66.0%**, versus 39.4% random and 35.1% lexical (descriptive subset). Staged queries reach 88.9% versus 66.7% random on 18 queries: useful tolerance to staging, with misses remaining.

Programs in the repo's `mark-28` stack language and I/O examples map to 128-dimensional unit vectors: **387,344 parameters**, **1.5 MB** of weights; **12,837 programs**, **14.4 MiB** index. Warm CPU I/O queries: median **5.09 ms**, p95 **8.91 ms** (45 queries, four threads; encoder/scan/top-k, excluding startup/verification). Encoding performs no execution.

```python
import torch
from wandering_light.program_embedding import ProgramEmbedding, ProgramIndex

run = "/home/abhishekrao/repos/wandering-light/checkpoints/semantic-embeddings-20261024/aligned-s0"
model = ProgramEmbedding.load(f"{run}/best.pt")
with torch.inference_mode():
    # fold accumulates add over the input list: its sum.
    programs = model.encode_programs(["0 [ add ] fold", "0 [ swap add ] fold"])
    similarity = (programs[0] @ programs[1]).item()
    examples = model.encode_io([[("[ (2 3) ]", "[ 5 ]")]])
index = ProgramIndex.load(f"{run}/library.pt")
ids, scores = index.search_io([("[ (2 3) ]", "[ 5 ]")], k=10)
neighbors = [index.programs[i] for i in ids.tolist()]
# Or precompute your own bank once:
small = ProgramIndex(model).fit(["0 [ swap add ] fold", "nil [ cons ] fold"])
ids, scores = small.search_program("0 [ add ] fold", k=2)
```

Programs use `mark-28`, the 28-operation basis. Outer brackets delimit a stack; `(2 3)` is one list value. Example order is invariant.

**Protocol.** One-layer bidirectional GRUs, hidden 128/token 48; AdamW, batch 64, temperature 0.07, weight decay 0.01, gradient cap 1, seed 0. Joint: 1,500 steps, rate 0.001, half-batch alternate-text sampling, equal I/O↔program/program↔program losses. Program-only: 500 steps, rate 0.0002, alternate-text sampling only. Frozen-program alignment: 500 steps, rate 0.001, I/O↔program only, half-batch alternate-text sampling. Validation selected **aligned step 500** before test. Five visible/16 hidden examples maximum. Configuration/W&B lineage: `provenance.json`.

**Data/caveats.** 9,702 fit / 1,224 validation / 1,290 test behaviors; five insufficient train records omitted. Exact programs merge with aliases; shared templates/alternative skeletons stay in one split. All 45 official v4 behaviors remain held out. Hidden inputs never enter visible views. Contrived/precondition-flagged rewrites are excluded. All 1,101 retained alternate programs pass every stored merged-domain example. Excluded bad Opus `mbpp-876` SHA-256: `0f80c1f45e2d148534defa33e2b6efce5fddaa9dc2bf6837ce97e62fad537a43`.

Peer snapshots are immutable under `/home/abhishekrao/repos/wandering-light/checkpoints/semantic-embeddings-20261024/data/{validated-v2,author-validation-v2}.jsonl`; canonical v4 and reviewed witness sources/digests are recorded in provenance. This is one seed. Equivalence is finite-domain evidence. **Held-out answers are present in the bank**: ranking success does not establish synthesis or coverage by a training-only library.

Related [staged-witness-reauthoring (#155)](https://github.com/abhishekraok/wandering-light/issues/155) found a retrieval gain; its [corrected fixed-reference comparison](../../docs/STATUS.md#superseded) leaves assembled behavior counts level.

```bash
uv run python reports/program-embeddings-20261024/plot_results.py --run-dir /home/abhishekrao/repos/wandering-light/checkpoints/semantic-embeddings-20261024/aligned-s0
uv run pytest tests/test_program_embedding{,_data,_train,_artifacts}.py -q
```

Evaluation is reusable through `wandering_light.program_embedding_eval.evaluate`;
the existing `program_embedding_train.evaluate`, `retrieval_metrics` and
`lexical_vectors` imports remain available. Ranking, tie order, seeded query
sampling, complete banks and execution grading are unchanged. `evaluate` adds
`evaluation_config` with the effective options, including defaults.

New `evaluation.json`, `untrained.json`, `initial.json` and `validation.json`
reports embed `provenance`: evaluator commit/dirty flag, actual model config,
arguments, data manifest/digests, seed, W&B URL, and `evaluation_source`.
`checkpoint_path`/`checkpoint_digest` identify the evaluated checkpoint, with
its complete training lineage in `checkpoint_metadata`; random initialization
records null checkpoint fields. Parent `initial_checkpoint_*` fields are retained
when evaluating a newly trained best checkpoint. Each report snapshots this
metadata independently of `provenance.json` and records effective evaluation
settings separately from training arguments.

New `history.json` files use a version-1 envelope with `history` (the unchanged
step/loss/validation entries) and a detached training-run `provenance` snapshot.
`program_embedding_artifacts.load_training_history(path)` returns
`(entries, provenance)` for new files and `(entries, None)` for legacy bare lists,
including empty histories. Historical history files are not rewritten.

Existing weights-only model (v2) and persistent index (v1) formats are unchanged.
Historical evaluations, published metrics and figures are preserved. Plotting
accepts older reports without embedded provenance and records their evaluation
source as unknown with a caveat; it does not infer evaluator provenance from the
library's training commit. Missing metadata never prevents loading or evaluation.
