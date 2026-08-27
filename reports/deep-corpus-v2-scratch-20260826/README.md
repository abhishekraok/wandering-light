# Deep-corpus-v2 scratch feasibility

Issue [#54](https://github.com/abhishekraok/wandering-light/issues/54) stopped at its predeclared one-RTX-4090
feasibility gate. The exact 360M arm fits in physical VRAM with required
non-reentrant gradient checkpointing, but its three learning-rate pilots plus
three-epoch final project to **17.282 hours**. That exceeds the 16-hour hard
limit, so the primary three-arm comparison was not launched.

No learning-rate pilot, final training run, validation generation, W&B run or
report, or test opening occurred. No model weights were published. Therefore
there is no solve-rate result or scratch-scaling conclusion; issue #54's model
question remains unanswered under this compute constraint.

## Frozen inputs

- Corpus: 313,438 newly generated, witness-reexecuted records from 2,400 roots;
  no root or task overlap with `deep-corpus-v1`. Manifest digest
  `sha256:37543f84d0bb376f285c1639ee6fe98bbdc693e41501741f5562424d3de1e4ee`.
  The tracked manifest is 41,828 bytes and pins the detached 2,400-root
  expansion telemetry as
  `sha256:168671e01e5a711a1f4a39471b2b9a14f2bd887eb3c5a418a0f3873c5fa95326`.
- Public payload: [`abhishekraok/wandering-light-deep-corpus-v2`](https://huggingface.co/datasets/abhishekraok/wandering-light-deep-corpus-v2/tree/36753a013f3a4e965892b33bc44a8cc0fd567d99)
  at revision `36753a013f3a4e965892b33bc44a8cc0fd567d99`.
- Shared discovery-only tokenizer: vocabulary 4,196; 245,554 retained discovery
  rows and seven disclosed over-length exclusions; artifact digest
  `sha256:d4736f3a1623a89945daeaa8495250f28b8b47725d60f4d678fa9a0c1d8d5f66`.
  Its tracked manifest is 39,103 bytes; full token-length histograms and the
  seven per-record exclusions are detached as `tokenizer_detail.json`, digest
  `sha256:ab3c92afd67ea2c7821d576dc2123267d2ba37dea50bc48cd861ad885a0b7b46`,
  at Hub revision `41de16325415207eead597b266f2755bcb564659`.
- Frozen protocol: source commit
  `32012c747cd6480416656889453e5d46ec7784b6`; protocol digest
  `sha256:4d1c45355d59e0879bccf4ae5e80bd7721af9d4dc20a5e8059d91a9b5942a45d`.
- Frozen 96-task cohorts: validation
  `sha256:6b04ecc02936d24483164adbf1d74939355ec9afaf529d3221b161922dadae2e`;
  unopened test
  `sha256:2f664a13c3974e8f573eb0a79f383b7563420306ad6b1646d66476c0e593f223`.

The corpus distance histogram is 18,899 / 47,419 / 48,000 / 48,000 /
48,000 / 47,960 / 47,960 / 7,200 for certified distances 1 through 8.

## Measured gate

Every arm was constructed with `AutoModelForCausalLM.from_config`; no
pretrained weights were loaded. Each measurement used bf16, batch size 8,
maximum length 2,048, `PYTHONHASHSEED=0`, and
`CUBLAS_WORKSPACE_CONFIG=:16:8` on one physical RTX 4090. The longest batch
contained the same eight retained sequences for every arm and padded to 1,956
tokens.

| arm | exact parameters | checkpointing | physical peak | 500 steps | full validation | projected 3 pilots + final | gate |
|---|---:|---|---:|---:|---:|---:|---|
| 10M scratch | 10,080,896 | off | 5.31 GiB | 27.381 s | 76.341 s | 2.258 h | direct |
| OPT-style 125M scratch | 125,265,408 | off | 10.41 GiB | 48.487 s | 119.105 s | 3.947 h | direct |
| SmolLM2-style 360M scratch | 359,951,040 | non-reentrant, required | 8.13 GiB | 220.421 s | 334.810 s | **17.282 h** | **blocked: over 16 h** |

The projection covers all three quarter-epoch pilots, the three-epoch final,
every scheduled validation and weights-only checkpoint write, and a fixed 15%
factor for setup, online W&B, best-checkpoint reload, and final serialization.
Physical peaks are conservative whole-GPU `nvidia-smi` samples because WSL did
not expose a compute-process row. A first attempt was discarded atomically
after one `nvidia-smi` sample failed with WSL `Errno 14`; the sealed retry
completed all arms with 1,818 / 2,188 / 4,390 physical-memory samples.

Checkpointing was required, not elective. The preceding exact, pre-compaction
no-checkpoint 360M longest-batch proof reported 27,468,560,896 torch-allocated
bytes against 25,756,696,576 physical bytes, with 25,291,653,120 bytes observed
on the physical GPU. WSL oversubscription let the operation complete, but the
gate rejected it as an invalid physical fit. Compacting provenance changed the
logical corpus/tokenizer identities but not their payload, token IDs, longest
batch, model configuration, or weights; this historical precursor explains the
locked checkpointing choice but is not part of the current launch identity. It is
[`feasibility_manifest_uncheckpointed_rejected.json`](feasibility_manifest_uncheckpointed_rejected.json),
artifact digest
`sha256:9d7441faba83291cbb067548e8b37900761fe85b48fd1ff0c1db2e9aac75fe7f`.

The sealed evidence is
[`feasibility_manifest.json`](feasibility_manifest.json), logical artifact
digest
`sha256:6b43b37535475cef0d96fd07591a74e7c5ecfc04802918dfb49cf9ecb5891271`.
Validate its content seal and recomputed gate with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  .venv/bin/python -c \
  'from pathlib import Path; from experiments.probe_deep_corpus_v2_scratch import load_probe_manifest; m = load_probe_manifest(Path("reports/deep-corpus-v2-scratch-20260826/feasibility_manifest.json")); assert m["status"] == "complete"; assert m["launch_gate"]["all_arms_authorized"] is False; print(m["artifact_digest"], m["launch_gate"]["blocked_arm_ids"])'
```
