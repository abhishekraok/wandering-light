# Deep-corpus-v2 scratch feasibility

Issue [#54](https://github.com/abhishekraok/wandering-light/issues/54) stopped at its predeclared one-RTX-4090
feasibility gate. The exact 360M arm fits in physical VRAM with required
non-reentrant gradient checkpointing, but its three learning-rate pilots plus
three-epoch final project to **17.023 hours**. That exceeds the 16-hour hard
limit, so the primary three-arm comparison was not launched.

No learning-rate pilot, final training run, validation generation, W&B run or
report, or test opening occurred. No model weights were published. Therefore
there is no solve-rate result or scratch-scaling conclusion; issue #54's model
question remains unanswered under this compute constraint.

## Frozen inputs

- Corpus: 313,438 newly generated, witness-reexecuted records from 2,400 roots;
  no root or task overlap with `deep-corpus-v1`. Manifest digest
  `sha256:a3ba242346f5c43b081e498984380d471a67ee4b91c7c094612100622791916e`.
- Public payload: [`abhishekraok/wandering-light-deep-corpus-v2`](https://huggingface.co/datasets/abhishekraok/wandering-light-deep-corpus-v2/tree/8a69a300dddbd2a65e107d0459c800e3f8429014)
  at revision `8a69a300dddbd2a65e107d0459c800e3f8429014`.
- Shared discovery-only tokenizer: vocabulary 4,196; 245,554 retained discovery
  rows and seven disclosed over-length exclusions; artifact digest
  `sha256:128279939a1e8e8acf533a474998dcf2a508726767ecbe2043cfed149f0be1ef`.
- Frozen protocol: source commit
  `30e1e1ef12f270127cd081a4a9c3dde18eb1c9a8`; protocol digest
  `sha256:d64ac7a8c7d33274424ff1ada3b13d0d42b3d808df5da4ef4a18a892757f6c36`.
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
| 10M scratch | 10,080,896 | off | 3.54 GiB | 28.363 s | 75.275 s | 2.326 h | direct |
| OPT-style 125M scratch | 125,265,408 | off | 8.66 GiB | 45.438 s | 117.635 s | 3.718 h | direct |
| SmolLM2-style 360M scratch | 359,951,040 | non-reentrant, required | 6.39 GiB | 217.414 s | 323.278 s | **17.023 h** | **blocked: over 16 h** |

The projection covers all three quarter-epoch pilots, the three-epoch final,
every scheduled validation and weights-only checkpoint write, and a fixed 15%
factor for setup, online W&B, best-checkpoint reload, and final serialization.
Physical peaks are conservative whole-GPU `nvidia-smi` samples because WSL did
not expose a compute-process row.

Checkpointing was required, not elective. The preceding exact
no-checkpoint 360M longest-batch proof reported 27,468,560,896 torch-allocated
bytes against 25,756,696,576 physical bytes, with 25,291,653,120 bytes observed
on the physical GPU. WSL oversubscription let the operation complete, but the
gate rejected it as an invalid physical fit. That sealed precursor is
[`feasibility_manifest_uncheckpointed_rejected.json`](feasibility_manifest_uncheckpointed_rejected.json),
artifact digest
`sha256:9d7441faba83291cbb067548e8b37900761fe85b48fd1ff0c1db2e9aac75fe7f`.

The sealed evidence is
[`feasibility_manifest.json`](feasibility_manifest.json), logical artifact
digest
`sha256:83b42594f21fb573616e57ea48d775ee672300e27c32be9000f7e03f03ee1806`.
Validate its content seal and recomputed gate with:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
  .venv/bin/python -c \
  'from pathlib import Path; from experiments.probe_deep_corpus_v2_scratch import load_probe_manifest; m = load_probe_manifest(Path("reports/deep-corpus-v2-scratch-20260826/feasibility_manifest.json")); assert m["status"] == "complete"; assert m["launch_gate"]["all_arms_authorized"] is False; print(m["artifact_digest"], m["launch_gate"]["blocked_arm_ids"])'
```
