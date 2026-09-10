---
configs:
- config_name: default
  data_files:
  - split: train
    path: stack_pbe_world_v1_train.jsonl.gz
---

# `stack_pbe_world_v1`

World-authored stack-programming-by-example corpus: MBPP and LeetCode-easy
behaviours with `mark-28` witnesses authored in macro form. Train-only; eval
stays `stack_pbe_v4`. The tracked `manifest.json` pins the payload and records
its provenance. See stack-pbe-world-v1 (#120).
