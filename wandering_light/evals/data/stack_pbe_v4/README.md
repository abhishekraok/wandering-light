---
configs:
- config_name: default
  data_files:
  - split: train
    path: stack_pbe_v4_train.jsonl.gz
  - split: eval
    path: stack_pbe_v4_eval.jsonl.gz
---

# `stack_pbe_v4`

Behaviour-first stack-programming-by-example corpus split on program skeleton.
The tracked `manifest.json` pins the payloads and records their provenance.
