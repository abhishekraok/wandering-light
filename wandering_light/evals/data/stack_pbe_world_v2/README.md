---
configs:
- config_name: default
  data_files:
  - split: eval
    path: stack_pbe_world_v2_eval.jsonl.gz
  - split: train
    path: stack_pbe_world_v2_train.jsonl.gz
---

# `stack_pbe_world_v2`

World-authored stack-programming-by-example corpus: codearc, leetcode, mbpp
behaviours with `mark-28` witnesses authored in macro form. Split policy: by-behaviour-hash eval_share=0.1.
The tracked `manifest.json` pins the payload and records its provenance.
See https://github.com/abhishekraok/wandering-light/issues/127.
