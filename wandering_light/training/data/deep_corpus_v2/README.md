---
configs:
- config_name: default
  data_files:
  - split: discovery
    path: discovery.jsonl.gz
  - split: validation
    path: validation.jsonl.gz
  - split: test
    path: test.jsonl.gz
---

# `deep_corpus_v2`

Certified-shortest-path training corpus. The tracked `manifest.json` pins the
payloads, detached expansion telemetry, and generation provenance.
