# branch-and-check (#164), O1-A archive audit

Source telemetry is [archive_audit.json](archive_audit.json); the measurement row is in `docs/RESULTS_DETAILED.md`. The preregistration and interpretation are on [the issue](https://github.com/abhishekraok/wandering-light/issues/164#issuecomment-5965483510).

Reproduce from the main checkout's archive, without any training or held-out data:

```bash
uv run python experiments/branch_check/o1_archive.py --archive /home/abhishekrao/repos/wandering-light/checkpoints/short-segment-search/short-segment-difference-20260923/archive/train-capped-s0-e3-segment-k16-d8.jsonl.gz --out /path/to/new/audit.json
```

The source path and SHA256, replay-code revision and per-task audit are in the JSON. The archived source is not copied into this report. Candidate states are genuinely closed on every shown example; repeated states use their last occurrence; goal states are excluded. Recorded suffix lengths are upper bounds rather than optimal distances. Fitting shown examples does not establish behaviour equivalence. No ranker was trained and no solver was changed by this audit.

[Independent high-effort review](review.md) passed on `d9101b7` and reproduced the audit. Its nonblocking caveats constrain the next candidate builder: a recorded path loop can yield a shorter continuation by splicing equal states, and genuine-closure candidates do not coincide with the combinator-cut states the segment search ranks. Before training, register a prefix-controlled/cross-path panel and its tested continuation catalogue; plain within-witness ordering has a perfect prefix-position baseline. The committed audit retains its original recorded-suffix definition and counts.
