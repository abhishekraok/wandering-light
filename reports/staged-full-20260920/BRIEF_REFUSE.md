# Brief: re-fuse the pilot's staged witnesses

staged-witness-reauthoring (#155). The pilot staged 20 behaviours; each needs a **re-fused twin**.
Your behaviours are in the `assign_r<n>.txt` you were given. For each, find its staged program in
`reports/staged-pilot-20260920/programs_{a,b,c,d}.jsonl` (field `program`, with the author's `notes`).

Read the sections "The machine", "Commands" and "The re-fused control" of
`reports/staged-full-20260920/BRIEF.md`. Write **the same algorithm as the staged program** with
consecutive cycles merged back into as few quotations as you reasonably can. No size rules apply; it
must verify (`--try <behaviour> "<program>"` prints `"verified": true`; ignore the verdict) and should
execute fewer times than the staged program. Do not return to the original fused witness's algorithm.
If nothing merges, repeat the staged program and say so.

Write `reports/staged-full-20260920/programs_r<n>.jsonl`, one line per behaviour:

```json
{"behaviour": "...", "program": "<the pilot's staged program, copied unchanged>", "refused_program": "...", "status": "pilot", "notes": "what you merged"}
```

Finish by running `--file` on it; every row must show `"verified": true` for both. Do not edit any
other file, do not commit, push, post to GitHub, or delegate to other agents.
