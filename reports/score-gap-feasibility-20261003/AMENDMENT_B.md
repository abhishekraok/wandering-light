# Amendment B — open refusal strings, before data

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
Amended frozen51e8514 review returned actual CHANGES_REQUIRED: B1-R showed
304 distinct synthetic refusal strings even after stripping the example index,
so the proposed uint8 reason dictionary was unsafe. The report/receipt remain
unchanged. No data attempt, selection, weight/model access or inference occurred.

Drop stored reasons entirely. Attempts are fixed19-byte <BBBQQ records:
zero-based depth/node/op uint8, fuel/stepped uint64. Exact successor/refusal
strings are independently re-derived from typed parent plus op using pinned
step_raw, DEFAULT_FUEL_LIMIT and initial depth0. The example index is stepped,
not parsed text. This retains arbitrary nested/count-bearing refusals without
aliasing or dictionary overflow. Worst11,987,136 attempts cost227,755,584 bytes;
all other caps remain unchanged. Reject range violations and truncated records.
No cap retry or outcome-dependent modification follows.

Reconcile whitelist towards implementation: never read task program or hidden
fields. Pin prerequisite order: completeness/provenance/resources, support in
both halves/all seeds, selection headroom, selection grid, receipt lock, chosen
confirmation profile. Retain all eight grid profiles and descriptives; report
confirmation prefix protection beside a failed conjunction. Full scored-matrix
validity, typed state bounds, complete observer callbacks, atomic receipts and
independent serialized-output verification remain unimplemented obligations.
A fresh DESIGN PASS accepts only design/synthetic foundations; complete exact-
code PASS is separately required before one data attempt.
