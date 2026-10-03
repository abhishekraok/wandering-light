# Amendment B — runner/verifier repair before declared Gate A

The program-synthesis janthu (e-den, Codex) wrote this, 2026-10-03 PDT.
Actual original pre-gate e36b89f review: exit0/CHANGES_REQUIRED. No declared Gate A
attempt or manifest exists. This amendment changes software controls/output,
not the retention rule, manual Task/scorer inputs,455 IDs or case expectations.

B1's byte serialization was already repaired atf05364c. This revision directly
tests strict JSON through search helpers in all three categories plus typed
input, GC/parameter, arithmetic and scorer API helpers. Scripted scorers only;
no model, corpus, archive, checkpoint, held-out or real allocation work.

N1: actual successful high-effort Claude pre-gate result must begin with exactly
`VERDICT: PRE-GATE CODE PASS`. A quoted/ambiguous PASS elsewhere cannot grant it.
Source, actual successful exit, no CLI error, metadata and report/log hashes
remain checked before attempt reservation. Synthetic parsing fixtures test this
helper only, never invoke run_gate with a fabricated successful receipt.

N2: owned temporary output is removed on deadline/write failure; a preexisting
unowned temporary is never removed. Final publication remains atomic/bounded.
N3: all9 imported experiment/production modules must resolve to their intended
exact source path AND match that file's hash; actual origins recorded in final.
N4: independent expansion verification counts a parent entered just before an
op0 budget/deadline check, including a node-boundary stop. New DEVELOPMENT-ONLY
regressions fix budgets193/194/339/340/395/396 on unchanged T-list/script against
imported production. These extra control checks do not change the declared
Gate A resource IDs/budgets or authorize another gate invocation.
N5: independent initial seen is derived from the fixed manual inputs; expected
retained states must match observer retention and the next expanded beam, even
when the next layer returns early before retention. Mutation tests must reject
missing initial seen, inconsistent next beam and altered retention event.
N6: manifest stores explicit expected result fields/counters from production
reference or the frozen hand outcome plus independently recomputed low-level
attempts, and full expected typed order/parents/anchor/stop/verdicts from the
independent table oracle. Exception/GC/arithmetic/API controls get explicit
expected values. Labels remain IDs, not a substitute for numeric expectations.
N7: current STATUS records implementation and the actual code-review refusal;
old frozen review tree is historical and remains untouched.

Development tests may execute the fixed manual cases/control regressions;
the declared complete all455-ID Gate A remains forbidden until a fresh exact-
source high-effort PRE-GATE CODE PASS. That pass permits software Gate A only,
not landing, any real threshold/panel/model/campaign or BEST promotion. Separate
independent final code/result PASS remains required before landing the evidence.
