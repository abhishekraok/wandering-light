"""Independent high-effort Claude review of the completed benchmark update."""

import argparse
import json
from pathlib import Path

from wandering_light.solver import ClaudeExecTokenGenerator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--confirming", action="store_true")
confirming = parser.parse_args().confirming

files = [
    ".gitattributes",
    "README.md",
    "docs/RESULTS_DETAILED.md",
    "experiments/stack_pbe_baseline.py",
    "wandering_light/stack_pbe_eval.py",
    "reports/stack-pbe-v4-effort-sweep-20260927/README.md",
    "reports/stack-pbe-v4-effort-sweep-20260927/analyse.py",
    "reports/stack-pbe-v4-effort-sweep-20260927/plot.py",
    "reports/stack-pbe-v4-effort-sweep-20260927/summary.json",
    "reports/stack-pbe-v4-effort-sweep-20260927/table.md",
    "reports/stack-pbe-v4-sonnet55-20260928/README.md",
    "reports/stack-pbe-v4-sonnet55-20260928/run.sh",
    "reports/stack-pbe-v4-sonnet55-20260928/preflight.py",
    "reports/stack-pbe-v4-sonnet55-20260928/analyse.py",
    "reports/stack-pbe-v4-sonnet55-20260928/validation.json",
    "reports/stack-pbe-v4-sonnet55-20260928/table.md",
]
files += [
    f"reports/stack-pbe-v4-sonnet55-20260928/sonnet55_{e}_v4_sample40.json"
    for e in ("low", "medium", "high", "xhigh", "max")
]
prompt = """Review only. You are the independent reviewer from the other agent family.
A Codex agent was asked to benchmark Sonnet 5.5 and add it to the README plot.
The unchanged stack_pbe_baseline harness makes one generator call per task,
on the fixed seed-17 40-task v4 eval sample, with canonical held-out grading.
Existing plot costs estimate a fixed 1300-token prompt plus recorded output,
including thinking. Official Sonnet 5.5 price is $2/$10 per million tokens:
https://www.anthropic.com/claude-sonnet-5-5 (verified by the author and a reviewer).
Check measurement integrity, arithmetic, agreement of raw results and claims,
and plot/report integration. Check the four refusal failures and the adapter's
derived tool_calls field (extra-turn/permission proxy, not an execution trace).
Do not treat pre-existing corpus limitations or
pre-existing results as new defects. Return PASS if no material findings;
otherwise list actionable findings with priority and file/line references.
You have no tools; all review evidence is included below.
"""
if confirming:
    prompt += """This is the single confirming review after your initial PASS.
Check the changes and responses to the findings, and return PASS if no
material issue remains. The PNG was visually checked by GPT-6 Astra/high.
"""
    files += [
        "reports/stack-pbe-v4-sonnet55-20260928/review.md",
        "reports/stack-pbe-v4-sonnet55-20260928/review-response.md",
    ]
for file in files:
    prompt += f"\n--- {file} ---\n{(ROOT / file).read_text()}\n"
solver = (ROOT / "wandering_light/solver.py").read_text()
start = solver.index("CLAUDE_EXEC_DROPPED_ENV =")
stop = solver.index("def json_dumps", start)
prompt += (
    "\n--- wandering_light/solver.py: ClaudeExecTokenGenerator ---\n"
    + solver[start:stop]
)
generator = ClaudeExecTokenGenerator("claude-opus-5-5", effort="high", timeout=1800)
answer = generator.generate(prompt)
assert generator.last_meta["model"] == ["claude-opus-5-5"]
stem = "review-confirming" if confirming else "review"
(HERE / f"{stem}.md").write_text("Claude Opus 5.5, high effort\n\n" + answer + "\n")
(HERE / f"{stem}.json").write_text(json.dumps(generator.last_meta, indent=2) + "\n")
print(answer)
