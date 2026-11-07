"""Re-ask each call that failed with a service error, once, and splice it in.

A dropped connection, or a 900 s deadline on gemini-4-argon at low (which does
not think and otherwise answers in under a minute), is the service, not the
model. Each failed candidate gets one more call; if that fails too it stays
unsolved. The shard file keeps the first error under `retry.first_errors`, and
the untouched original is saved alongside as `*.orig.json`.
Usage: retry_failed.py [OUT]. Needs GEMINI_API_KEY set; grading runs programs,
so run it under the #172 memory cap.
"""

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments"))
from stack_pbe_baseline import llm_prompt, parse_response

from wandering_light.solver import GeminiTokenGenerator
from wandering_light.stack_pbe import load
from wandering_light.stack_pbe_eval import best

OUT = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else Path.home() / "repos/wandering-light/checkpoints/gemini-sweep-20261106"
)


def ask(model: str, effort: str, prompt: str) -> tuple[str, dict | None, str | None]:
    generator = GeminiTokenGenerator(model, effort=effort)
    try:
        return generator.generate(prompt), generator.last_meta, None
    except Exception as error:  # broad on purpose, as solve_with_llm
        return "", generator.last_meta, type(error).__name__


def main() -> None:
    tasks = None
    jobs = []
    for path in sorted(OUT.glob("*_shard?.json")):
        run = json.loads(path.read_text())
        failed = [c for c in run["candidates"] if c.get("errors") and "retry" not in c]
        if not failed:
            continue
        tasks = tasks or load(run["corpus"])
        jobs += [(path, run, c, tasks[c["task_index"]]) for c in failed]
    print(f"{len(jobs)} failed calls to retry", file=sys.stderr)
    with ThreadPoolExecutor(len(jobs) or 1) as pool:
        answers = list(
            pool.map(
                lambda j: ask(j[1]["model"], j[1]["effort"], llm_prompt(j[3])), jobs
            )
        )
    touched = {}
    for (path, run, cand, task), (text, meta, error) in zip(jobs, answers, strict=True):
        score = best([parse_response(text)], task)
        cand["retry"] = {"first_errors": cand.pop("errors")}
        if error:
            cand["retry"]["error"] = error
        if meta:
            cand.setdefault("generation", []).append(meta)
        cand["candidate"] = " ".join(score.program)
        cand["verdict"] = (
            "solved" if score.solved else "memorised" if score.shown else "failed"
        )
        print(f"{path.name} {cand['behaviour']}: {cand['verdict']} {error or ''}")
        touched[path] = run
    for path, run in touched.items():
        backup = path.with_suffix(".orig.json")
        if not backup.exists():
            shutil.copy(path, backup)
        run["solved"] = sum(c["verdict"] == "solved" for c in run["candidates"])
        run["memorised"] = sum(c["verdict"] == "memorised" for c in run["candidates"])
        path.write_text(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
