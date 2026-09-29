"""Verify source, pinned corpus, sample order and model access before the sweep."""

import json
import random
from pathlib import Path

import wandering_light
from wandering_light.corpus_hub import fetch_corpus
from wandering_light.solver import ClaudeExecTokenGenerator
from wandering_light.stack_pbe import load

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
assert Path(wandering_light.__file__).resolve().is_relative_to(ROOT)
corpus = ROOT / "wandering_light/evals/data/stack_pbe_v4"
fetch_corpus(corpus / "manifest.json")
all_tasks = load(corpus)
index_of = {id(t): i for i, t in enumerate(all_tasks)}
tasks = [t for t in all_tasks if t.split == "eval"]
random.Random(17).shuffle(tasks)
indices = [index_of[id(t)] for t in tasks[:40]]
reference = json.loads(
    (
        ROOT / "reports/stack-pbe-v4-new-models-20260926/opus55_low_v4_sample40.json"
    ).read_text()
)
assert indices == [c["task_index"] for c in reference["candidates"]]
generator = ClaudeExecTokenGenerator("claude-sonnet-5-5", effort="low", timeout=120)
answer = generator.generate("Reply with the single word OK.")
meta = generator.last_meta
assert meta["model"] == ["claude-sonnet-5-5"], meta
assert not meta["tool_calls"] and not meta["text_tool_call"], meta
(HERE / "probe.json").write_text(
    json.dumps({"answer": answer, **meta}, indent=2) + "\n"
)
print(f"Source: {wandering_light.__file__}")
print(f"Pinned corpus verified; same {len(indices)} task indices as Opus 5.5.")
print(f"Probe: {answer!r}; served {meta['model']}; tools {meta['tool_calls']}.")
