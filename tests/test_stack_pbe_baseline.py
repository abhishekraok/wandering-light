"""Baselines see only the shown half; the grader then checks the other one."""

import dataclasses
import gzip
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from stack_pbe_baseline import (
    bfs_solve,
    llm_prompt,
    main,
    parse_response,
    random_solve,
)

from wandering_light.stack_pbe import BY_NAME, generate
from wandering_light.stack_pbe_eval import grade


def task_for(name, seed=0):
    (task,), _ = generate(1, seed, catalogue=[BY_NAME[name]])
    return task


def test_bfs_finds_shortest_fit():
    task = task_for("add_pair")
    assert bfs_solve(task, budget=1000) == ("add",)
    assert grade(bfs_solve(task, budget=1000), task).solved


def test_bfs_finds_multi_op_program():
    task = task_for("head")
    found = bfs_solve(task, budget=5000)
    assert found is not None and len(found) <= 3
    assert grade(found, task).shown


def test_bfs_uses_only_shown_examples():
    task = dataclasses.replace(
        task_for("add_pair"),
        inputs=("[ 1 1 ]", "[ 0 1 ]"),
        outputs=("[ 2 ]", "[ 1 ]"),
        held_out_inputs=("[ 3 4 ]",),
        held_out_outputs=("[ 99 ]",),
    )
    found = bfs_solve(task, budget=1000)
    score = grade(found, task)
    assert score.shown and not score.held_out


def test_bfs_respects_budget_and_depth():
    task = task_for("head")
    assert bfs_solve(task, budget=0) is None
    assert bfs_solve(task, budget=1) is None
    assert bfs_solve(task, budget=5000, max_depth=1) is None


def test_random_solve():
    task = task_for("add_pair")
    found = random_solve(task, budget=2000, rng=random.Random(0))
    assert found is not None and grade(found, task).shown
    assert random_solve(task, budget=0, rng=random.Random(0)) is None


def test_random_solve_is_seeded():
    task = task_for("add_pair")
    first = random_solve(task, 500, random.Random(5), max_length=3)
    assert first == random_solve(task, 500, random.Random(5), max_length=3)
    assert first is None or len(first) <= 3


def test_llm_prompt_shows_ops_brackets_and_only_shown_pairs():
    task = task_for("sum")
    prompt = llm_prompt(task)
    assert "fold" in prompt and "( lst acc q -- x )" in prompt
    assert "[ 1 add ]" in prompt
    for given, wanted in zip(task.inputs, task.outputs, strict=True):
        assert f"{given} -> {wanted}" in prompt
    for hidden in task.held_out_inputs:
        assert f"{hidden} ->" not in prompt


def test_parse_response():
    assert parse_response("") == ""
    assert parse_response("add") == "add"
    assert parse_response("Reasoning...\n\n0 [add] fold\n") == "0 [ add ] fold"
    assert parse_response("<think>dup mul</think>\n`[add] map`") == "[ add ] map"
    assert parse_response("Here:\n```\n0 [ add ] fold\n```\n") == "0 [ add ] fold"
    assert parse_response("```text\nfoo [bar]\n```") == "foo [ bar ]"
    assert parse_response("<think>x</think>") == ""
    assert parse_response("\n\n  \n") == ""


def test_main_reports(tmp_path, capsys):
    tasks, _ = generate(1, 0, catalogue=[BY_NAME["add_pair"], BY_NAME["head"]])
    corpus = tmp_path / "corpus.jsonl.gz"
    with gzip.open(corpus, "wt") as handle:
        for task in tasks:
            handle.write(json.dumps(dataclasses.asdict(task)) + "\n")
    out = tmp_path / "rows.jsonl"
    assert main(["--corpus", str(corpus), "--split", "all", "--out", str(out)]) == 0
    assert "2/2 solved" in capsys.readouterr().out
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [row["behaviour"] for row in rows] == ["add_pair", "head"]
    assert all(row["shown"] and row["held_out"] for row in rows)


def _anthropic_reply(stop_reason, *texts):
    from types import SimpleNamespace

    blocks = [SimpleNamespace(type="thinking", thinking="hmm")]
    blocks += [SimpleNamespace(type="text", text=text) for text in texts]
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


def test_anthropic_generator_sends_effort_and_no_fallback(mocker):
    from wandering_light.solver import AnthropicTokenGenerator

    generator = AnthropicTokenGenerator()
    assert (generator.model, generator.effort) == ("claude-opus-5", "high")
    create = mocker.patch.object(
        generator.client.messages,
        "create",
        return_value=_anthropic_reply("end_turn", "think\n", "0 [add] fold"),
    )
    reply = generator.generate("prompt")
    assert reply == "think\n0 [add] fold"
    assert parse_response(reply) == "0 [ add ] fold"
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5"
    assert kwargs["output_config"] == {"effort": "high"}
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]
    assert not any("fallback" in key for key in kwargs)


def test_anthropic_generator_refusal_is_empty(mocker):
    from wandering_light.solver import AnthropicTokenGenerator

    generator = AnthropicTokenGenerator(model="claude-sonnet-5", effort="low")
    assert (generator.model, generator.effort) == ("claude-sonnet-5", "low")
    mocker.patch.object(
        generator.client.messages,
        "create",
        return_value=_anthropic_reply("refusal", "partial add"),
    )
    assert generator.generate("prompt") == ""
