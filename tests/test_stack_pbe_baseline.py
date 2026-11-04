"""The first baseline runner, and the Anthropic generator it drives.

Every solver here sees only the shown half of a task; the grader in
``stack_pbe_eval`` decides what it found. The API generator is tested
against a fake client, pinning the request shape (model, effort, no
fallbacks) and the refusal path rather than any model's answers.
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

import stack_pbe_baseline as baseline

from wandering_light import solver
from wandering_light.stack_machine import MARK_OPS
from wandering_light.stack_pbe import (
    CATALOGUE,
    Task,
    build_task,
    draw_task,
)
from wandering_light.stack_pbe_eval import grade

BY_NAME = {behaviour.name: behaviour for behaviour in CATALOGUE}


def catalogue_task(name: str, seed: int = 5, examples: int = 4) -> Task:
    behaviour = BY_NAME[name]
    return build_task(
        behaviour, draw_task(behaviour, random.Random(seed), examples=examples)
    )


@pytest.fixture
def add_pair() -> Task:
    return catalogue_task("add_pair")


@pytest.fixture
def tail() -> Task:
    return catalogue_task("tail")


# -- bfs_solve ----------------------------------------------------------------


def test_bfs_finds_one_op_program(add_pair):
    program = baseline.bfs_solve(add_pair, budget=100)
    assert program == ("add",)
    assert grade(program, add_pair).solved


def test_bfs_finds_two_op_program_and_it_generalises(tail):
    program = baseline.bfs_solve(tail, budget=2_000, max_depth=3)
    assert program is not None
    assert len(program) == 2
    assert grade(program, tail).solved


def test_bfs_returns_first_in_basis_order():
    # `drop` fits at depth 1, so the two-op fits (`swap drop` is not one, but
    # `dup drop drop`-style detours exist at depth 3) are never reached:
    # breadth-first answers at the shallowest depth, in basis order.
    task = Task(
        behaviour="x",
        summary="",
        split="eval",
        task_hash_split="eval",
        inputs=("[ 0 1 ]", "[ 1 1 ]"),
        outputs=("[ 0 ]", "[ 1 ]"),
        held_out_inputs=("[ 1 0 ]",),
        held_out_outputs=("[ 1 ]",),
        program=("drop",),
        length=1,
        examples=2,
        held_out=1,
        fuel=0,
        tags=(),
    )
    assert baseline.bfs_solve(task, budget=10) == ("drop",)


def test_bfs_respects_budget(tail):
    # One expansion (the root) only reaches one-op programs; `tail` needs two.
    assert baseline.bfs_solve(tail, budget=1, max_depth=3) is None
    assert baseline.bfs_solve(tail, budget=0, max_depth=3) is None


def test_bfs_respects_max_depth(tail):
    assert baseline.bfs_solve(tail, budget=10_000, max_depth=1) is None
    assert baseline.bfs_solve(tail, budget=10_000, max_depth=2) is not None


def test_bfs_only_reads_the_shown_half(tail):
    # Strip the held-out half entirely: the solver must not notice.
    blind = Task(**{**asdict(tail), "held_out_inputs": (), "held_out_outputs": ()})
    assert baseline.bfs_solve(blind, budget=2_000) == baseline.bfs_solve(
        tail, budget=2_000
    )


def test_bfs_dedup_and_ops_in_basis_order(add_pair):
    start, _target = baseline.shown(add_pair)
    # Every move the search makes is one of the corpus basis, in its order.
    moves = [op.name for op, _state in baseline.legal_ops(start)]
    names = [op.name for op in MARK_OPS]
    assert moves == [name for name in names if name in moves]
    assert "[" in moves  # the bracket basis, not executable-26


# -- random_solve -------------------------------------------------------------


def test_random_finds_one_op_program(add_pair):
    program = baseline.random_solve(add_pair, budget=500, rng=random.Random(0))
    assert program == ("add",)


def test_random_is_deterministic_under_a_seed(tail):
    first = baseline.random_solve(tail, budget=300, rng=random.Random(11))
    second = baseline.random_solve(tail, budget=300, rng=random.Random(11))
    assert first == second


def test_random_returns_none_without_budget(add_pair):
    assert baseline.random_solve(add_pair, budget=0, rng=random.Random(0)) is None


def test_random_result_fits_the_shown_half(tail):
    program = baseline.random_solve(
        tail, budget=2_000, rng=random.Random(2), max_length=4
    )
    if program is not None:
        assert grade(program, tail).shown


# -- llm_prompt ---------------------------------------------------------------


def test_prompt_lists_every_op_with_its_effect(add_pair):
    prompt = baseline.llm_prompt(add_pair)
    for op in MARK_OPS:
        assert f"{op.name:<8} {op.effect}" in prompt


def test_prompt_shows_the_pairs_and_hides_the_held_out_half():
    task = Task(
        behaviour="x",
        summary="",
        split="eval",
        task_hash_split="eval",
        inputs=("[ 2 3 ]", "[ 4 5 ]"),
        outputs=("[ 5 ]", "[ 9 ]"),
        held_out_inputs=("[ 7 8 ]",),
        held_out_outputs=("[ 15 ]",),
        program=("add",),
        length=1,
        examples=2,
        held_out=1,
        fuel=0,
        tags=(),
    )
    prompt = baseline.llm_prompt(task)
    assert "[ 2 3 ] -> [ 5 ]" in prompt
    assert "[ 4 5 ] -> [ 9 ]" in prompt
    assert "[ 7 8 ]" not in prompt
    assert "15" not in prompt
    assert "add" not in prompt.split("The ops are:")[0]  # no witness leaks


def test_prompt_explains_brackets(add_pair):
    prompt = baseline.llm_prompt(add_pair)
    assert "[" in prompt and "]" in prompt
    assert "quotation" in prompt.lower()
    assert "last line" in prompt


# -- parse_response -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("0 [ add ] fold", "0 [ add ] fold"),
        ("The program is:\n\n0 [ add ] fold\n", "0 [ add ] fold"),
        ("`0 [ add ] fold`", "0 [ add ] fold"),
        ("```\n0 [ add ] fold\n```", "0 [ add ] fold"),
        ("```text\n[dup add] map\n```\n", "[ dup add ] map"),
        ("[add]", "[ add ]"),
        ("[[dup add]]", "[ [ dup add ] ]"),
        ("<think>\nlet me think\nadd\n</think>\n\nsub", "sub"),
        ("some thinking\n</think>\n```\nmul\n```", "mul"),
        ("", ""),
        ("   \n\n", ""),
        ("```\n```", ""),
    ],
)
def test_parse_response(text, want):
    assert baseline.parse_response(text) == want


def test_parse_response_never_repairs_unknown_tokens():
    assert baseline.parse_response("frobnicate [add] mapp") == "frobnicate [ add ] mapp"
    assert baseline.parse_response("0 [ add ] fold.") == "0 [ add ] fold."
    assert not grade(baseline.parse_response("frob"), catalogue_task("sum")).parsed


def test_parse_response_strips_thinking_with_remove_thinking(monkeypatch):
    called = []

    def fake(text):
        called.append(text)
        return "add"

    monkeypatch.setattr(solver, "remove_thinking", fake)
    assert baseline.parse_response("whatever") == "add"
    assert called == ["whatever"]


# -- the runner ---------------------------------------------------------------


def test_evaluate_grades_on_the_held_out_half(add_pair, tail):
    results = baseline.evaluate([add_pair, tail], lambda task: ("add",))
    scores = [score for score, _ in results]
    assert scores[0].solved
    assert not scores[1].solved  # `add` is refused on a one-list stack
    rows = [row for _, row in results]
    assert rows[0]["verdict"] == "solved"
    assert rows[1]["verdict"] == "illegal"
    assert rows[0]["witness"] == "add"


def test_evaluate_treats_none_as_the_empty_program(add_pair):
    [(score, row)] = baseline.evaluate([add_pair], lambda task: None)
    assert not score.parsed
    assert row["verdict"] == "unparsed" and row["program"] == ""


def test_main_runs_bfs_over_a_small_corpus(tmp_path, capsys):
    corpus = tmp_path / "tiny.jsonl.gz"
    tasks = [catalogue_task("add_pair", seed=s) for s in range(3)]
    with gzip.open(corpus, "wt") as handle:
        for task in tasks:
            handle.write(json.dumps(asdict(task)) + "\n")
    out = tmp_path / "out.json"
    assert (
        baseline.main(
            [
                "--corpus",
                str(corpus),
                "--solver",
                "bfs",
                "--split",
                "all",
                "--tasks",
                "3",
                "--budget",
                "50",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "3/3 solved (100.0%)" in printed
    assert "shown-only would say 100.0% (+0.0%, 0 memorised)" in printed
    written = json.loads(out.read_text())
    assert written["report"]["solved"] == 3
    assert [row["program"] for row in written["rows"]] == ["add"] * 3
    assert written["model"] is None


# -- AnthropicTokenGenerator --------------------------------------------------


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeAnthropic:
    def __init__(self, response):
        self.messages = FakeMessages(response)


def text_response(*texts, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=text) for text in texts]
    return SimpleNamespace(content=content, stop_reason=stop_reason)


@pytest.fixture
def anthropic_client(monkeypatch):
    holder = {}

    def factory(response):
        client = FakeAnthropic(response)
        monkeypatch.setattr(solver.anthropic, "Anthropic", lambda: client)
        holder["client"] = client
        return client

    return factory


def test_generator_defaults_and_attributes(anthropic_client):
    anthropic_client(text_response("add"))
    generator = solver.AnthropicTokenGenerator()
    assert generator.model == "claude-opus-5"
    assert generator.effort == "high"
    custom = solver.AnthropicTokenGenerator(model="claude-sonnet-5-5", effort="low")
    assert (custom.model, custom.effort) == ("claude-sonnet-5-5", "low")


def test_generator_calls_messages_create_with_effort_and_no_fallbacks(
    anthropic_client,
):
    client = anthropic_client(text_response("0 [ add ] fold"))
    generator = solver.AnthropicTokenGenerator(model="claude-opus-5", effort="xhigh")
    assert generator.generate("a prompt") == "0 [ add ] fold"
    [call] = client.messages.calls
    assert call["model"] == "claude-opus-5"
    assert call["output_config"] == {"effort": "xhigh"}
    assert call["messages"] == [{"role": "user", "content": "a prompt"}]
    assert "fallbacks" not in call
    assert "betas" not in call
    assert generator.llm_io_history == [("a prompt", "0 [ add ] fold")]


def test_generator_joins_text_blocks_and_skips_thinking(anthropic_client):
    anthropic_client(text_response("0 [ add ]", " fold"))
    assert solver.AnthropicTokenGenerator().generate("p") == "0 [ add ] fold"


def test_generator_returns_empty_on_refusal(anthropic_client):
    anthropic_client(text_response("not this", stop_reason="refusal"))
    generator = solver.AnthropicTokenGenerator()
    assert generator.generate("p") == ""
    assert generator.llm_io_history == [("p", "")]


def test_generator_batch_is_sequential(anthropic_client):
    client = anthropic_client(text_response("add"))
    outputs = solver.AnthropicTokenGenerator().generate_batch(["a", "b"])
    assert outputs == ["add", "add"]
    assert [call["messages"][0]["content"] for call in client.messages.calls] == [
        "a",
        "b",
    ]


def test_runner_wires_generator_prompt_and_parser(anthropic_client, add_pair):
    client = anthropic_client(text_response("I think it is\n\n`add`"))
    solve = baseline._anthropic("claude-opus-5", "high")
    assert solve(add_pair) == "add"
    [call] = client.messages.calls
    assert call["messages"][0]["content"] == baseline.llm_prompt(add_pair)
    assert grade(solve(add_pair), add_pair).solved
