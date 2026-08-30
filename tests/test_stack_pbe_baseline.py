"""The baseline runner: the search backend and the model-response parser.

The parser is the part worth testing without an API key. A model's reply is
prose plus a program, and normalising it wrong either invents a solve or
throws one away.
"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from stack_pbe_baseline import (
    bfs_solve,
    llm_prompt,
    parse_response,
    random_solve,
    solve_with_llm,
)

from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_pbe_eval import grade


@pytest.fixture(scope="module")
def sum_task():
    rng = random.Random(4)
    return build_task(BY_NAME["sum"], draw_task(BY_NAME["sum"], rng, 4, held_out=6))


@pytest.fixture(scope="module")
def add_task():
    rng = random.Random(6)
    return build_task(
        BY_NAME["add_pair"], draw_task(BY_NAME["add_pair"], rng, 4, held_out=6)
    )


@pytest.mark.parametrize(
    "reply",
    [
        "0 [ add ] fold",
        "The answer is:\n```\n0 [ add ] fold\n```",
        "<think>fold it</think>\n0 [add] fold",
        "`0 [ add ] fold`",
        "Sum the list.\n0 [ add ] fold",
    ],
)
def test_parse_response_recovers_the_program(reply, sum_task):
    assert grade(parse_response(reply), sum_task).solved


def test_parse_response_does_not_repair_a_wrong_answer(sum_task):
    """An invented op name must stay invented, so grading can reject it."""
    assert not grade(parse_response("0 [ plus ] fold"), sum_task).parsed


def test_parse_response_survives_an_empty_reply(sum_task):
    assert parse_response("") == ""
    assert not grade(parse_response(""), sum_task).parsed


def test_bfs_finds_a_one_op_program(add_task):
    assert bfs_solve(add_task, budget=500) == ("add",)


def test_bfs_returns_nothing_past_its_budget(sum_task):
    """`0 [ add ] fold` is five ops; a tiny budget cannot reach it."""
    assert bfs_solve(sum_task, budget=5) is None


def test_bfs_never_reads_the_held_out_examples(sum_task, monkeypatch):
    """The protocol depends on this: a solver sees the shown examples only."""
    from dataclasses import replace

    blinded = replace(sum_task, held_out_inputs=(), held_out_outputs=(), held_out=0)
    assert bfs_solve(blinded, budget=20_000) == bfs_solve(sum_task, budget=20_000)


def test_random_finds_a_one_op_program(add_task):
    found = random_solve(add_task, budget=2_000, rng=random.Random(1), max_length=2)
    assert found is not None
    assert grade(found, add_task).shown


def test_prompt_shows_the_examples_and_hides_the_held_out_set(sum_task):
    prompt = llm_prompt(sum_task)
    for given in sum_task.inputs:
        assert given in prompt
    for hidden in sum_task.held_out_inputs:
        assert hidden not in prompt
    assert "fold" in prompt


def test_every_provider_is_reachable_by_name():
    """The four backends exist and take a model name. No key needed to check."""
    import wandering_light.solver as solver_module

    for name in (
        "OpenAITokenGenerator",
        "GeminiTokenGenerator",
        "OllamaTokenGenerator",
    ):
        assert hasattr(solver_module, name)
    generator = solver_module.AnthropicTokenGenerator("claude-opus-5", effort="low")
    assert generator.model == "claude-opus-5"
    assert generator.effort == "low"


def test_a_failed_generation_costs_one_candidate_not_the_run(sum_task):
    """The deadline that bounds a runaway must not end the evaluation itself.

    Without this the first timeout propagates out of the task loop and the run
    dies before writing any summary, which is worse than the hang it replaced.
    """

    class Exploding:
        def generate(self, prompt):
            raise TimeoutError("read timeout")

    assert solve_with_llm(sum_task, Exploding(), samples=2) == ["", ""]
    assert not grade("", sum_task).parsed
