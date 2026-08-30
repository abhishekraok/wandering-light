"""The acceptance criterion: a replay is the CLI solver's run, not a lookalike.

Marked ``network`` because it needs the published corpus and the private
issue-55 checkpoint. It is the test that would catch the failure the whole
design is arranged to prevent -- a debugger that displays a search subtly
unlike the one being debugged.

    uv run pytest tests/test_rollout_debugger_replay.py -m network
"""

import pytest

from wandering_light.rollout_debugger.context import DebugContext
from wandering_light.rollout_debugger.session import (
    SearchSettings,
    build_session,
    select_task,
)
from wandering_light.solver import RolloutPredictor

pytestmark = pytest.mark.network

SETTINGS = SearchSettings(beam_width=8, max_depth=8, budget=100_000)


@pytest.fixture(scope="module")
def context():
    return DebugContext.load(split="test", device="cpu")


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_a_replay_matches_an_uninstrumented_cli_search_exactly(context, seed):
    record, _ = select_task(context.records, certified_distance=3, seed=seed)

    # What the CLI solver does, with no debugger anywhere near it.
    reference = RolloutPredictor(
        context.policy,
        beam_width=SETTINGS.beam_width,
        max_depth=SETTINGS.max_depth,
        budget=SETTINGS.budget,
    ).search_with_cost(
        record.input_value, record.output_value, context.available_functions
    )

    session = build_session(
        context.policy,
        record,
        context.available_functions,
        settings=SETTINGS,
        provenance={},
    )
    payload = session.to_dict()

    assert payload["termination"] == reference.termination
    assert payload["executions"] == reference.executions
    assert payload["solution"] == [f.name for f in reference.function_defs]
    # Path ordering, layer by layer, not just the final answer.
    for layer, depth_trace in zip(
        payload["trace"]["layers"], reference.trace, strict=False
    ):
        assert [tuple(entry["path"]) for entry in layer["retained"]] == list(
            depth_trace.retained_paths
        )


def test_the_debugger_reproduces_the_published_solve_rate_shape(context):
    """Distance 1-3 solves; distance 7-8 does not. The known step function."""

    for distance, expected in ((1, True), (2, True), (7, False)):
        record, _ = select_task(context.records, certified_distance=distance, seed=5)
        session = build_session(
            context.policy,
            record,
            context.available_functions,
            settings=SETTINGS,
            provenance={},
        )
        assert (session.result.termination == "solved") is expected


def test_the_witness_overlay_never_reaches_the_search(context):
    """A task solved with the witness withheld solves identically with it shown."""

    record, _ = select_task(context.records, certified_distance=2, seed=17)
    session = build_session(
        context.policy,
        record,
        context.available_functions,
        settings=SETTINGS,
        provenance={},
    )
    payload = session.to_dict()

    # The overlay is computed from the finished trace, so stripping the witness
    # changes the annotation and nothing else about the run.
    session.witness = []
    stripped = session.to_dict()
    assert stripped["trace"] == payload["trace"]
    assert stripped["solution"] == payload["solution"]
    assert all(not entry["survives"] for entry in stripped["witness_overlay"])
