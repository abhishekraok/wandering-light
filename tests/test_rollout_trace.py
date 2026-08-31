"""The debugger's trace must describe the production search, not another one.

Two properties carry that. Observing is inert: the same search run with and
without a recorder returns the identical result and charges the identical
executions. And the recorded trace agrees with the result it accompanies --
its termination, its execution count, and the retained paths the search itself
reports in ``RolloutSearchResult.trace``.
"""

import math

import pytest

from wandering_light.function_def import FunctionDef, FunctionDefSet
from wandering_light.rollout_debugger.session import SearchSettings, select_task
from wandering_light.rollout_debugger.trace import TraceRecorder
from wandering_light.solver import ActionScorer, RolloutPredictor
from wandering_light.typed_list import TypedList


def make_function(name: str, code: str, input_type: str = "builtins.int"):
    return FunctionDef(
        name=name,
        input_type=input_type,
        output_type="builtins.int" if input_type == "builtins.int" else input_type,
        code=code,
    )


class StaticScorer(ActionScorer):
    """Deterministic scores, so a trace is reproducible without a checkpoint."""

    def __init__(self, table=None):
        self.table = table or {}

    def score_actions_batch(self, problems, available_functions):
        rows = []
        for state, _target in problems:
            key = tuple(state.items)
            row = self.table.get(key)
            rows.append(
                dict(row)
                if row is not None
                else {function.name: 0.0 for function in available_functions}
            )
        return rows


def basis():
    return FunctionDefSet(
        [
            make_function("increment", "return x + 1"),
            make_function("double", "return x * 2"),
            make_function("halve", "return x // 2"),
            make_function("explode", "raise ValueError('no')"),
        ]
    )


SCENARIOS = {
    # A solved search, an alternate-optimal one (two orderings reach 6 from 2),
    # a depth-terminated one, and a budget-terminated one.
    "solved": {
        "input": [2],
        "target": [6],
        "beam_width": 4,
        "max_depth": 4,
        "budget": 1000,
    },
    "alternate_optimal": {
        "input": [2],
        "target": [8],
        "beam_width": 8,
        "max_depth": 4,
        "budget": 1000,
    },
    "max_depth": {
        "input": [2],
        "target": [9999],
        "beam_width": 4,
        "max_depth": 2,
        "budget": 10000,
    },
    "budget": {
        "input": [2],
        "target": [9999],
        "beam_width": 8,
        "max_depth": 6,
        "budget": 25,
    },
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_observing_a_search_does_not_change_it(name):
    settings = SCENARIOS[name]
    functions = basis()
    predictor = RolloutPredictor(
        StaticScorer(),
        beam_width=settings["beam_width"],
        max_depth=settings["max_depth"],
        budget=settings["budget"],
    )
    start = TypedList(settings["input"])
    target = TypedList(settings["target"])

    plain = predictor.search_with_cost(start, target, functions)
    recorder = TraceRecorder()
    observed = predictor.search_with_cost(start, target, functions, observer=recorder)

    assert observed == plain
    assert observed.executions == plain.executions
    assert observed.termination == plain.termination
    assert [f.name for f in observed.function_defs] == [
        f.name for f in plain.function_defs
    ]


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_trace_agrees_with_the_result_it_records(name):
    settings = SCENARIOS[name]
    functions = basis()
    recorder = TraceRecorder()
    result = RolloutPredictor(
        StaticScorer(),
        beam_width=settings["beam_width"],
        max_depth=settings["max_depth"],
        budget=settings["budget"],
    ).search_with_cost(
        TypedList(settings["input"]),
        TypedList(settings["target"]),
        functions,
        observer=recorder,
    )
    trace = recorder.to_dict()

    assert trace["termination"] == result.termination
    assert trace["executions"] == result.executions
    # Every layer the search reported retaining is in the trace, with the same
    # paths in the same order.
    recorded = {layer["depth"]: layer for layer in trace["layers"]}
    for depth_trace in result.trace:
        entry = recorded[depth_trace.depth]
        assert [tuple(path["path"]) for path in entry["retained"]] == list(
            depth_trace.retained_paths
        )
        assert entry["unique_successors"] == depth_trace.unique_successors


def test_trace_separates_raw_logits_from_normalized_probabilities():
    functions = basis()
    table = {(2,): {"increment": 3.0, "double": 1.0, "halve": 0.0, "explode": -1.0}}
    recorder = TraceRecorder()
    RolloutPredictor(
        StaticScorer(table), beam_width=2, max_depth=1, budget=1000
    ).search_with_cost(TypedList([2]), TypedList([9999]), functions, observer=recorder)

    node = recorder.to_dict()["layers"][0]["nodes"][0]
    assert node["mask_size"] == 4
    by_name = {entry["function"]: entry for entry in node["compatible"]}
    # The raw score is preserved as scored, and the normalized log-probability
    # is separate from it. Probability itself is exp() of that, derived where
    # it is displayed rather than shipped a third time.
    assert by_name["increment"]["logit"] == 3.0
    assert math.isclose(
        sum(math.exp(entry["log_probability"]) for entry in node["compatible"]),
        1.0,
        rel_tol=1e-6,
    )
    assert (
        by_name["increment"]["log_probability"] > by_name["double"]["log_probability"]
    )
    # Raw rows stay out of the default trace; the panel fetches them.
    assert "logits" not in node


def test_logits_are_captured_only_when_asked_for():
    functions = basis()
    recorder = TraceRecorder(capture_logits=True)
    RolloutPredictor(
        StaticScorer(), beam_width=2, max_depth=1, budget=1000
    ).search_with_cost(TypedList([2]), TypedList([9999]), functions, observer=recorder)

    node = recorder.to_dict()["layers"][0]["nodes"][0]
    assert set(node["logits"]) == {function.name for function in functions}


def test_a_raising_function_is_charged_and_recorded_as_raised():
    functions = basis()
    recorder = TraceRecorder()
    RolloutPredictor(
        StaticScorer(), beam_width=4, max_depth=1, budget=1000
    ).search_with_cost(TypedList([2]), TypedList([9999]), functions, observer=recorder)

    expansions = recorder.to_dict()["layers"][0]["expansions"]
    raised = [entry for entry in expansions if entry["function"] == "explode"]
    assert len(raised) == 1
    assert raised[0]["outcome"] == "raised"
    assert "ValueError" in raised[0]["error"]
    assert raised[0]["successor"] is None


def test_width_pruned_successors_are_distinguished_from_unreached_ones():
    functions = basis()
    recorder = TraceRecorder()
    RolloutPredictor(
        StaticScorer(), beam_width=1, max_depth=2, budget=1000
    ).search_with_cost(TypedList([2]), TypedList([9999]), functions, observer=recorder)

    layer = recorder.to_dict()["layers"][0]
    assert len(layer["retained"]) == 1
    # increment/double/halve from 2 give 3, 4 and 1: three unique successors,
    # one retained, so the other two are pruned by width and say so.
    assert layer["unique_successors"] == 3
    assert len(layer["pruned_by_width"]) == 2


def test_selecting_a_task_by_id_is_exact_and_missing_ids_raise():
    class Record:
        def __init__(self, task_id, source_index):
            self.task_id = task_id
            self.source_index = source_index
            self.metadata = {"certified_distance": 3}

    records = [Record("aa", 0), Record("bb", 1)]
    record, provenance = select_task(records, task_id="bb")
    assert record.task_id == "bb"
    assert provenance == {"mode": "task_id", "task_id": "bb"}

    with pytest.raises(KeyError):
        select_task(records, task_id="zz")


def test_sampling_is_seeded_and_records_enough_to_reproduce_it():
    class Record:
        def __init__(self, task_id, source_index, distance):
            self.task_id = task_id
            self.source_index = source_index
            self.metadata = {"certified_distance": distance}

    records = [Record(f"t{index}", index, 5) for index in range(20)]
    first, provenance = select_task(records, certified_distance=5, seed=1234)
    again, _ = select_task(records, certified_distance=5, seed=1234)

    assert first.task_id == again.task_id
    assert provenance["seed"] == 1234
    assert provenance["pool_size"] == 20
    assert provenance["task_id"] == first.task_id


def test_search_settings_round_trip_for_deep_links():
    settings = SearchSettings(beam_width=8, max_depth=5, budget=1000)
    assert settings.to_dict() == {"beam_width": 8, "max_depth": 5, "budget": 1000}
