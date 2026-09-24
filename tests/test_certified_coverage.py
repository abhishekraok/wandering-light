"""certified-coverage (#170): pair sampler, rollouts and scoring."""

import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.certified_coverage import pairs as sampler
from experiments.certified_coverage import score as scoring
from experiments.certified_coverage.rollout import Procedure, rollout_hits
from experiments.certified_coverage.run import snapshot
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import OP_NAMES


def drawn(name: str, seed: int):
    return build_task(
        BY_NAME[name], draw_task(BY_NAME[name], random.Random(seed), 4, 6)
    )


def pairs_along(task, program, **kw):
    nodes = sampler.lineage_nodes(sampler.parse_inputs(task.inputs), program.split())
    return list(
        sampler.tree_pairs(
            nodes,
            source="F1",
            behaviour=task.behaviour,
            example_set=0,
            inputs=task.inputs,
            **kw,
        )
    )


def test_lineage_nodes_are_inputs_cuts_and_end():
    task = drawn("negate_each", 3)
    program = "[ 0 swap sub ] map [ 0 swap sub ] map"
    nodes = sampler.lineage_nodes(sampler.parse_inputs(task.inputs), program.split())
    assert [len(n.program) for n in nodes] == [0, 6, 12]
    assert nodes[0].product == nodes[2].product  # negating twice is the identity


def test_pairs_skip_equal_ends_and_count_cuts():
    task = drawn("negate_each", 3)
    found = pairs_along(task, "[ 0 swap sub ] map [ 0 swap sub ] map")
    # (0, 12) has s == t and is dropped; the two one-cut pairs remain.
    assert sorted(
        (len(p.s_program), len(p.t_program), p.d_cuts, p.d_ops) for p in found
    ) == [
        (0, 6, 1, 6),
        (6, 12, 1, 6),
    ]


def test_final_state_is_a_node_even_when_not_closed():
    task = drawn("sum", 4)
    found = pairs_along(task, "0 [ add ] fold 1 add")  # ends on a plain op
    assert sorted((len(p.s_program), len(p.t_program), p.d_cuts) for p in found) == [
        (0, 5, 1),
        (0, 7, 2),
        (5, 7, 1),
    ]
    assert len(pairs_along(task, "0 [ add ] fold 1 add", max_cuts=1)) == 2


def test_archive_pairs_join_only_archive_nodes(tmp_path):
    task = drawn("negate_each", 3)
    half = "[ 0 swap sub ] map"
    record = {
        "behaviour": task.behaviour,
        "example_set": 0,
        "inputs": list(task.inputs),
        "outputs": list(task.outputs),
        "program": half,
        "segments": [
            {
                "roots": [{"program": "", "score": 0.0}],
                "states": [
                    # kept, but its closed parent (the first map) is not
                    {"program": f"{half} [ 0 swap sub ] map [ 1 add ] map"},
                ],
            }
        ],
    }
    path = tmp_path / "a.jsonl.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(json.dumps(record) + "\n")
    found = sampler.archive_pairs(path)
    got = sorted((len(p.s_program), len(p.t_program), p.d_cuts) for p in found)
    # inputs -> fit end (6 ops, one cut); fit end -> kept state (11 more ops,
    # two cuts); inputs -> kept state (three cuts). The unkept 12-op state is
    # passed, never an end.
    assert got == [(0, 6, 1), (0, 17, 3), (6, 17, 2)]
    assert sampler.archive_pairs(path, max_cuts=1)[0].d_cuts == 1


def test_dedupe_counts_multiplicity_and_keeps_shortest():
    task = drawn("sum", 4)
    short = pairs_along(task, "0 [ add ] fold")
    # Adding 0 twice returns to the fold's state: (0, 9) is (0, 5) again, by a
    # longer route, and (5, 9) has s == t.
    long = pairs_along(task, "0 [ add ] fold 0 add 0 add")
    merged = sampler.dedupe([*short, *short, *long])
    assert len(merged) == 1
    only = merged[0]
    assert (only.multiplicity, only.d_ops, only.d_cuts) == (4, 5, 1)


class OneOp:
    """A scorer that always prefers one op when it is legal."""

    def __init__(self, name: str):
        self.index = OP_NAMES.index(name)

    def encoder(self, target):
        return lambda product: np.zeros((len(product), 1), dtype=np.int64)

    def log_probs(self, blocks, legal):
        logits = np.where(legal, 0.0, -np.inf)
        logits[:, self.index] = np.where(legal[:, self.index], 50.0, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def test_rollouts_hit_when_the_policy_takes_the_path():
    task = drawn("sum", 4)
    inputs = sampler.parse_inputs(task.inputs)
    after = sampler.replay(inputs, ["0"])[-1]
    pair = sampler.Pair("F1", "sum", 0, task.inputs, (), ("0",), 1, 1, inputs, after)
    procedure = Procedure("test", budget=3, n=8)
    assert rollout_hits(OneOp("0"), [pair], procedure, seed=0).tolist() == [8]
    assert rollout_hits(OneOp("nil"), [pair], procedure, seed=0).tolist() == [0]
    # batching splits walkers without changing the count
    assert rollout_hits(
        OneOp("0"), [pair, pair], procedure, seed=0, batch=3
    ).tolist() == [8, 8]


def test_wilson_interval_and_hit_rules():
    low, high = scoring.wilson(np.array([0, 8, 4]), 8)
    assert np.allclose(low, [0.0, 0.747, 0.249], atol=1e-3)
    assert np.allclose(high, [0.253, 1.0, 0.751], atol=1e-3)
    v_hat = np.array([0.0, 1.0, 0.3, 0.3])
    hits = np.array([0, 8, 4, 8])
    assert scoring.hit_wilson(v_hat, hits, 8).tolist() == [True, True, True, False]
    assert scoring.hit_tau(v_hat, hits, 8, 0.15).tolist() == [True, True, False, False]


def test_report_keeps_coverage_and_false_certification_apart():
    rule = scoring.Rule()
    strata = [1, 1, 1, 2]
    v_hat = [0.0, 0.0, 0.0, 0.5]
    gate = [True, True, False, True]
    hits = np.array([0, 8, 0, 4])
    out = scoring.report(strata, v_hat, gate, hits, 8, rule)
    one = out["1"]["wilson"]
    assert (one["certified"], one["false_certified"], one["gate_open"]) == (1, 1, 2)
    assert out["1"]["v_emp_histogram"][0] == 2 and out["1"]["v_emp_histogram"][8] == 1
    assert out["2"]["wilson"]["certified_coverage"] == 1.0


def test_baselines_fit_and_gate():
    d_ops = np.array([4] * 20 + [12] * 20)
    hits = np.array([8] * 20 + [0] * 10 + [8] * 10)
    constant = scoring.fit_constant(np.ones(40, dtype=int), hits, 8)
    assert constant["by_stratum"][1] == 0.75
    distance = scoring.fit_distance(d_ops, hits, 8)
    predicted = scoring.predict_distance(distance, np.array([4, 12]))
    assert np.allclose(predicted, [1.0, 0.5], atol=1e-3)
    open_bins = scoring.fit_gate(
        d_ops, scoring.predict_distance(distance, d_ops), hits, 8, scoring.Rule()
    )
    assert open_bins == [4]  # d_ops 12 is a coin flip, never certifiable at 0.5
    assert scoring.apply_gate(open_bins, np.array([4, 12, 7])).tolist() == [
        True,
        False,
        False,
    ]


def test_snapshot_drops_a_member_mid_write(tmp_path):
    live = tmp_path / "live.jsonl.gz"
    for i in range(2):
        with gzip.open(live, "at") as handle:
            handle.write(json.dumps({"i": i}) + "\n")
    whole = gzip.compress((json.dumps({"i": 2}) + "\n").encode())
    with open(live, "ab") as handle:
        handle.write(whole[: len(whole) // 2])
    target, count = snapshot(live, tmp_path / "out")
    assert count == 2
    assert [r["i"] for r in sampler.archive_records(target)] == [0, 1]


def test_baselines_score_every_source():
    from experiments.certified_coverage.run import score_baselines

    task = drawn("sum", 4)
    one = pairs_along(task, "0 [ add ] fold 1 add")
    judged = {"archive": one * 5, "F1": one, "F2": one[:1]}
    hits = {
        "archive": np.array([8, 0, 8] * 5),
        "F1": np.array([8, 0, 8]),
        "F2": np.array([8]),
    }
    out = score_baselines(judged, hits, 8, scoring.Rule(min_bin=1))
    assert set(out) == {"constant", "distance"}
    for baseline in out.values():
        assert set(baseline["scores"]) == {"archive", "F1", "F2"}
        for source in baseline["scores"].values():
            assert set(source) == {"gated", "always_open"}
    json.dumps(out)  # the report must serialise


def test_draw_is_per_behaviour_capped_and_order_free():
    def fake(behaviour, i):
        return sampler.Pair(
            "F1", behaviour, i, (), (), (str(i),), 1, 1, ((i,),), ((i + 1,),)
        )

    pairs = [fake("a", i) for i in range(50)] + [fake("b", i) for i in range(3)]
    drawn_pairs = sampler.draw(pairs, 10, seed=7)
    assert sum(p.behaviour == "a" for p in drawn_pairs) == 10
    assert sum(p.behaviour == "b" for p in drawn_pairs) == 3  # capped at what it has
    shuffled = list(pairs)
    random.Random(1).shuffle(shuffled)
    again = sampler.draw(shuffled, 10, seed=7)
    assert [p.key for p in again] == [p.key for p in drawn_pairs]
    assert [p.key for p in sampler.draw(pairs, 10, seed=8)] != [
        p.key for p in drawn_pairs
    ]


def test_archive_task_nodes_come_in_program_order():
    task = drawn("sum", 4)
    record = {
        "inputs": list(task.inputs),
        "program": None,
        "segments": [
            {
                "roots": [{"program": ""}],
                "states": [
                    {"program": "0 [ add ] fold"},
                    {"program": "1 drop 0 [ add ] fold"},
                    {"program": "nil [ cons ] fold"},
                ],
            }
        ],
    }
    nodes = sampler.archive_task_nodes(record)
    assert [n.program for n in nodes] == sorted(n.program for n in nodes)
