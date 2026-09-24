"""novelty-reduction (#171): generator, loop archive, selector, heads, RND, runner."""

import dataclasses
import json
import random
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.certified_coverage import pairs as sampler
from experiments.novelty_reduction import canon, evaluate, heads, programs
from experiments.novelty_reduction import cycle as runner
from experiments.novelty_reduction import select as selector
from experiments.novelty_reduction.archive import PROPOSAL, ROLLOUT, LoopArchive
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import closed_cuts, has_mark
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task

SMALL = ("sum", "reverse", "negate_each", "keep_evens")
TINY = PolicyArchitecture(8, 16, 16)


def inputs_of(name: str, seed: int, k: int = 4):
    task = build_task(
        BY_NAME[name], draw_task(BY_NAME[name], random.Random(seed), k, 1)
    )
    return sampler.parse_inputs(task.inputs)


def test_proposals_are_capped_closed_and_fresh():
    rng = random.Random(0)
    made = 0
    for seed in range(40):
        start = inputs_of(SMALL[seed % len(SMALL)], seed)
        proposal = programs.propose(start, rng)
        if not proposal.cuts:
            continue
        made += 1
        assert proposal.cuts == closed_cuts(proposal.states, proposal.ops)
        assert len(proposal.cuts) <= 3
        edges = [0, *proposal.cuts]
        for a, b in pairwise(edges):
            segment = proposal.ops[a:b]
            assert len(segment) <= programs.SEGMENT_OPS
            body = segment[segment.index("[") + 1 : segment.index("]")]
            assert 1 <= len(body) <= programs.BODY_ATOMS and "[" not in body
            assert segment[-1] in programs.COMBINATORS
        ends = [proposal.states[c] for c in edges]
        assert len(set(ends)) == len(
            ends
        )  # every segment changes the state, no revisits
        assert not any(has_mark(proposal.states[c]) for c in proposal.cuts)
    assert made >= 30


def test_proposals_repeat_under_a_seed():
    start = inputs_of("sum", 1)
    a = programs.propose(start, random.Random(5))
    b = programs.propose(start, random.Random(5))
    assert a.ops == b.ops


def test_fresh_pairs_are_one_segment_and_distinct():
    fresh = programs.fresh_pairs(15, random.Random(3), list(SMALL))
    assert len(fresh) == 15
    assert all(p.d_cuts == 1 and p.source == "F3" for p in fresh)
    assert len({p.key for p in fresh}) == 15
    assert all(3 <= p.k <= 6 for p in fresh)


def test_archive_pairs_relaxations_and_existing_nodes():
    start = inputs_of("negate_each", 3)
    archive = LoopArchive()
    ops = [
        "[",
        "0",
        "swap",
        "sub",
        "]",
        "map",
        "1",
        "drop",
        "[",
        "0",
        "swap",
        "sub",
        "]",
        "map",
    ]
    states = sampler.replay(start, ops)
    added, relaxed = archive.add_path(
        states, ops, PROPOSAL, 1, start_origin=(("x",), ())
    )
    # nodes at 0, 6 and 14; 0 and 14 are the same state, so (0, 14) is skipped
    # and (s, s) never stored
    assert len(archive.states) == 2 and relaxed == 0
    assert added == 2
    record = archive.pairs[(1, 0)]
    assert (record.d_ops, record.d_cuts) == (8, 1)
    # A rollout meets the mid state and returns to the inputs by a shorter
    # route: the plain 'drop' stays out, 6 ops beat 8.
    shorter = ["[", "0", "swap", "sub", "]", "map"]
    back = sampler.replay(archive.states[1], shorter)
    added, relaxed = archive.add_path(back, shorter, ROLLOUT, 2)
    assert relaxed == 1 and archive.pairs[(1, 0)].d_ops == 6
    assert archive.relaxations[-1] == {
        "cycle": 2,
        "s": 1,
        "t": 0,
        "old_d_ops": 8,
        "new_d_ops": 6,
        "by": ROLLOUT,
    }
    assert archive.origins[1] == (("x",), tuple(ops[:6]))


def test_archive_counts_existing_nodes_met_mid_path():
    start = inputs_of("sum", 4)
    archive = LoopArchive()
    ops = ["0", "[", "add", "]", "fold"]
    archive.add_path(sampler.replay(start, ops), ops, PROPOSAL, 1)
    # A path that reaches the fold's state by a plain op (not a cut) still
    # makes it a node end, because it is archived.
    total = archive.states[1]
    other = sampler.replay(start, ["dup"])
    archive.add_path([start, other[-1]], ["dup"], ROLLOUT, 1)
    assert len(archive.states) == 2  # 'dup' ends on no cut and no archived state
    assert archive.index[total] == 1


def test_archive_round_trips(tmp_path):
    start = inputs_of("sum", 4)
    archive = LoopArchive()
    ops = ["0", "[", "add", "]", "fold"]
    archive.add_path(sampler.replay(start, ops), ops, PROPOSAL, 1)
    archive.save(tmp_path / "a.pkl.gz")
    again = LoopArchive.load(tmp_path / "a.pkl.gz")
    assert again.summary() == archive.summary()


def fake_archive(pairs: int) -> LoopArchive:
    archive = LoopArchive()
    for i in range(pairs + 1):
        archive.node(((i,),))
    for i in range(pairs):
        archive.add_path([((i,),), ((i + 1,),)], ["0"], PROPOSAL, 1)
    return archive


def test_selector_rules():
    archive = fake_archive(30)  # both ends already archived, so each path is a pair
    assert len(archive.pairs) == 30
    config = selector.Selector(size=10)
    first = selector.select(archive, 1, random.Random(0), config)
    assert len(first) == 10 and {rule for _, rule in first} == {selector.NOVEL}
    for key, _ in first:
        archive.pairs[key].attempts += 1
    for i in range(1, 31):
        archive.visits[i] = 5 if i <= 25 else 0  # only targets 26-30 still novel
    second = selector.select(archive, 2, random.Random(0), config)
    rules = [rule for _, rule in second]
    assert rules.count(selector.REATTEMPT) == 1
    assert rules.count(selector.NOVEL) == 5  # the five still-novel targets
    assert rules.count(selector.NOVEL_FILL) == 4
    assert len({key for key, _ in second}) == 10


def test_predicted_progress_is_ranked_not_used():
    history = {
        1: {(i, i + 1): 0.5 for i in range(30)},
        2: {(i, i + 1): 0.5 - i / 100 for i in range(30)},
    }
    top = selector.predicted_progress(history, 3, 3)
    assert [(r["s"], r["t"]) for r in top] == [(29, 30), (28, 29), (27, 28)]
    assert selector.predicted_progress(history, 2, 3) == []


def test_partitions_keep_lineages_whole():
    groups = [f"g{i % 7}" for i in range(70)]
    parts = heads.partition(groups, seed=0)
    for g in set(groups):
        assert len({parts[i] for i, x in enumerate(groups) if x == g}) == 1
    assert set(parts.tolist()) == {0, 1, 2}


def test_heads_partitioned_learn_p_and_e_targets_are_out_of_sample():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(600, 6)).astype(np.float32)
    rate = 1 / (1 + np.exp(-3 * features[:, 0]))
    hits = rng.binomial(8, rate)
    parts = heads.partition([i // 3 for i in range(600)], seed=0)
    p, e, facts = heads.fit_partitioned(features, hits, 8, parts, seed=0)
    assert np.corrcoef(p.predict(features), rate)[0, 1] > 0.9
    assert facts["pairs_by_part"] == np.bincount(parts, minlength=3).tolist()
    assert 0 <= facts["e_target_rate"] <= 1
    assert e.predict(features).shape == (600,)


def test_rnd_separates_seen_from_unseen():
    seen = [inputs_of(SMALL[i % 4], i) for i in range(40)]
    unseen = [inputs_of(SMALL[i % 4], 1000 + i) for i in range(40)]
    rnd = heads.RND(TINY, seed=0)
    for epoch in range(30):
        rnd.train(seen, epochs=1, batch=16, seed=epoch)
    rnd.normalise(seen + unseen)
    check = heads.calibrate_threshold(rnd.novelty(seen), rnd.novelty(unseen))
    assert check["seen_pass"] >= 0.9
    assert check["unseen_pass"] < check["seen_pass"]


def test_policy_rows_cap_rollout_rows_at_proposal_rows():
    start = inputs_of("sum", 4)
    archive = LoopArchive()
    ops = ["0", "[", "add", "]", "fold"]
    archive.add_path(sampler.replay(start, ops), ops, PROPOSAL, 1)
    for extra in ("1 [ add ] fold", "nil [ cons ] fold", "0 [ sub ] fold"):
        path = extra.split()
        archive.add_path(sampler.replay(start, path), path, ROLLOUT, 1)
    rows, facts, paths = runner.policy_rows(archive, runner.Config(), 1, {})
    assert facts["proposal_rows"] == 5
    assert facts["rollout_rows"] <= facts["proposal_rows"]
    assert len(rows) == facts["proposal_rows"] + facts["rollout_rows"]
    assert facts["path_turnover"] is None
    _rows, facts, _ = runner.policy_rows(archive, runner.Config(), 2, paths)
    assert facts["path_turnover"] == 0.0


def test_witness_distance_survives_relaxation():
    start = inputs_of("negate_each", 3)
    archive = LoopArchive()
    ops = [
        "[",
        "0",
        "swap",
        "sub",
        "]",
        "map",
        "1",
        "drop",
        "[",
        "0",
        "swap",
        "sub",
        "]",
        "map",
    ]
    archive.add_path(sampler.replay(start, ops), ops, PROPOSAL, 1)
    shorter = ["[", "0", "swap", "sub", "]", "map"]
    archive.add_path(sampler.replay(archive.states[1], shorter), shorter, ROLLOUT, 2)
    record = archive.pairs[(1, 0)]
    assert (record.d_ops, record.witness_d_ops) == (6, 8)
    assert record.witness_path == tuple(ops[6:])


def test_canonical_keys_ignore_example_order():
    s = inputs_of("sum", 4)
    t = sampler.replay(s, ["0", "[", "add", "]", "fold"])[-1]
    flipped = (tuple(reversed(s)), tuple(reversed(t)))
    assert canon.pair_key(s, t) == canon.pair_key(*flipped)
    assert canon.state_key(s) == canon.state_key(flipped[0])
    # misaligned: the same stacks paired differently is a different pair
    if len(set(t)) > 1:
        assert canon.pair_key(s, t) != canon.pair_key(s, tuple(reversed(t)))


def test_truncation_and_encoded_collisions():
    from wandering_light.stack_machine import Int, Lst

    small = ((Int(1),),)
    big = ((Lst(tuple(Int(i) for i in range(200))),),)
    assert not canon.truncated(small, small)
    assert canon.truncated(big, small) and canon.truncated(small, big)
    a, b = np.zeros((2, 3), dtype=np.int16), np.ones((2, 3), dtype=np.int16)
    assert canon.encoded_collisions([a, a, b], ["x", "y", "z"]).tolist() == [
        True,
        True,
        False,
    ]
    assert canon.encoded_collisions([a, a], ["x", "x"]).tolist() == [False, False]


def test_body_atoms_and_grammar_level():
    assert programs.body_atoms(["0", "[", "add", "]", "fold"]) == 1
    assert (
        programs.body_atoms(
            ["[", "0", "swap", "sub", "]", "map", "[", "dup", "]", "map"]
        )
        == 3
    )
    assert (
        programs.body_atoms(["[", "[", "1", "]", "map", "]", "i"]) == 2
    )  # nested atoms count, brackets not
    assert (
        programs.path_segments(
            ["[", "0", "swap", "sub", "]", "map", "[", "dup", "]", "map"]
        )
        == 2
    )
    assert programs.body_stratum(4) == "3-4" and programs.body_stratum(0) == "0"
    small = programs.Grammar.parse("2,1,none")
    rng = random.Random(0)
    for seed in range(20):
        proposal = programs.propose(inputs_of("sum", seed), rng, small)
        assert len(proposal.cuts) <= 1
        assert programs.body_atoms(proposal.ops) <= 2
    try:
        programs.Grammar.parse("2,1,nested")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("nesting is not in the M2 grammar")


def test_proper_scores_and_controls():
    hits = np.array([0, 8, 4])
    assert np.allclose(evaluate.brier(np.array([0.0, 1.0, 0.5]), hits, 8), [0, 0, 0.25])
    loss = evaluate.log_loss(np.array([0.5, 0.5, 0.5]), hits, 8)
    assert np.allclose(loss, np.log(2))
    # the Wilson-gaming constant: all-zero labels are passed by a small constant
    assert evaluate.wilson_constant(np.zeros(50, dtype=int), 8) <= 0.25
    interval = evaluate.clustered_interval(np.arange(10.0), [i // 2 for i in range(10)])
    assert interval["low"] <= interval["mean"] <= interval["high"]
    assert interval["clusters"] == 5
    risk = evaluate.risk_at_coverage(
        np.array([0.9, 0.8, 0.1, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0, 0, 8, 8]),
        8,
        coverages=(0.5, 1.0),
    )
    assert risk == {"0.5": 0.0, "1.0": 0.5}
    model = evaluate.fit_logistic(
        np.array([[0.0], [1.0]] * 20), np.array([0, 8] * 20), 8
    )
    assert evaluate.predict_logistic(model, np.array([[1.0]]))[0] > 0.9


def tiny_config(**changes):
    config = runner.Config(
        seed=0,
        cycles=2,
        proposals=20,
        calibration=12,
        test_fresh=8,
        test_fresh_final=10,
        test_archive=8,
        paired_check=6,
        tracked_new=20,
        architecture=(8, 16, 16),
        train_device="cpu",
        budget=4,
        n=2,
        behaviours=SMALL,
        bootstrap_proposals=20,
        bootstrap_block_epochs=1,
        bootstrap_max_epochs=2,
        bootstrap_check=6,
        rnd_cohort=10,
        floor_success=0.0,
        floor_pairs=0.0,
        uninformative_zero_share=1.1,
        collapse_ratio=0.0,
    )
    return dataclasses.replace(config, **changes)


def test_two_tiny_cycles_end_to_end(tmp_path):
    reports = runner.run(tiny_config(), tmp_path, log=lambda line: None)
    assert [r["cycle"] for r in reports] == [0, 1, 2]
    assert reports[0]["floor_met_at_block"] == 1
    assert reports[0]["blocks"][0]["histogram"]
    for report in reports[1:]:
        assert report["test_left_archive_unchanged"]
        assert report["rows"]["rollout_rows"] <= report["rows"]["proposal_rows"]
        assert report["policy"]["steps"] >= 1
        arms = report["test"]["F3"]["arms"]
        assert {"full", "p_open", "p_e", "distance_learned_gate", "cheap"} <= set(arms)
        assert {"constant", "distance", "zero", "wilson_constant"} <= set(arms)
        assert "full_count_gate" in report["test"]["archive"]["arms"]
        assert set(arms["full"]["by"]) >= {
            "d_ops",
            "body",
            "truncated",
            "novelty_class",
        }
        assert "log_loss" in arms["zero"]
        assert "unseen_pass" in report["rnd"] and "cohort_pass" in report["rnd"]
        assert report["executions"]["rollouts_this_cycle"] > 0
        assert report["diagnostics"]["fresh_success"]["sufficient"] is False  # 6 < 100
        assert set(report["timings"]) >= {"propose", "policy", "calibration_rollouts"}
    assert reports[2]["test"]["F3"]["facts"]["drawn"] == 10  # the final pool size
    assert reports[2]["calibration"]["rules"].get(selector.REATTEMPT, 0) >= 1
    assert reports[2]["propose"].get("from_archive", 0) >= 1
    saved = json.loads((tmp_path / "cycle-2" / "report.json").read_text())
    assert saved["cycle"] == 2
    assert (tmp_path / "cycle-2" / "policy" / "model.safetensors").exists()
    assert (tmp_path / "cycle-2" / "calibration.jsonl.gz").exists()


def test_bootstrap_floor_not_met_stops_the_run(tmp_path):
    reports = runner.run(
        tiny_config(floor_success=1.1), tmp_path, log=lambda line: None
    )
    assert len(reports) == 1
    assert reports[0]["stop"] == "bootstrap floor not met"
    assert len(reports[0]["blocks"]) == 2  # the whole budget, blocks recorded


def test_uninformative_labels_stop_after_the_cycle(tmp_path):
    reports = runner.run(
        tiny_config(uninformative_zero_share=0.0), tmp_path, log=lambda line: None
    )
    assert reports[-1]["cycle"] == 1
    assert reports[-1]["stop"].startswith("uninformative")
