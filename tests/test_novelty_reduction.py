"""novelty-reduction (#171): generator, loop archive, selector, heads, RND, runner."""

import json
import random
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.certified_coverage import pairs as sampler
from experiments.novelty_reduction import cycle as runner
from experiments.novelty_reduction import heads, programs
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
    first = selector.select(archive, 1, {}, random.Random(0), config)
    assert len(first) == 10 and {rule for _, rule in first} == {selector.NOVEL}
    for key, _ in first:
        archive.pairs[key].attempts += 1
    for i in range(1, 31):
        archive.visits[i] = 5 if i <= 25 else 0  # only targets 26-30 still novel
    history = {
        1: {(i, i + 1): 0.5 for i in range(30)},
        2: {(i, i + 1): 0.5 - i / 100 for i in range(30)},
    }
    third = selector.select(archive, 3, history, random.Random(0), config)
    rules = [rule for _, rule in third]
    assert rules.count(selector.REATTEMPT) == 1
    assert rules.count(selector.PROGRESS) == 4
    progress = [key for key, rule in third if rule == selector.PROGRESS]
    assert (29, 30) in progress  # the largest predicted drop
    assert len({key for key, _ in third}) == 10
    assert set(rules) <= {
        selector.REATTEMPT,
        selector.PROGRESS,
        selector.NOVEL,
        selector.NOVEL_FILL,
    }


def test_heads_cross_fit_learns_v():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 6)).astype(np.float32)
    rate = 1 / (1 + np.exp(-3 * features[:, 0]))
    hits = rng.binomial(8, rate)
    v, e, facts, oof = heads.fit_heads(features, hits, 8, seed=0)
    predicted = v.predict(features)
    assert np.corrcoef(predicted, rate)[0, 1] > 0.9
    assert oof.shape == (400,) and 0 <= facts["cross_fit_miss_rate"] <= 1
    assert e.predict(features).shape == (400,)


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
    rows, facts = runner.policy_rows(archive, runner.Config())
    assert facts["proposal_rows"] == 5
    assert facts["rollout_rows"] <= facts["proposal_rows"]
    assert len(rows) == facts["proposal_rows"] + facts["rollout_rows"]


def test_two_tiny_cycles_end_to_end(tmp_path):
    config = runner.Config(
        seed=0,
        cycles=2,
        proposals=20,
        calibration=12,
        test_fresh=8,
        test_archive=8,
        tracked_new=20,
        architecture=(8, 16, 16),
        train_device="cpu",
        budget=4,
        n=2,
        behaviours=SMALL,
    )
    reports = runner.run(config, tmp_path, log=lambda line: None)
    assert [r["cycle"] for r in reports] == [1, 2]
    for report in reports:
        assert report["test_left_archive_unchanged"]
        assert report["rows"]["rollout_rows"] <= report["rows"]["proposal_rows"]
        assert set(report["test"]["heads"]) == {"F3", "archive"}
        assert set(report["test"]["baselines"]) == {"constant", "distance"}
        assert "unseen_pass" in report["rnd"]
        assert report["policy"]["steps"] >= 1
    assert reports[1]["propose"].get("from_archive", 0) >= 1
    assert (tmp_path / "cycle-2" / "proposals.jsonl.gz").exists()
    assert reports[1]["calibration"]["rules"].get(selector.REATTEMPT, 0) >= 1
    saved = json.loads((tmp_path / "cycle-2" / "report.json").read_text())
    assert saved["cycle"] == 2
    assert (tmp_path / "cycle-2" / "policy" / "model.safetensors").exists()
    assert (
        LoopArchive.load(tmp_path / "cycle-2" / "archive.pkl.gz").summary()
        == reports[1]["archive"]
    )
