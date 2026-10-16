"""#184 synthetic contracts only: no corpus, pretrained model or profile access."""

from __future__ import annotations

import hashlib
import struct
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import policy_item_reuse_run as runner
from experiments.policy_item_reuse import (
    ItemRecorder,
    RecordingScorer,
    eligibility,
    replay_counts,
)
from wandering_light.stack_beam import beam_solve
from wandering_light.stack_policy import PAD, PolicyScorer


def block(values):
    return np.asarray(values, dtype=np.int16)


def test_order_eviction_and_task_reset():
    rec = ItemRecorder(capacity=2)
    rec.observe([block([[1], [2], [1]])])
    rec.observe([block([[2], [3], [1]])])
    assert rec.totals == {
        "calls": 2,
        "items": 6,
        "positions": 6,
        "hits": 2,
        "hit_positions": 2,
        "within_call_hits": 1,
        "cross_call_hits": 1,
    }
    assert replay_counts(rec.calls, capacity=2) == rec.totals
    fresh = ItemRecorder(capacity=2)
    fresh.observe([block([[1]])])
    assert fresh.totals["hits"] == 0


def test_registered_capacity_boundary():
    rec = ItemRecorder()
    rec.observe([np.arange(4096, dtype=np.int16).reshape(-1, 1)])
    rec.observe([block([[0], [4096], [1]])])
    assert rec.totals["hits"] == 1  # 0 is touched; 1 is evicted before revisiting.
    assert len(rec.cache) == 4096
    assert replay_counts(rec.calls) == rec.totals


def test_padding_order_width_and_canonical_bytes():
    rec = ItemRecorder()
    rec.observe([block([[1], [1]]), block([[1, PAD]])])
    expected = hashlib.sha256(struct.pack("<Qqq", 2, 1, PAD)).hexdigest()
    assert rec.calls == [{"n": 3, "width": 2, "keys": [expected] * 3}]
    assert rec.totals["hit_positions"] == 4
    rec.observe([np.asarray([[1]], dtype=">i8")])
    assert rec.calls[-1]["keys"] == [
        hashlib.sha256(struct.pack("<Qq", 1, 1)).hexdigest()
    ]
    assert rec.totals["hits"] == 2  # A different padded width is a miss.
    assert replay_counts(rec.calls) == rec.totals


def test_active_digest_collision_aborts():
    rec = ItemRecorder(digest=lambda _key: "0" * 64)
    with pytest.raises(ValueError, match="distinct active"):
        rec.observe([block([[1], [2]])])


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        ItemRecorder(capacity)
    with pytest.raises(ValueError):
        replay_counts([], capacity)


@pytest.mark.parametrize(
    "blocks",
    [
        [],
        [block([])],
        [np.empty((0, 2), dtype=np.int16)],
        [block([[]])],
        [np.ones((1, 2))],
    ],
)
def test_invalid_blocks(blocks):
    with pytest.raises(ValueError):
        ItemRecorder().observe(blocks)


@pytest.mark.parametrize(
    "record",
    [
        {"n": 0, "width": 1, "keys": []},
        {"n": True, "width": 1, "keys": ["0" * 64]},
        {"n": 1, "width": 0, "keys": ["0" * 64]},
        {"n": 2, "width": 1, "keys": ["0" * 64]},
        {"n": 1, "width": 1, "keys": ["NOT_A_DIGEST"]},
        {"n": 1, "width": 1, "keys": ["0" * 64], "extra": 1},
    ],
)
def test_invalid_raw_records(record):
    with pytest.raises(ValueError):
        replay_counts([record])


def test_raw_active_key_cannot_change_width():
    with pytest.raises(ValueError, match="different padded width"):
        replay_counts(
            [
                {"n": 1, "width": 1, "keys": ["a" * 64]},
                {"n": 1, "width": 2, "keys": ["a" * 64]},
            ]
        )


def totals(hit, positions=100):
    return {"hit_positions": hit, "positions": positions}


def test_exact_eligibility_boundaries():
    assert eligibility([totals(25)] * 3)["reading"] == "REUSE_OPPORTUNITY"
    exact_boundary = eligibility([totals(25), totals(30), totals(25)])
    assert exact_boundary["every_fraction_at_least_quarter"]
    assert not exact_boundary["range_below_five_points"]
    assert exact_boundary["range_exact"] == [1, 20]
    assert exact_boundary["reading"] == "NOT_JUSTIFIED"
    assert (
        eligibility([totals(249, 1000), totals(250, 1000), totals(250, 1000)])[
            "reading"
        ]
        == "NOT_JUSTIFIED"
    )
    for bad in [totals(0, 0), totals(-1), totals(101), totals(25.0)]:
        with pytest.raises(ValueError):
            eligibility([bad] * 3)


def test_scorer_delegates_original_blocks_outputs_and_features(monkeypatch):
    calls = []
    features = np.arange(4).reshape(2, 2)
    raw = np.arange(56, dtype=np.float32).reshape(2, 28) / 20

    def original(self, blocks):
        calls.append(blocks)
        self.last_features = features
        return raw

    monkeypatch.setattr(PolicyScorer, "logits", original)
    scorer = object.__new__(RecordingScorer)
    scorer.recorder = ItemRecorder()
    blocks = [block([[1], [1]]), block([[2]])]
    assert scorer.logits(blocks) is raw
    assert calls == [blocks] and calls[0] is blocks
    assert scorer.last_features is features
    legal = np.ones((2, 28), dtype=bool)
    legal[0, 1] = False
    legal[1, 2] = False
    control = object.__new__(PolicyScorer)
    expected = control.log_probs(blocks, legal)
    got = scorer.log_probs(blocks, legal)
    np.testing.assert_array_equal(got, expected)
    assert scorer.last_features is features
    assert scorer.recorder.totals["calls"] == 2


@pytest.mark.parametrize(
    ("inputs", "outputs"),
    [(("2 3", "3 4"), ("12", "24")), (("(1 2)", "(2 3)"), ("999", "999"))],
)
def test_observer_does_not_change_manual_machine_search(monkeypatch, inputs, outputs):
    def original(self, blocks):
        self.last_features = np.zeros((len(blocks), 2))
        return np.broadcast_to(-np.arange(28, dtype=np.float32) / 20, (len(blocks), 28))

    monkeypatch.setattr(PolicyScorer, "logits", original)
    control = object.__new__(PolicyScorer)
    control.model = SimpleNamespace(encoding="machine")
    observed = object.__new__(RecordingScorer)
    observed.model = control.model
    observed.recorder = ItemRecorder()
    task = runner.ShownTask(inputs, outputs)
    kwargs = {"beam_width": 8, "max_depth": 3, "guard": "both", "max_state_atoms": 32}
    a, b = beam_solve(task, control, **kwargs), beam_solve(task, observed, **kwargs)
    assert {k: v for k, v in asdict(a).items() if k != "seconds"} == {
        k: v for k, v in asdict(b).items() if k != "seconds"
    }
    assert observed.recorder.calls
    assert replay_counts(observed.recorder.calls) == observed.recorder.totals


def synthetic_cells():
    shown = [{"source_index": i, "inputs": ["2"], "outputs": ["3"]} for i in range(24)]
    rec = ItemRecorder()
    rec.observe([block([[1], [1]])])
    rows = [
        {
            "source_index": i,
            "program": ["1", "add"],
            "termination": "fit",
            "executions": 2,
            "fuel": 2,
            "solve_seconds": 0.25,
            "observed": rec.totals,
            "calls": rec.calls,
        }
        for i in range(24)
    ]
    cells = {
        str(label): {
            "complete": True,
            "label": label,
            "threads": 1,
            "interop": 1,
            "host": runner.platform.node(),
            "rows": deepcopy(rows),
        }
        for label in runner.LABELS
    }
    return cells, shown


def test_summary_replays_counts_and_shown_without_hidden_grading():
    cells, shown = synthetic_cells()
    result = runner.summarize(cells, shown)
    assert result["reading"] == "REUSE_OPPORTUNITY"
    assert result["fractions_exact"] == [[1, 2]] * 3
    assert all(
        c["shown_fits"] == 24 and c["solve_seconds"] == 6
        for c in result["cells"].values()
    )
    cells["17"]["rows"][0]["observed"]["hits"] += 1
    with pytest.raises(ValueError, match="counters disagree"):
        runner.summarize(cells, shown)


@pytest.mark.parametrize("change", ["program", "rows", "thread", "host", "terminal"])
def test_summary_rejects_invalid_saved_cell(change):
    cells, shown = synthetic_cells()
    if change == "program":
        cells["17"]["rows"][0]["program"] = ["drop"]
    elif change == "rows":
        cells["17"]["rows"].reverse()
    elif change == "thread":
        cells["17"]["threads"] = 2
    elif change == "host":
        cells["17"]["host"] = "FOREIGN_HOST"
    else:
        cells["17"]["rows"][0]["termination"] = "max_depth"
    with pytest.raises(ValueError):
        runner.summarize(cells, shown)


def test_selector_original_row_and_new_hash_salt():
    tasks = [
        SimpleNamespace(
            split="train", skeleton=s, behaviour=s, inputs=("2",), outputs=("3",)
        )
        for s in ["z", "a", "z", "b"]
    ]
    identities, shown = runner.select_panel(tasks, count=3)
    expected = sorted(
        ["z", "a", "b"],
        key=lambda s: (
            hashlib.sha256(b"policy-item-reuse-v1\0" + s.encode()).hexdigest(),
            s,
        ),
    )
    assert [i["skeleton"] for i in identities] == expected
    assert {i["source_index"] for i in identities} == {0, 1, 3}
    assert all(set(s) == {"source_index", "inputs", "outputs"} for s in shown)
    tasks[0].split = "eval"
    with pytest.raises(ValueError):
        runner.select_panel(tasks, count=3)


@pytest.mark.parametrize("fail_cell", [False, True])
def test_actual_phase_path_with_explicit_acquisition_process_mocks(
    monkeypatch, tmp_path, fail_cell
):
    """Exercise parent orchestration; mocked acquisition/runtime is not real provenance."""
    cells, shown = synthetic_cells()
    payload = tmp_path / "dummy-train"
    payload.write_text("synthetic")
    monkeypatch.setattr(runner, "MAIN", tmp_path)
    monkeypatch.setattr(runner, "TRAIN_SHA", runner.sha(payload))
    monkeypatch.setattr(
        runner, "prepare_inputs", lambda _out: (payload, tmp_path, [{}] * 24, shown)
    )
    monkeypatch.setattr(runner, "provenance", lambda _path: {"MOCK_SOURCE": True})
    monkeypatch.setattr(runner, "host_snapshot", lambda: {"MOCK_HOST": True})
    events = []
    original_summary = runner.summarize

    def process(cmd, **kwargs):
        label = cmd[cmd.index("--label") + 1]
        assert kwargs["env"]["PYTHONPATH"] == str(runner.ROOT)
        assert all(kwargs["env"][k] == "1" for k in runner.THREAD_ENV)
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        assert "OPENAI_API_KEY" not in kwargs["env"]
        events.append(label)
        if fail_cell and label == "29":
            raise RuntimeError("synthetic child failure")
        data = deepcopy(cells[label])
        data["shown_sha256"] = runner.sha(cmd[cmd.index("--shown") + 1])
        runner.save(cmd[cmd.index("--out") + 1], data)

    def summarize(loaded, panel):
        assert events == ["17", "29", "43"]
        events.append("after_all_timing")
        return original_summary(loaded, panel)

    monkeypatch.setattr(runner.subprocess, "run", process)
    monkeypatch.setattr(runner, "summarize", summarize)
    out = tmp_path / "checkpoints" / "profile"
    if fail_cell:
        with pytest.raises(RuntimeError):
            runner.run_phase(out)
        assert runner.read(out / "INCOMPLETE.json")["complete"] is False
        assert not (out / "summary.json").exists()
    else:
        runner.run_phase(out)
        assert runner.read(out / "summary.json")["reading"] == "REUSE_OPPORTUNITY"
        assert events[-1] == "after_all_timing"


def test_cell_path_uses_fixed_controls_task_local_recorders(monkeypatch, tmp_path):
    """Only dummy model and stub search; real ItemRecorder/scorer delegation are used."""
    import torch

    from wandering_light import stack_beam, stack_policy

    @dataclass
    class Result:
        program: tuple = ("1", "add")
        termination: str = "fit"
        executions: int = 2
        fuel: int = 2

    monkeypatch.setattr(torch, "set_num_threads", lambda _n: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _n: None)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(
        stack_policy, "load_policy", lambda *_a, **_kw: torch.nn.Identity()
    )
    monkeypatch.setattr(runner.gc, "freeze", lambda: None)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 100.0)
    monkeypatch.setattr(
        PolicyScorer, "logits", lambda _self, blocks: np.zeros((len(blocks), 28))
    )
    for key in runner.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    seen = []

    def search(_task, scorer, **kwargs):
        assert kwargs.keys() == {
            "beam_width",
            "max_depth",
            "guard",
            "max_state_atoms",
            "deadline",
        }
        assert (
            kwargs["beam_width"],
            kwargs["max_depth"],
            kwargs["guard"],
            kwargs["max_state_atoms"],
        ) == (64, 48, "both", 16384)
        assert kwargs["deadline"] == 105.0
        seen.append(scorer.recorder)
        assert scorer.recorder.totals["items"] == 0
        scorer.logits([block([[1], [1]])])
        return Result()

    monkeypatch.setattr(stack_beam, "beam_solve", search)
    _, shown = synthetic_cells()
    runner.save(tmp_path / "shown.json", shown)
    runner.run_cell(
        tmp_path / "shown.json", tmp_path / "DUMMY", tmp_path / "cell.json", 17
    )
    result = runner.read(tmp_path / "cell.json")
    assert result["complete"] and len(seen) == len({id(x) for x in seen}) == 24
    assert all(row["observed"]["hits"] == 1 for row in result["rows"])


def test_real_synthetic_cnn_preserves_bits_and_observes_actual_collate_rows():
    from wandering_light.rollout_policy import PolicyArchitecture
    from wandering_light.stack_machine import Int
    from wandering_light.stack_policy import StackPairPolicy, collate

    model = StackPairPolicy(
        PolicyArchitecture(byte_embedding=4, conv_channels=8, hidden=8)
    )
    control = PolicyScorer(model, "cpu")
    observed = RecordingScorer(model, "cpu")
    observed.recorder = ItemRecorder()
    encode = control.encoder(((Int(12),), (Int(24),)))
    blocks = [
        encode(((Int(2), Int(3)), (Int(3), Int(4)))),
        encode(((Int(3), Int(4)), (Int(4), Int(5)))),
    ]
    raw = control.logits(blocks)
    features = control.last_features.copy()
    result = observed.logits(blocks)
    assert raw.dtype == result.dtype and raw.tobytes() == result.tobytes()
    assert observed.last_features.tobytes() == features.tobytes()
    dense, _counts, _mask = collate(blocks, "cpu")
    expected = [
        hashlib.sha256(
            struct.pack("<Q", dense.shape[1]) + row.numpy().astype("<i8").tobytes()
        ).hexdigest()
        for row in dense
    ]
    assert observed.recorder.calls[0] == {
        "n": len(dense),
        "width": dense.shape[1],
        "keys": expected,
    }
    legal = np.ones((2, 28), dtype=bool)
    legal[0, 3] = False
    legal[1, 7] = False
    baseline = control.log_probs(blocks, legal)
    recorded = observed.log_probs(blocks, legal)
    assert recorded.dtype == baseline.dtype and recorded.tobytes() == baseline.tobytes()
    assert replay_counts(observed.recorder.calls) == observed.recorder.totals
