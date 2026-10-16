"""Synthetic/API tests only; runtime/source mocks do not qualify real provenance."""

import dataclasses
import hashlib
import json
import os
import pickle
import subprocess
import sys
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import list_hash_memo_run as run
from wandering_light.stack_machine import Bool, Int, Lst, Mark, Sym, value_key


class HashProxy:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return self.value


def old_hash(value):
    """Rebuild old field hashes recursively; never read any memo slot."""
    if isinstance(value, Lst):
        return hash((tuple(HashProxy(old_hash(v)) for v in value.items),))
    if isinstance(value, Mark):
        return hash(())
    return hash((value.name if isinstance(value, Sym) else value.value,))


def values():
    leaves = [
        Int(0),
        Int(1),
        Int(-1),
        Int(sys.hash_info.modulus),
        Bool(False),
        Bool(True),
        Sym("add"),
        Sym("["),
        Mark(),
        Lst(()),
    ]
    return [
        *leaves,
        Lst(tuple(leaves)),
        Lst((Lst(tuple(leaves)), Lst(tuple(leaves)))),
        Lst(tuple(Int(i) for i in range(100))),
        Lst((Sym("日本語"), Int(2**2000))),
    ]


@pytest.mark.parametrize("index", range(14))
def test_original_hash_schema_and_copy_contract(index):
    v = values()[index]
    expected = old_hash(v)
    before = (repr(v), value_key(v), dataclasses.asdict(v), dataclasses.astuple(v))
    for _ in range(3):
        assert hash(v) == expected
    for clone in (
        copy(v),
        deepcopy(v),
        pickle.loads(pickle.dumps(v)),
        dataclasses.replace(v),
    ):
        assert clone == v and hash(clone) == expected
    assert before == (
        repr(v),
        value_key(v),
        dataclasses.asdict(v),
        dataclasses.astuple(v),
    )
    if isinstance(v, Lst):
        assert [f.name for f in dataclasses.fields(v)] == ["items"]
        assert v.__match_args__ == ("items",)
        assert v.__getstate__() == [v.items]
        with pytest.raises(dataclasses.FrozenInstanceError):
            v.items = ()


def test_equal_but_distinct_and_unequal_hash_collision():
    a = Lst((Int(0), Sym("add")))
    b = Lst((Int(0), Sym("add")))
    assert a is not b and a == b and hash(a) == hash(b)
    x, y = Lst((Int(0),)), Lst((Int(sys.hash_info.modulus),))
    assert x != y and hash(x) == hash(y)
    assert len({x, y}) == 2
    assert Int(0) != Bool(False)
    assert len({Int(0), Bool(False)}) == 2


def test_failed_hash_never_publishes_memo():
    bad = Lst(
        ([],)
    )  # Deliberately outside typed union: preserve original hash failure.
    for _ in range(2):
        with pytest.raises(TypeError):
            hash(bad)
        assert getattr(bad, "_cached_hash", None) is None


def probe(root, seed, incoming=None):
    script = """
import base64,json,pickle
from wandering_light.stack_machine import Lst,Sym,Int,value_key
fresh=Lst((Sym("salted-name"),Lst((Int(7),Sym("add")))))
current=pickle.loads(base64.b64decode(INCOMING)) if INCOMING else fresh
assert current==fresh and hash(current)==hash(fresh)
hash(current)
assert current.__getstate__()==[current.items]
print(json.dumps({"pickle":base64.b64encode(pickle.dumps(current)).decode(),
                  "hash":hash(current),"key":value_key(current)}))
""".replace("INCOMING", repr(incoming))
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "PYTHONHASHSEED": str(seed),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return json.loads(
        subprocess.check_output(
            [sys.executable, "-c", script], cwd=root, env=env, text=True
        )
    )


def test_cross_seed_pickle_and_optional_genuine_source_directions():
    # With LIST_HASH_CONTROL_ROOT set, also test actual old/new source directions.
    control = Path(os.environ.get("LIST_HASH_CONTROL_ROOT", run.ROOT))
    a = probe(control, 17)
    b = probe(run.ROOT, 29, a["pickle"])
    c = probe(control, 43, b["pickle"])
    assert a["key"] == b["key"] == c["key"]
    # Same source schema yields identical pickle bytes, despite different salted hashes.
    assert a["pickle"] == b["pickle"] == c["pickle"]


def test_selector_fixed_salt_first_source_and_invalid_no_refill():
    tasks = [
        SimpleNamespace(
            split="train",
            skeleton=str(i // 2),
            behaviour=str(i),
            inputs=("[ 1 ]",),
            outputs=("[ 2 ]",),
        )
        for i in range(60)
    ]
    identities, shown = run.select_panel(tasks)
    expected = sorted(
        {t.skeleton for t in tasks},
        key=lambda s: (
            hashlib.sha256(b"list-hash-memo-v1\0" + s.encode()).hexdigest(),
            s,
        ),
    )[:24]
    assert [x["skeleton"] for x in identities] == expected
    assert [x["source_index"] for x in identities] == [2 * int(s) for s in expected]
    assert all(set(x) == {"source_index", "inputs", "outputs"} for x in shown)
    with pytest.raises(ValueError):
        run.select_panel(tasks[:20])
    tasks[0].split = "eval"
    with pytest.raises(ValueError):
        run.select_panel(tasks)


def synthetic():
    shown = [
        {"source_index": i, "inputs": ["[ 2 ]"], "outputs": ["[ 3 ]"]}
        for i in range(24)
    ]
    pins = {a: {"root": f"/SYNTHETIC/{a}", "head": a} for a in ("control", "candidate")}
    for pin in pins.values():
        pin["files"] = {
            str(Path(pin["root"]) / (m.replace(".", "/") + ".py")): "MOCK_SHA"
            for m in run.MODULES
        }
    model, limits = {"MOCK_MODEL": True}, {"MOCK_RUNTIME": True}
    cells, clocks = {}, {}
    for arm, label in run.ORDER:
        name = f"{arm}-{label}"
        cells[name] = {
            "complete": True,
            "arm": arm,
            "label": label,
            "source": pins[arm],
            "checkpoint": model,
            "runtime": limits,
            "host": run.platform.node(),
            "threads": 1,
            "interop": 1,
            "hash_seed": "0",
            "device": "cpu",
            "model_parameter_dtypes": ["torch.float32"],
            "thread_env": dict.fromkeys(run.THREAD_ENV, "1"),
            "module_origins": {
                m: str(Path(pins[arm]["root"]) / (m.replace(".", "/") + ".py"))
                for m in run.MODULES
            },
            "rows": [
                {
                    "source_index": i,
                    "program": ["1", "add"],
                    "termination": "fit",
                    "seconds": 1.0,
                    "solve_seconds": 1.0,
                    "executions": 2,
                    "fuel": 2,
                    "expansions": 2,
                    "depth": 2,
                    "pruned": False,
                }
                for i in range(24)
            ],
        }
        cells[name]["callable_origins"] = {
            name: str(Path(pins[arm]["root"]) / "wandering_light/stack_machine.py")
            for name in run.FUNCTIONS
        }
        cells[name]["list_hash_code"] = (
            "<string>"
            if arm == "control"
            else str(Path(pins[arm]["root"]) / "wandering_light/stack_machine.py")
        )
        clocks[name] = 9.0 if arm == "candidate" else 10.0
    return cells, clocks, shown, pins, model, limits


def test_exact_boundary_and_substantive_counter_difference():
    args = synthetic()
    result = run.paired_summary(*args)
    assert result["reading"] == "QUALIFY_RELEASE_COHORT"
    assert result["pairs"][0]["process_ratio_exact"] == [9, 10]
    args[0]["candidate-29"]["rows"][0]["executions"] = 3
    assert run.paired_summary(*args)["reading"] == "PARK_FIXED_CONSTRUCTION"
    args[0]["candidate-29"]["rows"][0]["executions"] = 2
    args[1]["candidate-43"] = 9.000000000000002
    assert run.paired_summary(*args)["reading"] == "PARK_FIXED_CONSTRUCTION"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "order",
        "threads",
        "origin",
        "deadline",
        "bad_fit",
        "negative",
        "nan",
        "zero_clock",
    ],
)
def test_invalid_results_are_not_compatibility_evidence(change):
    args = synthetic()
    cell = args[0]["candidate-17"]
    if change == "missing":
        args[0].pop("candidate-43")
    elif change == "order":
        cell["rows"].reverse()
    elif change == "threads":
        cell["threads"] = 2
    elif change == "origin":
        cell["module_origins"][run.MODULES[0]] = "/FOREIGN/module.py"
    elif change == "deadline":
        cell["rows"][0].update(program=None, termination="wall_clock_cap")
    elif change == "bad_fit":
        cell["rows"][0]["program"] = ["0", "add"]
    elif change == "negative":
        cell["rows"][0]["fuel"] = -1
    elif change == "nan":
        cell["rows"][0]["seconds"] = float("nan")
    else:
        args[1]["candidate-17"] = 0.0
    with pytest.raises(ValueError):
        run.paired_summary(*args)


@pytest.mark.parametrize("fail", [False, True])
def test_phase_real_helper_path_with_explicit_source_runtime_process_mocks(
    tmp_path, monkeypatch, fail
):
    args = synthetic()
    cells, _clocks, shown, pins, model, limits = args
    payload = tmp_path / "DUMMY_TRAIN"
    payload.write_text("not real data")
    monkeypatch.setattr(run, "MAIN", tmp_path)
    monkeypatch.setattr(run, "TRAIN_SHA", run.sha(payload))
    monkeypatch.setattr(run, "runtime_limits", lambda: limits)
    monkeypatch.setattr(run, "validate_sources", lambda *_: pins)
    monkeypatch.setattr(run, "source_info", lambda p: pins[str(p)])
    monkeypatch.setattr(run, "checkpoint_pins", lambda _: model)
    monkeypatch.setattr(
        run, "prepare_inputs", lambda _: (payload, tmp_path, [{}] * 24, shown)
    )
    monkeypatch.setattr(run, "host_snapshot", lambda: {"MOCK_HOST": True})
    events = []
    original = run.paired_summary

    def child(cmd, **kwargs):
        arm = cmd[cmd.index("--arm") + 1]
        label = int(cmd[cmd.index("--label") + 1])
        assert kwargs["env"]["PYTHONPATH"] == pins[arm]["root"]
        assert kwargs["env"]["PYTHONHASHSEED"] == "0"
        assert "ANTHROPIC_API_KEY" not in kwargs["env"]
        assert "OPENAI_API_KEY" not in kwargs["env"]
        events.append((arm, label))
        if fail and label == 29:
            raise RuntimeError("synthetic child failure")
        cell = deepcopy(cells[f"{arm}-{label}"])
        cell["shown_sha256"] = run.sha(cmd[cmd.index("--shown") + 1])
        run.save(cmd[cmd.index("--out") + 1], cell)

    def analyze(*arguments):
        assert tuple(events) == run.ORDER
        events.append("analysis_after_all_timing")
        return original(*arguments)

    monkeypatch.setattr(run.subprocess, "run", child)
    monkeypatch.setattr(run, "paired_summary", analyze)
    out = tmp_path / "checkpoints/phase"
    if fail:
        with pytest.raises(RuntimeError):
            run.run_phase(out, "control", "candidate", "control", "candidate")
        assert (out / "INCOMPLETE.json").exists() and not (
            out / "summary.json"
        ).exists()
    else:
        run.run_phase(out, "control", "candidate", "control", "candidate")
        assert run.read(out / "summary.json")["complete"]
        assert events[-1] == "analysis_after_all_timing"


def test_actual_production_search_and_untrained_scorer_match_canonical_hash(
    monkeypatch,
):
    """Software comparison only; this patched class is NEVER a timing control."""
    from wandering_light.rollout_policy import PolicyArchitecture
    from wandering_light.stack_beam import BeamObserver, beam_solve
    from wandering_light.stack_machine import run_body
    from wandering_light.stack_policy import PolicyScorer, StackPairPolicy

    model = StackPairPolicy(
        PolicyArchitecture(byte_embedding=4, conv_channels=8, hidden=8)
    )
    task = SimpleNamespace(
        inputs=("[ (1 2) 2 ]", "[ (3 4) 3 ]"), outputs=("[ (1 2) 4 ]", "[ (3 4) 9 ]")
    )

    class Watch(BeamObserver):
        def __init__(self):
            self.scored = []

        def layer_scored(self, depth, probs, legal):
            self.scored.append((depth, probs.tobytes(), legal.tobytes()))

    class Scorer(PolicyScorer):
        def __init__(self):
            super().__init__(model, "cpu")
            self.raw = []

        def logits(self, blocks):
            result = super().logits(blocks)
            self.raw.append((result.tobytes(), self.last_features.tobytes()))
            return result

    def solve():
        run_body.cache_clear()
        watch = Watch()
        scorer = Scorer()
        result = beam_solve(
            task,
            scorer,
            beam_width=64,
            max_depth=3,
            guard="both",
            budget=5000,
            max_state_atoms=16384,
            observer=watch,
        )
        return (
            {k: v for k, v in dataclasses.asdict(result).items() if k != "seconds"},
            watch.scored,
            scorer.raw,
        )

    actual = solve()
    assert actual[0]["program"] is not None and actual[1] and actual[2]
    with monkeypatch.context() as m:
        m.setattr(Lst, "__hash__", old_hash)
        reference = solve()
    assert actual == reference


@pytest.mark.parametrize("bad", [None, "head", "production", "driver", "same"])
def test_two_genuine_sources_validation(tmp_path, monkeypatch, bad):
    control, candidate = tmp_path / "control", tmp_path / "candidate"
    for p in (control, candidate):
        (p / "experiments").mkdir(parents=True)
        (p / "experiments/list_hash_memo_run.py").write_bytes(
            Path(run.__file__).read_bytes()
        )
    monkeypatch.setattr(run, "ROOT", candidate)
    monkeypatch.setattr(run, "source_info", lambda p: {"root": str(p), "head": p.name})

    def git(_root, *args):
        if "--name-only" in args:
            return "wandering_light/stack_machine.py"
        return "unregistered production change" if bad == "production" else ""

    monkeypatch.setattr(run, "git", git)
    if bad == "driver":
        (control / "experiments/list_hash_memo_run.py").write_text("different driver")
    if bad:
        with pytest.raises(ValueError):
            run.validate_sources(
                candidate if bad == "same" else control,
                candidate,
                "wrong" if bad == "head" else "control",
                "candidate",
            )
    else:
        assert (
            run.validate_sources(control, candidate, "control", "candidate")["control"][
                "head"
            ]
            == "control"
        )


def test_source_info_rejects_dirty_before_any_source_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "git", lambda *_: "dirty")
    with pytest.raises(ValueError, match="dirty"):
        run.source_info(tmp_path)


@pytest.mark.parametrize("bad", [None, "memory", "swap", "runtime"])
def test_actual_control_reader_rejects_mismatched_enforcement(monkeypatch, bad):
    memory = "123" if bad == "memory" else str(12 * 1024**3)
    swap = "1" if bad == "swap" else "0"
    runtime = "10min" if bad == "runtime" else "20min"
    cg = "/user.slice/synthetic.service"
    files = {
        "/proc/self/cgroup": "0::" + cg + "\n",
        "/sys/fs/cgroup" + cg + "/memory.max": memory,
        "/sys/fs/cgroup" + cg + "/memory.swap.max": swap,
    }
    monkeypatch.setattr(Path, "read_text", lambda self, **_: files[str(self)])
    monkeypatch.setattr(
        run.subprocess,
        "check_output",
        lambda *_a, **_k: (
            f"MemoryMax={memory}\nMemorySwapMax={swap}\nRuntimeMaxUSec={runtime}\n"
        ),
    )
    if bad:
        with pytest.raises(ValueError):
            run.runtime_limits()
    else:
        assert run.runtime_limits()["memory_max"] == str(12 * 1024**3)


def test_cell_actual_function_with_dummy_model_and_explicit_runtime_source_mocks(
    tmp_path, monkeypatch
):
    import torch

    from wandering_light import stack_beam, stack_policy
    from wandering_light.rollout_policy import PolicyArchitecture

    model = stack_policy.StackPairPolicy(
        PolicyArchitecture(byte_embedding=4, conv_channels=8, hidden=8)
    )
    for f in ("set_num_threads", "set_num_interop_threads"):
        monkeypatch.setattr(torch, f, lambda _n: None)
    for f in ("get_num_threads", "get_num_interop_threads"):
        monkeypatch.setattr(torch, f, lambda: 1)
    for key in run.THREAD_ENV:
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setattr(run, "runtime_limits", lambda: {"MOCK_RUNTIME": True})
    monkeypatch.setattr(run, "source_info", lambda _: {"MOCK_SOURCE": True})
    monkeypatch.setattr(run, "checkpoint_pins", lambda _: {"MOCK_MODEL": True})
    monkeypatch.setattr(stack_policy, "load_policy", lambda *_a, **_k: model)
    monkeypatch.setattr(run.gc, "freeze", lambda: None)
    seen = []

    def search(_task, _scorer, **kwargs):
        assert kwargs == {
            "beam_width": 64,
            "max_depth": 48,
            "guard": "both",
            "max_state_atoms": 16384,
            "budget": 100000,
            "budget_unit": "executions",
            "deadline": None,
        }
        seen.append(True)
        return stack_beam.BeamResult(("1", "add"), "fit", 0.0, 2, 2, 2, 2, False)

    monkeypatch.setattr(stack_beam, "beam_solve", search)
    shown = synthetic()[2]
    run.save(tmp_path / "shown.json", shown)
    run.run_cell(
        tmp_path / "shown.json",
        tmp_path / "DUMMY",
        tmp_path / "cell.json",
        "candidate",
        17,
    )
    data = run.read(tmp_path / "cell.json")
    assert data["complete"] and len(seen) == len(data["rows"]) == 24


def test_checkpoint_pin_guard_uses_actual_bytes(tmp_path, monkeypatch):
    path = tmp_path / "DUMMY_WEIGHT"
    path.write_bytes(b"untrained dummy bytes")
    monkeypatch.setattr(run, "CHECKPOINT_SHA", {path.name: run.sha(path)})
    assert run.checkpoint_pins(tmp_path)[path.name] == run.sha(path)
    path.write_bytes(b"changed dummy bytes")
    with pytest.raises(ValueError):
        run.checkpoint_pins(tmp_path)
