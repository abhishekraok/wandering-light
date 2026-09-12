"""Teaching claims, real-machine parity, and the workshop API contract."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from wandering_light.stack_machine import Int, MARK_OPS, step_raw, value_key
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS
from wandering_light.stack_workshop.engine import (
    FUEL, MACROS, compile_program, execute, inspect_op, stack_from,
)
from wandering_light.stack_workshop.server import Handler, ThreadingHTTPServer, api


@pytest.mark.parametrize("lesson", LESSONS, ids=lambda x: x["id"])
def test_curriculum_references_pass_independent_examples(lesson):
    result = execute(lesson["solution"], lesson["examples"] + lesson["heldout"])
    assert result["matches"] == result["count"] == 6
    assert api("/api/verify", {"lesson": lesson["id"], "program": lesson["solution"]})["passed"]


def test_references_stay_out_of_initial_context():
    context = api("/api/context", {})
    assert len(context["lessons"]) == 14
    assert all("solution" not in task and "hints" not in task and "heldout" not in task for task in context["lessons"])
    assert len(context["macros"]) == len(MACROS) >= 100


@pytest.mark.parametrize("program,inputs", [
    ("swap sub", [2, 9]), ("[ dup mul ] map", [[-2, 0, 3]]),
    ("0 [ add ] fold", [[2, 4, -1]]), ("[ 1 add ] dip", [3, 12]),
    ("[ 0 lt? false eq? ] filter", [[-1, 0, 4]]),
    ("[ dup 0 lt? ] [ 0 swap sub ] [ ] ifte", [-3]),
])
def test_each_recorded_prefix_agrees_with_canonical_machine(program, inputs):
    by_name = {o.name: o for o in MARK_OPS}
    stack = stack_from(inputs)
    result = execute(program, [{"input": inputs, "output": []}])
    spent = 0
    for index, token in enumerate(result["tokens"]):
        stack, cost = step_raw(by_name[token["op"]], stack, FUEL - spent, 0)
        spent += cost
        frame = result["examples"][0]["frames"][index + 1]
        assert frame["fuel"] == spent
        from wandering_light.stack_workshop.engine import snapshot
        assert frame["stack"] == snapshot(stack)


def test_macro_source_spans_and_nested_expansion():
    program = "# an explanation\n[ sq ] map sum"
    tokens = compile_program(program)
    assert [t["op"] for t in tokens] == "[ dup mul ] map 0 [ add ] fold".split()
    assert all(program[t["start"]:t["end"]] == "sq" for t in tokens if t["macro"] == "sq")
    assert execute(program, [{"input": [[2, -3]], "output": [13]}])["matches"] == 1


def test_warm_cache_does_not_hide_nested_execution():
    examples = [{"input": [[2, 4]], "output": [[4, 8]]}]
    program = "[ dup add ] map"
    execute(program, examples)
    first = inspect_op(program, examples, 4)
    second = inspect_op(program, examples, 4)
    assert [f["op"] for f in first["frames"]] == ["dup", "add", "dup", "add", "map"]
    assert first == second
    assert not first["truncated"]


def test_error_halts_only_its_example_and_keeps_last_state():
    result = execute("uncons", [{"input": [[]], "output": []}, {"input": [[1]], "output": [[], 1]}])
    assert result["examples"][0]["error"] == "uncons: list is empty"
    assert result["examples"][0]["frames"][0]["stack"] == result["examples"][0]["frames"][1]["stack"]
    assert result["examples"][1]["matched"]


def test_custom_fit_is_labeled_as_example_evidence():
    result = api("/api/verify", {"lesson": "sandbox", "program": "dup add", "examples": [{"input": [2], "output": [4]}]})
    assert result["passed"] and result["evidence"] == "examples"
    assert result["extra_count"] == 0


def test_extra_tests_expose_a_shortcut():
    # Shown inputs avoid 1; the extra test is deliberately decisive.
    old = BY_ID["first-light"]
    BY_ID["test-shortcut"] = {**old, "examples": [{"input": [2], "output": [4]}], "heldout": [{"input": [1], "output": [3]}]}
    try:
        result = api("/api/verify", {"lesson": "test-shortcut", "program": "dup add"})
        assert result["shown"] == 1 and not result["passed"]
        assert result["extra"] == 0 and len(result["failures"]) == 1
    finally:
        del BY_ID["test-shortcut"]


@pytest.mark.parametrize("data", [[], [{"input": [1.5], "output": [1]}], [{"input": ["oops"], "output": []}], [{"input": "[ add ]", "output": []}], [{"input": [10**20], "output": []}]])
def test_invalid_examples_are_rejected(data):
    with pytest.raises(ValueError):
        execute("", data)


def test_bad_syntax_and_recursive_macro_are_rejected():
    with pytest.raises(ValueError, match="Unknown word"):
        compile_program("import os")
    MACROS["recursive-test"] = {"body": "recursive-test", "group": "test"}
    try:
        with pytest.raises(ValueError, match="Recursive macro"):
            compile_program("recursive-test")
    finally:
        del MACROS["recursive-test"]


def test_large_exact_reference_outputs_and_floor_division():
    huge = 2**150
    assert stack_from(f"[ {huge} ]", expected=True) == (Int(huge),)
    assert execute("div", [{"input": [-7, 2], "output": [-4]}])["matches"] == 1


def test_long_quotations_are_display_elided_without_changing_execution():
    code = "[ " + " ".join(["dup drop"] * 150) + " ] i"
    result = execute(code, [{"input": [7], "output": [7]}])
    assert result["matches"] == 1
    wide = result["examples"][0]["frames"][200]
    assert wide["quoting"] and any(v["type"] == "elision" for v in wide["stack"])
    compact = execute(code, [{"input": [7], "output": [7]}], capture=False)
    assert compact["matches"] == 1 and len(compact["examples"][0]["frames"]) == 2


def test_policy_adapter_uses_real_checkpoint_loader_and_legal_mask(tmp_path):
    pytest.importorskip("torch")
    import sys
    if sys.version_info < (3, 12):
        pytest.skip("Repository policy imports require Python 3.12")
    from wandering_light.rollout_policy import PolicyArchitecture
    from wandering_light.stack_policy import StackPairPolicy, save_policy
    from wandering_light.stack_workshop.model import Adviser
    import torch
    torch.manual_seed(19)
    checkpoint = tmp_path / "test-only-random-weights"
    save_policy(checkpoint, StackPairPolicy(PolicyArchitecture(8, 12, 16)),
                trainer_state={}, manifest={"purpose": "unit-test-only"}, wandb_url=None)
    adviser = Adviser()
    assert adviser.load(checkpoint)["ready"]
    suggestions = adviser.rank("", [{"input": [3], "output": [6]}])["suggestions"]
    assert len(suggestions) == 6
    assert all(s["op"] not in {"add", "sub", "mul", "div", "]"} for s in suggestions)
    assert all(0 <= s["probability"] <= 1 for s in suggestions)
    assert adviser.rank("dup", [{"input": [3], "output": [6]}])["ready"]
    # No authored solution or hidden examples enter the request.
    with pytest.raises(ValueError):
        adviser.rank("add", [{"input": [3], "output": [6]}])


def test_http_api_and_cross_origin_rejection():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        body = json.dumps({"lesson": "echo", "program": "dup add"}).encode()
        request = urllib.request.Request(url + "/api/evaluate", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["matches"] == 3
        bad = urllib.request.Request(url + "/api/evaluate", data=body, headers={"Origin": "https://example.com"})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(bad)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
