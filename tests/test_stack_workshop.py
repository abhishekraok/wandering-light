"""Signal House: the mark-28 workshop service (engine, lessons, adviser, server)."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from wandering_light import stack_machine
from wandering_light.stack_machine import MARK_OPS, Bool, Int, Lst, Mark, Sym
from wandering_light.stack_workshop import engine, server
from wandering_light.stack_workshop.curriculum import BY_ID, LESSONS, REFERENCES
from wandering_light.stack_workshop.engine import (
    FUEL,
    MACROS,
    OPS,
    compile_program,
    execute,
    inspect_op,
    snapshot,
    stack_from,
)
from wandering_light.stack_workshop.model import Adviser
from wandering_light.stack_workshop.server import Handler, api


def ex(given, wanted):
    return {"input": given, "output": wanted}


def final_text(result, row=0):
    stack = result["examples"][row]["frames"][-1]["stack"]
    return "[ " + " ".join(d["text"] for d in stack) + " ]" if stack else "[]"


# ---------------------------------------------------------------------------
# Ops and macros
# ---------------------------------------------------------------------------


def test_ops_are_the_canonical_mark_28_ops():
    assert list(OPS) == [op.name for op in MARK_OPS]
    assert all(OPS[op.name] is op for op in MARK_OPS)
    assert len(OPS) == 28
    assert FUEL == 100_000


def test_macros_are_read_from_the_library_with_groups():
    assert MACROS["inc"] == {"body": "1 add", "group": "scalars"}
    assert MACROS["2"]["group"] == "constants"
    assert MACROS["rot"]["group"] == "stack shuffles"
    # a group title with a colon is named by what precedes it
    assert MACROS["range"]["group"] == "ranges"
    assert MACROS["bxor"]["group"] == "bits"
    # comments are not bodies, and no macro shadows an op
    assert "#" not in MACROS["gcd"]["body"]
    assert not set(MACROS) & set(OPS)


def test_every_library_macro_compiles():
    for name in MACROS:
        assert compile_program(name), name


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------


def test_compile_spans_written_primitives_and_brackets():
    text = "1 [dup] add # one more\n"
    tokens = compile_program(text)
    assert [t["op"] for t in tokens] == ["1", "[", "dup", "]", "add"]
    for token in tokens:
        assert text[token["start"] : token["end"]] == token["op"]
        assert token["macro"] is None


def test_compile_expands_macros_with_outermost_name_and_word_span():
    text = "dup  sq\nmax2"
    tokens = compile_program(text)
    sq = [t for t in tokens if t["macro"] == "sq"]
    assert [t["op"] for t in sq] == ["dup", "mul"]
    assert all(text[t["start"] : t["end"]] == "sq" for t in sq)
    # max2 uses only primitives, but `le?` -> `gt?` -> ... nests: outermost wins
    nested = compile_program("le?")
    assert {t["macro"] for t in nested} == {"le?"}
    assert {(t["start"], t["end"]) for t in nested} == {(0, 3)}
    assert all(t["op"] in OPS for t in nested)
    assert [t["op"] for t in tokens if t["macro"] == "max2"] == engine.expansion("max2")


def test_compile_comment_only_and_empty_programs():
    assert compile_program("") == []
    assert compile_program("# nothing here\n   ") == []


def test_unknown_word_is_named():
    with pytest.raises(ValueError, match=r"^Unknown word 'frobnicate'\. "):
        compile_program("1 frobnicate")
    # symbol ops are full-52, not mark-28
    with pytest.raises(ValueError, match=r"^Unknown word ''add'\. "):
        compile_program("'add")


def test_unknown_word_inside_a_macro(monkeypatch):
    monkeypatch.setitem(MACROS, "broken", {"body": "1 nosuch", "group": None})
    with pytest.raises(ValueError, match=r"^Unknown word 'nosuch'\. "):
        compile_program("broken")


def test_recursive_macros_are_refused(monkeypatch):
    monkeypatch.setitem(MACROS, "loop", {"body": "1 loop", "group": None})
    with pytest.raises(ValueError, match=r"^Recursive macro: loop$"):
        compile_program("loop")
    monkeypatch.setitem(MACROS, "ping", {"body": "pong", "group": None})
    monkeypatch.setitem(MACROS, "pong", {"body": "dup ping", "group": None})
    with pytest.raises(ValueError, match=r"^Recursive macro: ping$"):
        compile_program("ping")


# ---------------------------------------------------------------------------
# Stacks in
# ---------------------------------------------------------------------------


def test_stack_from_json_and_text_agree():
    expected = (Int(1), Bool(True), Lst((Int(2), Lst(()))))
    assert stack_from([1, True, [2, []]]) == expected
    assert stack_from("[ 1 true (2 nil) ]") == expected
    assert stack_from("1 true (2 ())") == expected
    assert stack_from([]) == ()
    assert stack_from("[]") == ()
    # a JSON bool is a Bool, never Int(1)
    assert stack_from([True]) == (Bool(True),)


@pytest.mark.parametrize(
    "bad",
    [
        [1.5],
        ["1"],
        [None],
        [{"a": 1}],
        [[1, "x"]],
        "[ 1 dup ]",
        "[ (1 add) ]",
        "[ MARK ]",
        "[ (1 2 ]",
        [10**12 + 1],
        [-(10**12) - 1],
        "[ 1000000000001 ]",
        list(range(17)),
        [list(range(65))],
        3,
        None,
        {"input": [1]},
    ],
)
def test_stack_from_rejects(bad):
    with pytest.raises(ValueError):
        stack_from(bad)


def test_stack_from_limits_are_inclusive():
    assert stack_from([10**12, -(10**12)]) == (Int(10**12), Int(-(10**12)))
    assert len(stack_from(list(range(16)))) == 16
    assert len(stack_from([list(range(64))])[0].items) == 64
    nested12 = 0
    for _ in range(12):
        nested12 = [nested12]
    stack_from([nested12])
    with pytest.raises(ValueError):
        stack_from([[nested12]])


def test_expected_outputs_allow_512_digits():
    big = 10**511  # 512 digits
    assert stack_from([big], expected=True) == (Int(big),)
    assert stack_from(str(big), expected=True) == (Int(big),)
    with pytest.raises(ValueError):
        stack_from([big])
    with pytest.raises(ValueError):
        stack_from([10**512], expected=True)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def test_snapshot_types():
    stack = (Int(-3), Bool(False), Sym("dup"), Mark(), Lst((Int(1), Sym("add"))))
    shot = snapshot(stack)
    assert [d["type"] for d in shot] == ["int", "bool", "symbol", "mark", "list"]
    assert [d["text"] for d in shot] == ["-3", "false", "dup", "MARK", "(1 add)"]
    assert shot[4]["items"] == [
        {"type": "int", "text": "1"},
        {"type": "symbol", "text": "add"},
    ]
    assert snapshot((Lst(()),)) == [{"type": "list", "text": "nil", "items": []}]


def test_snapshot_elides_deep_stacks():
    assert len(snapshot(tuple(Int(i) for i in range(48)))) == 48
    shot = snapshot(tuple(Int(i) for i in range(60)))
    assert len(shot) == 49
    assert shot[0] == {"type": "int", "text": "0"}
    assert shot[1]["type"] == "elision"
    assert [d["text"] for d in shot[2:]] == [str(i) for i in range(13, 60)]


@pytest.mark.parametrize("process_limit", [0, 4300])
def test_snapshot_survives_integers_too_long_to_print(process_limit):
    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(process_limit)
    try:
        shot = snapshot((Int(-(2**20000)),))
        assert shot[0] == {"type": "int", "text": "-<about 6021 digits>"}
        assert snapshot((Int(10**3999),))[0]["text"] == str(10**3999)
    finally:
        sys.set_int_max_str_digits(previous)


# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------


def test_execute_frames_and_totals():
    result = execute("dup add", [ex([3], [6]), ex("[ 4 ]", [9])])
    assert result["count"] == 2
    assert result["matches"] == 1
    assert result["primitive_count"] == 2
    assert [t["op"] for t in result["tokens"]] == ["dup", "add"]
    first, second = result["examples"]
    assert [f["op"] for f in first["frames"]] == [None, "dup", "add"]
    assert [f["fuel"] for f in first["frames"]] == [0, 1, 2]
    assert first["frames"][1]["stack"] == [
        {"type": "int", "text": "3"},
        {"type": "int", "text": "3"},
    ]
    assert first["matched"] and first["error"] is None
    assert first["fuel"] == 2 and first["max_depth"] == 2
    assert not second["matched"] and second["error"] is None
    assert second["output"] == [{"type": "int", "text": "9"}]
    assert result["fuel"] == 4
    assert result["max_depth"] == 2


def test_execute_marks_quoting_frames():
    result = execute("[ dup ] i", [ex([2], [2, 2])])
    frames = result["examples"][0]["frames"]
    assert [f["quoting"] for f in frames] == [False, True, True, False, False]
    assert frames[2]["stack"][-1] == {"type": "symbol", "text": "dup"}
    assert result["matches"] == 1


def test_refusal_halts_only_its_example():
    result = execute("add 1 add", [ex([1, 2], [4]), ex([5], [6])])
    good, bad = result["examples"]
    assert good["matched"] and len(good["frames"]) == 4
    assert bad["error"] == "add: needs 2 value(s), stack has 1"
    assert not bad["matched"]
    assert len(bad["frames"]) == 2
    assert bad["frames"][-1]["error"] == bad["error"]
    assert bad["frames"][-1]["op"] == "add"
    assert bad["frames"][-1]["stack"] == bad["frames"][0]["stack"]
    assert bad["fuel"] == 0


def test_refusal_inside_a_quotation_is_named_by_the_token():
    result = execute("[ add ] map", [ex([[1, 2]], [[1, 2]])])
    assert result["examples"][0]["error"].startswith("map: add: ")


def test_extra_values_do_not_match():
    assert execute("dup", [ex([1], [1])])["matches"] == 0


def test_capture_false_keeps_first_and_final_frames():
    full = execute("1 add 1 add", [ex([0], [2])])
    lean = execute("1 add 1 add", [ex([0], [2])], capture=False)
    frames = lean["examples"][0]["frames"]
    assert len(frames) == 2
    assert frames == [
        full["examples"][0]["frames"][0],
        full["examples"][0]["frames"][-1],
    ]
    assert lean["examples"][0]["max_depth"] == 2
    halted = execute("1 add add", [ex([0], [2])], capture=False)["examples"][0]
    assert len(halted["frames"]) == 2 and halted["frames"][-1]["error"]
    empty = execute("", [ex([0], [0])], capture=False)["examples"][0]
    assert len(empty["frames"]) == 1 and empty["matched"]


@pytest.mark.parametrize(
    "examples", [[], [ex([1], [1])] * 17, "nope", [{"input": [1]}]]
)
def test_execute_example_count_and_shape(examples):
    with pytest.raises(ValueError):
        execute("dup", examples)


def test_execute_observes_step_raw(monkeypatch):
    """The workshop steps through the canonical function, with the stated budget."""
    calls = []
    real = stack_machine.step_raw

    def spy(op, stack, budget, depth):
        calls.append((op.name, budget, depth))
        return real(op, stack, budget, depth)

    monkeypatch.setattr(engine, "step_raw", spy)
    execute("1 add dup", [ex([1], [2, 2])])
    assert calls == [("1", FUEL, 0), ("add", FUEL - 1, 0), ("dup", FUEL - 2, 0)]


def test_out_of_fuel_is_a_refusal():
    # a self-applying quotation loops until the budget runs out
    result = execute("[ dup i ] dup i", [ex([], [])], capture=False)
    row = result["examples"][0]
    assert row["error"].startswith("i: ")
    assert "fuel" in row["error"] or "deep" in row["error"]


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


MAP_EXAMPLES = [ex([[1, 2, 3]], [[2, 3, 4]])]


def test_inspect_records_nested_calls_in_return_order():
    out = inspect_op("[ 1 add ] map", MAP_EXAMPLES, 4)
    assert out["error"] is None and not out["truncated"]
    assert [f["op"] for f in out["frames"]] == ["add", "add", "add"]
    assert [f["before"][0]["text"] for f in out["frames"]] == ["1", "2", "3"]
    assert [f["after"][0]["text"] for f in out["frames"]] == ["2", "3", "4"]
    assert all(f["depth"] > 0 and f["error"] is None for f in out["frames"])


def test_inspect_nested_return_order_inner_first():
    out = inspect_op("[ [ dup ] i ] i", [ex([5], [5, 5])], 6)
    assert [f["op"] for f in out["frames"]] == ["dup", "i"]
    assert out["frames"][0]["depth"] > out["frames"][1]["depth"]


def test_inspect_is_not_hidden_by_a_warm_cache():
    execute("[ 1 add ] map", MAP_EXAMPLES)
    first = inspect_op("[ 1 add ] map", MAP_EXAMPLES, 4)
    second = inspect_op("[ 1 add ] map", MAP_EXAMPLES, 4)
    assert len(first["frames"]) == len(second["frames"]) == 3


def test_inspect_primitive_has_no_nested_calls():
    out = inspect_op("dup add", [ex([1], [2])], 1)
    assert out == {"frames": [], "truncated": False, "error": None}


def test_inspect_reports_refusals():
    out = inspect_op("[ add ] map", [ex([[1, 2]], [[1]])], 3)
    assert out["error"].startswith("map: add: ")
    assert out["frames"][0]["op"] == "add"
    assert out["frames"][0]["after"] is None
    assert out["frames"][0]["error"].startswith("needs 2")
    halted = inspect_op("add [ 1 ] i", [ex([1], [1])], 3)
    assert halted["frames"] == [] and "add" in halted["error"]


def test_inspect_truncates():
    program = "1000 range"
    step = len(compile_program(program)) - 1
    out = inspect_op(program, [ex([], [[0]])], step)
    assert out["truncated"]
    assert len(out["frames"]) == engine.INSPECT_LIMIT == 350


def test_inspect_bounds():
    with pytest.raises(ValueError):
        inspect_op("dup", [ex([1], [1])], 1)
    with pytest.raises(ValueError):
        inspect_op("dup", [ex([1], [1])], 0, example=1)
    with pytest.raises(ValueError):
        inspect_op("dup", [ex([1], [1])], None)


def test_inspect_concurrent_with_execute():
    errors = []

    def inspect_many():
        for _ in range(20):
            if len(inspect_op("[ 1 add ] map", MAP_EXAMPLES, 4)["frames"]) != 3:
                errors.append("hidden")

    def execute_many():
        for _ in range(20):
            execute("[ 1 add ] map", MAP_EXAMPLES)

    threads = [threading.Thread(target=f) for f in (inspect_many, execute_many) * 2]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

FIELDS = {
    "id",
    "chapter",
    "title",
    "concept",
    "story",
    "goal",
    "starter",
    "solution",
    "examples",
    "heldout",
    "hints",
    "discovery",
    "review",
    "prediction",
    "macros",
}


def test_lessons_shape():
    assert len(LESSONS) == 14
    assert len(BY_ID) == 14
    assert LESSONS[0]["id"] == "first-light"
    assert LESSONS[1]["id"] == "echo"
    for lesson in LESSONS:
        assert set(lesson) == FIELDS, lesson["id"]
        assert len(lesson["examples"]) == 3
        assert len(lesson["heldout"]) == 3
        assert lesson["hints"]
        assert all(name in MACROS for name in lesson["macros"])
    assert BY_ID["first-light"]["solution"] == "1 add"
    assert BY_ID["echo"]["solution"] == "dup add"


def test_curriculum_covers_the_concepts():
    used = {t["op"] for lesson in LESSONS for t in compile_program(lesson["solution"])}
    assert {
        "swap",
        "over",
        "[",
        "]",
        "i",
        "map",
        "filter",
        "fold",
        "dip",
        "ifte",
    } <= used
    assert any(
        t["macro"] for lesson in LESSONS for t in compile_program(lesson["solution"])
    )


@pytest.mark.parametrize("lesson", LESSONS, ids=[lesson["id"] for lesson in LESSONS])
def test_solution_passes_all_six(lesson):
    result = execute(lesson["solution"], lesson["examples"] + lesson["heldout"])
    assert result["matches"] == result["count"] == 6


@pytest.mark.parametrize("lesson", LESSONS, ids=[lesson["id"] for lesson in LESSONS])
def test_outputs_come_from_the_reference(lesson):
    reference = REFERENCES[lesson["id"]]
    for example in lesson["examples"] + lesson["heldout"]:
        assert example["output"] == reference(*example["input"])
        stack_from(example["input"])
        stack_from(example["output"], expected=True)


@pytest.mark.parametrize("lesson", LESSONS, ids=[lesson["id"] for lesson in LESSONS])
def test_prediction_matches_the_machine(lesson):
    prediction = lesson["prediction"]
    given = prediction["input"]
    result = execute(prediction["program"], [ex(given, given)])
    assert result["examples"][0]["error"] is None
    assert final_text(result) == prediction["answer"]


def test_lucky_fit_starter_fits_but_does_not_explain():
    lesson = BY_ID["lucky-fit"]
    assert execute(lesson["starter"], lesson["examples"])["matches"] == 3
    verdict = api("/api/verify", {"lesson": "lucky-fit", "program": lesson["starter"]})
    assert verdict["shown"] == 3 and not verdict["passed"]
    assert verdict["extra"] < verdict["extra_count"]


# ---------------------------------------------------------------------------
# Adviser
# ---------------------------------------------------------------------------


def test_adviser_without_a_model():
    adviser = Adviser()
    assert adviser.status()["ready"] is False
    out = adviser.rank("dup", [ex([1], [2])])
    assert out["suggestions"] == [] and out["ready"] is False
    with pytest.raises(ValueError, match="add"):
        adviser.rank("add", [ex([1], [2])])


def test_adviser_load_records_errors(tmp_path):
    adviser = Adviser()
    status = adviser.load(tmp_path / "missing")
    assert status["ready"] is False
    assert status["error"]


class FakeScorer:
    """Prefers ops later in the basis, so ranking is checkable."""

    def __init__(self):
        self.calls = []

    def log_probs(self, blocks, legal):
        import numpy as np

        self.calls.append((blocks, legal))
        logits = np.where(legal, np.arange(legal.shape[1], dtype=float), -np.inf)
        peak = logits.max(axis=1, keepdims=True)
        shifted = logits - peak
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def test_adviser_ranks_ops_legal_on_every_example():
    pytest.importorskip("torch")
    adviser = Adviser()
    adviser.scorer = FakeScorer()
    examples = [ex([1, 2], [3]), ex([4], [8])]
    out = adviser.rank("", examples)
    suggestions = out["suggestions"]
    assert 0 < len(suggestions) <= 6
    probabilities = [s["probability"] for s in suggestions]
    assert probabilities == sorted(probabilities, reverse=True)
    for suggestion in suggestions:
        assert len(suggestion["previews"]) == 2
        # legal on every example: e.g. `add` needs two values and is refused on [4]
        assert suggestion["op"] != "add"
    names = [s["op"] for s in suggestions]
    # the fake prefers later ops; `]` is last but refused (no open quotation)
    assert "]" not in names
    assert names[0] == "["
    blocks, legal = adviser.scorer.calls[0]
    assert blocks[0].shape[0] == 2
    assert legal.shape == (1, 28)


def test_adviser_loads_a_real_checkpoint(tmp_path):
    pytest.importorskip("torch")
    from wandering_light.rollout_policy import PolicyArchitecture
    from wandering_light.stack_policy import StackPairPolicy, save_policy

    checkpoint = save_policy(
        tmp_path / "tiny",
        StackPairPolicy(PolicyArchitecture(4, 4, 4)),
        trainer_state={},
        manifest={"arm": "test"},
        wandb_url=None,
    )
    adviser = Adviser()
    status = adviser.load(checkpoint)
    assert status["ready"] and status["error"] is None
    out = adviser.rank("[ 1", [ex([[1, 2]], [[2, 3]])])
    assert 0 < len(out["suggestions"]) <= 6
    total = sum(s["probability"] for s in out["suggestions"])
    assert 0 < total <= 1 + 1e-6


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_context_hides_answers():
    context = api("/api/context", {})
    assert len(context["lessons"]) == 14
    for lesson in context["lessons"]:
        assert not {"solution", "heldout", "hints"} & set(lesson)
        assert lesson["extra_tests"] == 3
        assert lesson["hint_count"] == len(BY_ID[lesson["id"]]["hints"])
    assert [op["name"] for op in context["ops"]] == list(OPS)
    assert len(context["macros"]) == len(MACROS)
    inc = next(m for m in context["macros"] if m["name"] == "inc")
    assert inc["expansion"] == "1 add" and inc["group"] == "scalars"
    assert "ready" in context["model"]
    assert "available" in context["world"]
    json.dumps(context)


def test_world_shelf_never_fails(tmp_path):
    missing = server.world_shelf.__wrapped__(tmp_path / "manifest.json")
    assert missing["available"] is False and missing["tasks"] == []
    assert missing["reason"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"corpus": "x", "splits": {"eval": {"path": "e.gz"}}})
    )
    absent = server.world_shelf.__wrapped__(manifest)
    assert absent["available"] is False and absent["corpus"] == "x"


def test_world_shelf_reads_a_local_payload(tmp_path):
    import gzip

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"corpus": "w", "splits": {"eval": {"path": "e.gz"}}})
    )
    rows = [
        {
            "behaviour": "b1",
            "summary": "s",
            "inputs": ["[ 1 ]", "[ dup ]"],
            "outputs": ["[ 2 ]", "[ 1 ]"],
        },
        {
            "behaviour": "b2",
            "summary": "s",
            "inputs": ["[ foo ]"],
            "outputs": ["[ 1 ]"],
        },
    ]
    with gzip.open(tmp_path / "e.gz", "wt") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    shelf = server.world_shelf.__wrapped__(manifest)
    assert shelf["available"]
    assert [t["behaviour"] for t in shelf["tasks"]] == ["b1"]
    assert shelf["tasks"][0]["examples"] == [ex("[ 1 ]", "[ 2 ]")]


def test_evaluate_uses_shown_examples_only():
    result = api("/api/evaluate", {"lesson": "echo", "program": "dup add"})
    assert result["count"] == 3 and result["matches"] == 3
    sandbox = api(
        "/api/evaluate",
        {"lesson": "sandbox", "program": "1 add", "examples": [ex([1], [2])]},
    )
    assert sandbox["count"] == 1 and sandbox["matches"] == 1


def test_verify_lesson_and_sandbox():
    passed = api("/api/verify", {"lesson": "first-light", "program": "1 add"})
    assert passed == {
        "passed": True,
        "shown": 3,
        "shown_count": 3,
        "extra": 3,
        "extra_count": 3,
        "failures": [],
        "evidence": "shown and extra tests",
    }
    failed = api("/api/verify", {"lesson": "first-light", "program": "dup add"})
    assert not failed["passed"]
    assert 0 < len(failed["failures"]) <= 3
    assert all(not row["matched"] for row in failed["failures"])
    sandbox = api(
        "/api/verify",
        {"lesson": "sandbox", "program": "dup", "examples": [ex([1], [1, 1])]},
    )
    assert sandbox["passed"] and sandbox["evidence"] == "examples"
    assert sandbox["extra_count"] == 0 and sandbox["extra"] == 0


def test_inspect_hint_reveal_suggest_model_routes():
    out = api(
        "/api/inspect", {"lesson": "brighten", "program": "[ 1 add ] map", "step": 4}
    )
    assert [f["op"] for f in out["frames"]] == ["add", "add", "add"]
    assert (
        api("/api/hint", {"lesson": "echo", "index": 1})["hint"]
        == BY_ID["echo"]["hints"][1]
    )
    assert api("/api/reveal", {"lesson": "echo"})["solution"] == "dup add"
    suggestion = api("/api/suggest", {"lesson": "echo", "program": "dup"})
    assert "suggestions" in suggestion and "ready" in suggestion
    assert "ready" in api("/api/model", {})


@pytest.mark.parametrize(
    ("route", "data"),
    [
        ("/api/evaluate", {"lesson": "nope", "program": "1"}),
        ("/api/evaluate", {"lesson": "echo", "program": "nosuchword"}),
        ("/api/evaluate", {"lesson": "sandbox", "program": "1"}),
        ("/api/hint", {"lesson": "sandbox"}),
        ("/api/hint", {"lesson": "echo", "index": 99}),
        ("/api/reveal", {"lesson": "sandbox"}),
        ("/api/inspect", {"lesson": "echo", "program": "dup", "step": 5}),
        ("/api/nowhere", {}),
    ],
)
def test_api_errors_are_value_errors(route, data):
    with pytest.raises(ValueError):
        api(route, data)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


@pytest.fixture
def live_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def post(base, route, body, origin=None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    request = urllib.request.Request(base + route, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if origin is not None:
        request.add_header("Origin", origin)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_round_trip(live_server):
    status, body = post(
        live_server, "/api/evaluate", {"lesson": "echo", "program": "dup add"}
    )
    assert status == 200 and body["matches"] == 3
    status, body = post(live_server, "/api/context", {}, origin="http://localhost:5173")
    assert status == 200 and len(body["lessons"]) == 14
    status, _ = post(live_server, "/api/model", {}, origin="http://127.0.0.1:8018")
    assert status == 200


@pytest.mark.parametrize(
    "origin", ["https://evil.example", "http://127.0.0.1.evil.example", "null"]
)
def test_http_foreign_origin_is_forbidden(live_server, origin):
    status, body = post(live_server, "/api/context", {}, origin=origin)
    assert status == 403 and "error" in body


def test_http_errors_are_400(live_server):
    status, body = post(live_server, "/api/evaluate", b"{not json")
    assert status == 400 and body["error"]
    status, body = post(
        live_server, "/api/evaluate", {"lesson": "echo", "program": "zap"}
    )
    assert status == 400 and body["error"].startswith("Unknown word 'zap'")
    status, body = post(live_server, "/api/evaluate", [1, 2])
    assert status == 400


def test_main_parses_port(monkeypatch):
    seen = {}

    class FakeServer:
        def __init__(self, address, handler):
            seen["address"] = address
            seen["handler"] = handler
            self.server_address = address

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            seen["closed"] = True

    monkeypatch.setattr(server, "ThreadingHTTPServer", FakeServer)
    assert server.main([]) == 0
    assert seen["address"] == ("127.0.0.1", 8018)
    assert seen["handler"] is Handler and seen["closed"]
    server.main(["--port", "9000"])
    assert seen["address"] == ("127.0.0.1", 9000)
