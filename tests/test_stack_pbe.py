"""The behaviour-first corpus claims three things. Check all three.

1. Every behaviour has two independent statements -- a Python reference and a
   stack-machine witness -- and they agree on data neither of them chose.
2. No task input contains a quotation. That is the defect the generator exists
   to remove, so it is checked structurally rather than trusted.
3. A candidate can be graded on examples it was never shown, which means the
   held-out set is really disjoint and really solved by the witness.
"""

import gzip
import json
import os
import random
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from wandering_light.stack_machine import (
    DEFAULT_FUEL_LIMIT,
    Lst,
    Sym,
    Value,
    ops_for_basis,
    parse_stack,
    step_raw,
    value_key,
)
from wandering_light.stack_pbe import (
    BASIS,
    BY_NAME,
    CATALOGUE,
    Task,
    _informative,
    assign_split,
    build_task,
    catalogue_digest,
    catalogue_source_digest,
    check,
    draw_task,
    encode_task,
    generate,
    independent_groups,
    load,
    run_program,
    skeleton,
    to_stack,
    to_value,
    verify,
    write_corpus,
)

# The corpus payload lives on the Hub, not in git -- see `corpus_hub`'s
# docstring for why. Tests read a small committed fixture instead, so the suite
# stays offline and fast; the manifest machinery is exercised separately
# against a corpus generated into tmp_path.
DATA = Path(__file__).resolve().parent.parent / "wandering_light" / "evals" / "data"
CORPUS = Path(__file__).resolve().parent / "test_data" / "stack_pbe_fixture.jsonl.gz"
MANIFEST_V4 = DATA / "stack_pbe_v4" / "manifest.json"
MANIFEST_V3 = DATA / "stack_pbe_v3" / "manifest.json"


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return load(str(CORPUS))


def contains_quotation(value: Value) -> bool:
    if isinstance(value, Sym):
        return True
    if isinstance(value, Lst):
        return any(contains_quotation(item) for item in value.items)
    return False


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


def test_behaviour_names_are_unique():
    names = [b.name for b in CATALOGUE]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_witness_agrees_with_reference_on_fresh_data(behaviour):
    """The whole guarantee: two independent statements, checked against each other.

    Neither is privileged. A wrong label would need the same mistake made twice,
    in two languages, on inputs the sampler chose after both were written.
    """
    assert verify(behaviour, trials=120, seed=1234) is None
    assert len(behaviour.witnesses) == len(set(behaviour.witnesses))


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_witness_leaves_exactly_one_value(behaviour):
    """A task's output is a value, not a stack with the answer somewhere in it."""
    rng = random.Random(7)
    for _ in range(20):
        outcome = run_program(behaviour.program, to_stack(behaviour.sample(rng)))
        assert not isinstance(outcome, str), outcome
        assert len(outcome[0]) == 1


@pytest.mark.parametrize("behaviour", CATALOGUE, ids=lambda b: b.name)
def test_sampler_never_produces_a_quotation(behaviour):
    """Inputs are numbers and lists of numbers. Nothing else, ever."""
    rng = random.Random(11)
    for _ in range(20):
        for value in to_stack(behaviour.sample(rng)):
            assert not contains_quotation(value)


def test_a_wrong_witness_is_caught():
    """The agreement check has to be able to fail, or it proves nothing."""
    from dataclasses import replace

    broken = replace(BY_NAME["sum"], witnesses=("1 [ add ] fold",))
    assert verify(broken, trials=50, seed=3) is not None


# --------------------------------------------------------------------------
# Task drawing
# --------------------------------------------------------------------------


def test_held_out_inputs_are_disjoint_from_shown():
    rng = random.Random(5)
    for behaviour in CATALOGUE:
        drawn = draw_task(behaviour, rng, examples=4, held_out=6)
        shown = {json.dumps(a) for a in drawn.shown}
        hidden = {json.dumps(a) for a in drawn.held_out}
        assert len(shown) == 4
        assert len(hidden) == 6
        assert not (shown & hidden)


def test_constant_example_sets_are_rejected():
    """Every output the same means a constant program fits."""
    args = [([1, 2],), ([3, 4],)]
    stacks = [to_stack(a) for a in args]
    assert not _informative(args, [7, 7], stacks)


def test_projection_example_sets_are_rejected():
    """Output equal to a fixed input position means a projection fits."""
    args = [([1, 2], 5), ([3, 4], 6)]
    stacks = [to_stack(a) for a in args]
    assert not _informative(args, [5, 6], stacks)
    assert _informative(args, [3, 7], stacks)


def test_build_task_rejects_an_output_its_witness_does_not_produce():
    """Labels are made by execution, so a wrong one cannot be written out."""
    from dataclasses import replace

    from wandering_light.stack_machine import MachineError

    rng = random.Random(2)
    drawn = draw_task(BY_NAME["sum"], rng, examples=3, held_out=2)
    corrupted = replace(drawn, shown_results=[r + 1 for r in drawn.shown_results])
    with pytest.raises(MachineError, match="disagree"):
        build_task(BY_NAME["sum"], corrupted)

    corrupted_hidden = replace(
        drawn, held_out_results=[r + 1 for r in drawn.held_out_results]
    )
    with pytest.raises(MachineError, match="disagree"):
        build_task(BY_NAME["sum"], corrupted_hidden)


# --------------------------------------------------------------------------
# The split
# --------------------------------------------------------------------------


def test_split_is_on_the_skeleton_not_the_behaviour_or_the_task():
    """Behaviours sharing a program shape must land on the same side."""
    by_skeleton: dict[str, set[str]] = {}
    for behaviour in CATALOGUE:
        assert assign_split(behaviour.skeleton) in {"train", "eval"}
        by_skeleton.setdefault(behaviour.skeleton, set()).add(
            assign_split(behaviour.skeleton)
        )
    assert all(len(sides) == 1 for sides in by_skeleton.values())
    assert set().union(*by_skeleton.values()) == {"train", "eval"}


def test_the_current_catalogue_has_no_large_inseparable_group():
    """Every behaviour stands alone, so any split is possible."""
    groups = independent_groups(CATALOGUE)
    assert len(groups) == len(CATALOGUE)
    assert max(len(names) for names in groups.values()) == 1


def test_composed_behaviours_are_grouped_and_cannot_be_split_apart():
    """The audit that a cross-product catalogue must fail.

    Twelve predicates crossed with twelve maps give 144 distinct skeletons from
    24 ideas. Each passes `skeleton`, so the count says the catalogue is large;
    the groups say it is one blob that no split can hold out.
    """
    from dataclasses import replace

    seed = BY_NAME["sum"]
    grid = [
        replace(seed, name=f"keep_{p}_then_{m}")
        for p in ("even", "odd", "positive")
        for m in ("double", "square", "negate")
    ]
    groups = independent_groups(grid)
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == len(grid)


def test_skeleton_blanks_literals_and_interchangeable_ops():
    """`keep_zeros` and `keep_negative` are two functions of one shape."""
    assert BY_NAME["keep_zeros"].skeleton == BY_NAME["keep_negative"].skeleton
    assert skeleton(("[", "0", "eq?", "]", "filter")) == "[ _ _ ] filter"
    assert BY_NAME["sum"].skeleton != BY_NAME["reverse"].skeleton
    assert assign_split(BY_NAME["keep_zeros"].skeleton) == assign_split(
        BY_NAME["keep_negative"].skeleton
    )


def test_every_witness_of_a_behaviour_is_checked():
    """An alternative that is subtly wrong must not ride along on the first."""
    from dataclasses import replace

    behaviour = BY_NAME["sum"]
    assert len(behaviour.witnesses) > 1
    broken = replace(behaviour, witnesses=(behaviour.witness, "1 [ add ] fold"))
    assert verify(broken, trials=80, seed=1) is not None


def test_a_bare_string_of_witnesses_is_refused():
    """`witnesses="add"` would iterate characters. Fail loudly instead."""
    from dataclasses import replace

    from wandering_light.stack_machine import MachineError

    with pytest.raises(MachineError, match="tuple of programs"):
        replace(BY_NAME["sum"], witnesses="0 [ add ] fold")
    with pytest.raises(MachineError, match="at least one"):
        replace(BY_NAME["sum"], witnesses=())


@pytest.mark.parametrize("seed", ["0", "1", "31337"])
def test_split_does_not_depend_on_pythonhashseed(seed):
    """SHA-256, not `hash`, so regenerating cannot move a behaviour across."""
    script = """
import json
from wandering_light.stack_pbe import CATALOGUE, assign_split
print(json.dumps({b.name: assign_split(b.skeleton) for b in CATALOGUE}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )
    expected = {b.name: assign_split(b.skeleton) for b in CATALOGUE}
    assert json.loads(result.stdout) == expected


# --------------------------------------------------------------------------
# The committed corpus
# --------------------------------------------------------------------------


def test_fixture_covers_every_behaviour(tasks):
    assert {task.behaviour for task in tasks} == {b.name for b in CATALOGUE}
    assert {task.basis for task in tasks} == {BASIS}
    assert {task.witness_source for task in tasks} == {"authored"}


def test_no_task_input_contains_a_quotation(tasks):
    """The defect the generator exists to remove, checked on the artefact."""
    for task in tasks:
        for text in (*task.inputs, *task.held_out_inputs):
            for value in parse_stack(text, basis=BASIS):
                assert not contains_quotation(value), text


def test_every_witness_reproduces_its_shown_outputs(tasks):
    for task in tasks:
        for given, wanted in zip(task.inputs, task.outputs, strict=True):
            outcome = run_program(task.program, parse_stack(given, basis=BASIS))
            assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
            stack, _fuel = outcome
            want = parse_stack(wanted, basis=BASIS)
            assert tuple(map(value_key, stack)) == tuple(map(value_key, want))


def test_every_witness_reproduces_its_held_out_outputs(tasks):
    """Grading on hidden data only means something if the label is right."""
    for task in tasks:
        pairs = zip(task.held_out_inputs, task.held_out_outputs, strict=True)
        for given, wanted in pairs:
            outcome = run_program(task.program, parse_stack(given, basis=BASIS))
            assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
            stack, _fuel = outcome
            want = parse_stack(wanted, basis=BASIS)
            assert tuple(map(value_key, stack)) == tuple(map(value_key, want))


def test_held_out_examples_are_disjoint_from_shown_in_the_corpus(tasks):
    for task in tasks:
        assert not (set(task.inputs) & set(task.held_out_inputs))
        assert task.held_out >= 1


def test_corpus_data_is_varied(tasks):
    """The defect being fixed was 17 input sets across 1,857 tasks."""
    inputs = {i for task in tasks for i in (*task.inputs, *task.held_out_inputs)}
    assert len(inputs) > 10 * len(tasks)
    assert len({tuple(task.inputs) for task in tasks}) == len(tasks)


def test_no_behaviour_or_skeleton_spans_both_splits(tasks):
    for key in ("behaviour", "skeleton"):
        grouped: dict[str, set[str]] = {}
        for task in tasks:
            grouped.setdefault(getattr(task, key), set()).add(task.split)
        assert all(len(splits) == 1 for splits in grouped.values()), key


def test_shown_examples_never_hold_list_length_fixed(tasks):
    """The hole that produced v3's one repeated memorisation, closed."""
    for task in tasks:
        shapes = {
            tuple(
                len(v.items)
                for v in parse_stack(text, basis=BASIS)
                if isinstance(v, Lst)
            )
            for text in task.inputs
        }
        if next(iter(shapes)):
            assert len(shapes) > 1, task.behaviour


def test_alternative_programs_also_solve_their_task(tasks):
    """They are labels too, so they get executed like the canonical one."""
    checked = 0
    for task in tasks:
        for program in task.alternative_programs:
            for given, wanted in zip(task.inputs, task.outputs, strict=True):
                outcome = run_program(program, parse_stack(given, basis=BASIS))
                assert not isinstance(outcome, str), f"{task.behaviour}: {outcome}"
                want = parse_stack(wanted, basis=BASIS)
                assert tuple(map(value_key, outcome[0])) == tuple(map(value_key, want))
            checked += 1
    assert checked > 0


def test_manifests_describe_a_corpus_that_is_not_in_git():
    """Payloads live on the Hub; the manifest is what the repository keeps."""
    for path, version, policy in (
        (MANIFEST_V4, 2, "skeleton"),
        (MANIFEST_V3, 1, "behaviour"),
    ):
        manifest = json.loads(path.read_text())
        assert manifest["schema_version"] == version
        assert manifest["split_policy"] == policy
        assert manifest["basis"] == BASIS
        assert manifest["counts"]["tasks"] > 0
        assert manifest["splits"], path
        for split in manifest["splits"].values():
            assert split["sha256"].startswith("sha256:")
            assert split["records"] > 0


def test_written_corpus_verifies_against_its_manifest(tmp_path):
    """The digest check `fetch_corpus` relies on, exercised end to end."""
    from wandering_light.corpus_hub import verify_corpus_files

    made, _stats = generate(per_behaviour=1, seed=5)
    write_corpus(made, tmp_path, config={"per_behaviour": 1, "seed": 5})
    checked = verify_corpus_files(tmp_path)
    assert set(checked) == {task.split for task in made}
    assert load(tmp_path) and len(load(tmp_path)) == len(made)


def test_a_tampered_payload_is_refused(tmp_path):
    """Without this check the Hub payload would be trusted, not verified."""
    from wandering_light.corpus_hub import verify_corpus_files

    made, _stats = generate(per_behaviour=1, seed=6)
    write_corpus(made, tmp_path, config={})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    victim = next(iter(manifest["splits"].values()))["path"]
    (tmp_path / victim).write_bytes(b"not the corpus")
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_corpus_files(tmp_path)


def test_catalogue_digest_tracks_the_witnesses():
    """A changed witness changes the labels, so it must invalidate a manifest."""
    from dataclasses import replace

    before = catalogue_digest()
    changed = [*CATALOGUE[:-1], replace(CATALOGUE[-1], witnesses=("drop",))]
    assert catalogue_digest(changed) != before
    assert catalogue_digest() == before


def test_load_refuses_a_payload_that_is_not_the_one_the_manifest_describes(tmp_path):
    """`load` read whatever the manifest named; a regenerated payload loaded
    exactly like the published one (#142)."""
    made, _stats = generate(per_behaviour=1, seed=6)
    write_corpus(made, tmp_path, config={})
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    victim = next(iter(manifest["splits"].values()))["path"]
    (tmp_path / victim).write_bytes(gzip.compress(b""))
    with pytest.raises(ValueError, match="digest mismatch"):
        load(tmp_path)
    assert 0 < len(load(tmp_path, verify="off")) < len(made)


def test_load_warns_when_a_manifest_names_another_catalogue(tmp_path):
    """A warning, not an error: the catalogue grows past a published corpus."""
    made, _stats = generate(per_behaviour=1, seed=6)
    write_corpus(made, tmp_path, config={})
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["catalogue"]["digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match="catalogue digest"):
        assert len(load(tmp_path)) == len(made)
    # Another generator's catalogue is not this module's, so it is not compared
    # (the world corpora).
    manifest["generator"] = "experiments/world_behaviours/build_corpus.py"
    path.write_text(json.dumps(manifest))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(load(tmp_path)) == len(made)


def test_the_committed_v4_manifest_names_the_live_catalogue():
    """What `load` now checks on every read of the published corpus."""
    manifest = json.loads(MANIFEST_V4.read_text())
    assert manifest["catalogue"]["digest"] == catalogue_digest()


def test_the_source_digest_sees_what_the_witness_digest_cannot():
    """A sampler edit changes every task of a behaviour and no witness (#141)."""
    from dataclasses import replace

    from wandering_light.stack_pbe import POSITIVE, SMALL, _list_of

    before = catalogue_source_digest()
    assert before == catalogue_source_digest()
    variants = {
        "pool_small": replace(CATALOGUE[0], sample=_list_of(SMALL)),
        "pool_positive": replace(CATALOGUE[0], sample=_list_of(POSITIVE)),
        "reference": replace(CATALOGUE[0], reference=lambda *args: 0),
        "builtin_max": replace(CATALOGUE[0], reference=max),
        "builtin_min": replace(CATALOGUE[0], reference=min),
        "guard": replace(CATALOGUE[0], accept=lambda args, results: True),
    }
    digests = {
        name: catalogue_source_digest([b, *CATALOGUE[1:]])
        for name, b in variants.items()
    }
    # One source line, two closures: the pool must tell them apart.
    assert digests["pool_small"] != digests["pool_positive"]
    assert digests["builtin_max"] != digests["builtin_min"]
    assert len(set(digests.values())) == len(digests)
    for name in ("reference", "guard", "builtin_max", "pool_positive"):
        assert digests[name] != before, name
    # None of that moves the witness digest, which is the point of a second one.
    for b in variants.values():
        assert catalogue_digest([b, *CATALOGUE[1:]]) == catalogue_digest()


def test_written_manifest_records_the_source_digest_and_load_checks_it(tmp_path):
    made, _stats = generate(per_behaviour=1, seed=5)
    write_corpus(made, tmp_path, config={})
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["catalogue"]["source_digest"] == catalogue_source_digest()
    manifest["catalogue"]["source_digest"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.warns(UserWarning, match="source_digest"):
        load(tmp_path)
    # A manifest written before #142 has no source digest and is not refused
    # for it; the committed v4 manifest is one.
    del manifest["catalogue"]["source_digest"]
    path.write_text(json.dumps(manifest))
    assert len(load(tmp_path)) == len(made)


def test_a_corpus_written_from_another_catalogue_round_trips_with_it(tmp_path):
    """`write_corpus` takes a catalogue, so `load` must take the same one
    (Codex review on #143)."""
    only = [BY_NAME["keep_below"]]
    made = [
        t for t in generate(per_behaviour=1, seed=5)[0] if t.behaviour == "keep_below"
    ]
    write_corpus(made, tmp_path, config={}, catalogue=only)
    with pytest.warns(UserWarning, match="manifest catalogue") as caught:
        load(tmp_path)
    messages = sorted(str(w.message) for w in caught)
    assert len(messages) == 2
    assert "catalogue digest is" in messages[0]
    assert "catalogue source_digest is" in messages[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load(tmp_path, catalogue=only) == made
    assert load(tmp_path, verify="off") == made


def test_load_hub_mode_checks_the_pinned_revision_and_local_mode_never_does(
    tmp_path, monkeypatch
):
    from wandering_light import corpus_hub

    made, _stats = generate(per_behaviour=1, seed=5)
    write_corpus(made, tmp_path, config={})
    with pytest.raises(ValueError, match="no hub revision"):
        load(tmp_path, verify="hub")
    with pytest.raises(ValueError, match="verify must be"):
        load(tmp_path, verify="always")

    corpus_hub.record_hub_location(tmp_path, repo_id="acct/corpus", revision="a" * 40)
    calls = []

    def fake_verify(corpus_dir, *, repo_id, revision, **kwargs):
        calls.append((Path(corpus_dir), repo_id, revision))
        return {}

    monkeypatch.setattr(corpus_hub, "verify_published_corpus", fake_verify)
    assert len(load(tmp_path, verify="hub")) == len(made)
    assert calls == [(tmp_path, "acct/corpus", "a" * 40)]
    calls.clear()
    assert len(load(tmp_path)) == len(made)
    assert calls == []


@pytest.mark.xfail(
    strict=True,
    reason="aligned-matches-sampler (#141): _partly_matching_lists pins index 0 to "
    "match and index 1 to mismatch in every draw. The fix is a corpus version, "
    "not an edit: every published v4 number was measured on the current payload.",
)
def test_aligned_matches_examples_are_not_all_aligned_the_same_way():
    rng = random.Random(0)
    draws = [BY_NAME["aligned_matches"].sample(rng) for _ in range(200)]
    assert any(left[0] != right[0] for left, right in draws)
    assert any(left[1] == right[1] for left, right in draws)


def test_corpus_round_trips_through_load(tmp_path):
    made, _stats = generate(per_behaviour=1, seed=3)
    path = tmp_path / "corpus.jsonl.gz"
    from dataclasses import asdict

    with gzip.open(path, "wt") as handle:
        for task in made:
            handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
    assert load(str(path)) == made


def test_encode_task_shows_examples_and_hides_the_held_out_set(tasks):
    encoded = encode_task(tasks[0])
    assert encoded["prompt"].endswith("=>")
    assert encoded["completion"].strip() == " ".join(tasks[0].program)
    for hidden in tasks[0].held_out_inputs:
        assert hidden not in encoded["prompt"]


# --------------------------------------------------------------------------
# Why the basis is `mark-28` and not `executable-26`
# --------------------------------------------------------------------------


def test_executable_26_reaches_no_executable_quotation_from_data():
    """The reason this corpus cannot run on the default basis.

    No op in `executable-26` puts an op name on the stack, so from a data-only
    start every reachable quotation is a list of data -- which `map` can only
    apply as the identity, and which `filter` and `fold` reject. `mark-28`
    reaches executable quotations from the same start.
    """
    start = to_stack(([1, 2, 3],))

    def reachable_quotations(basis: str, depth: int) -> set:
        ops = ops_for_basis(basis)
        seen = {tuple(map(value_key, start))}
        frontier = [start]
        found = set()
        for _ in range(depth):
            nxt = []
            for stack in frontier:
                for op in ops:
                    outcome = step_raw(op, stack, DEFAULT_FUEL_LIMIT, 0)
                    if isinstance(outcome, str):
                        continue
                    moved = outcome[0]
                    for value in moved:
                        if contains_quotation(value):
                            found.add(value_key(value))
                    key = tuple(map(value_key, moved))
                    if key not in seen:
                        seen.add(key)
                        nxt.append(moved)
            frontier = nxt
        return found

    assert reachable_quotations("executable-26", 4) == set()
    assert reachable_quotations("mark-28", 4)


def test_check_reports_the_input_it_disagreed_on():
    from dataclasses import replace

    broken = replace(BY_NAME["reverse"], witnesses=("nil [ swap cons ] fold",))
    bad = check(broken, ([1, 2, 3],))
    assert bad is not None
    assert bad.args == ([1, 2, 3],)


def test_to_value_refuses_bools():
    from wandering_light.stack_machine import MachineError

    with pytest.raises(MachineError):
        to_value(True)
