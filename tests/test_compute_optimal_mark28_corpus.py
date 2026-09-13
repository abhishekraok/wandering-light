"""Corpus mixing in the #101 driver: scoping, the drop list, the pinned holdout.

world-plus-v4-policy (#134) trains the #101 recipe on two corpora at once. Each
test here pins a way that run could silently report a wrong number rather than
fail: a cohort drawn from the wrong pool, a behaviour that should have been
dropped still in the fit rows, or a validation set that is not the same
quantity in both arms. No Hub access — the corpora are synthetic Tasks.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from experiments import compute_optimal_mark28 as driver
from wandering_light.stack_machine import Config, format_stack
from wandering_light.stack_pbe import Task, run_program, skeleton, to_stack
from wandering_light.stack_policy import validation_skeleton


def _task(behaviour: str, split: str, program: tuple[str, ...]) -> Task:
    """A minimal Task whose witness really runs, since witness_rows replays it."""
    outcome = run_program(program, to_stack((1,)))
    assert not isinstance(outcome, str), outcome
    result = (format_stack(Config(outcome[0])),)
    return Task(
        behaviour=behaviour,
        summary=behaviour,
        split=split,
        task_hash_split=split,
        inputs=("[ 1 ]",),
        outputs=result,
        held_out_inputs=("[ 1 ]",),
        held_out_outputs=result,
        program=program,
        length=len(program),
        examples=1,
        held_out=1,
        fuel=len(program),
        tags=(),
        skeleton=skeleton(program),
    )


def _program(index: int) -> tuple[str, ...]:
    """`dup`^k `add`^k: executable from a one-value stack, one skeleton each.

    The 1-in-10 validation holdout hashes the skeleton, so a distinct skeleton
    per behaviour is what makes the holdout non-empty here as it is in the real
    corpora (alpha holds out 11, 19, 31, 32, 35; beta holds out 11 and 19).
    """
    k = index + 1
    return ("dup",) * k + ("add",) * k


def _corpus(name: str, count: int, split: str) -> list[Task]:
    return [_task(f"{name}-{i}", split, _program(i)) for i in range(count)]


@pytest.fixture
def two_corpora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Two manifests the driver can load, with disjoint behaviour names."""
    alpha = _corpus("alpha", 40, "train") + _corpus("alpha-e", 30, "eval")
    beta = _corpus("beta", 25, "train") + _corpus("beta-e", 20, "eval")
    tasks = {"alpha": alpha, "beta": beta}
    paths = {}
    for name in tasks:
        path = tmp_path / name / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"corpus": name, "hub": None, "splits": {}}))
        paths[name] = path

    def fake_fetch(manifest: Path):
        return manifest.parent

    def fake_load(directory: Path):
        return list(tasks[directory.name])

    monkeypatch.setattr(driver, "fetch_corpus", fake_fetch)
    monkeypatch.setattr(driver, "load", fake_load)
    monkeypatch.setattr(driver, "EVAL_CORPUS", None)
    monkeypatch.setattr(driver, "VALIDATION_CORPUS", None)
    monkeypatch.setattr(driver, "DROP_BEHAVIOURS", frozenset())
    monkeypatch.setattr(driver, "TRAIN_TASK_CAP", 0)
    # corpus() mutates CORPUS_OF in place; a fresh dict keeps the synthetic
    # behaviour names from leaking into later tests (Codex review on #135).
    monkeypatch.setattr(driver, "CORPUS_OF", {})
    return paths, tasks


def test_cohort_is_drawn_from_the_eval_corpus_alone(two_corpora, monkeypatch) -> None:
    """Mixing corpora must not change an existing cohort.

    The #101 cohort is a seed-17 shuffle of one corpus's eval split. Drawing it
    from the concatenated pool instead reseeds the shuffle over a larger set and
    silently yields different tasks, so every solve rate stops being comparable
    with the recorded campaign.
    """
    paths, _ = two_corpora
    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"]])
    alone = driver.cohort(driver.corpus())

    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"], paths["beta"]])
    mixed_tasks = driver.corpus()
    mixed = driver.cohort(mixed_tasks)  # primary corpus by default
    assert [i for i, _ in mixed] == [i for i, _ in alone]
    assert [t.behaviour for _, t in mixed] == [t.behaviour for _, t in alone]

    scoped = driver.cohort(mixed_tasks, eval_corpus="alpha")
    assert [i for i, _ in scoped] == [i for i, _ in alone]

    other = driver.cohort(mixed_tasks, eval_corpus="beta")
    assert {t.behaviour for _, t in other}.isdisjoint(t.behaviour for _, t in alone)
    assert all(driver.CORPUS_OF[t.behaviour] == "beta" for _, t in other)


def test_dropped_behaviours_leave_no_row(two_corpora, monkeypatch) -> None:
    """The eval-overlap filter has to remove rows, not just tasks.

    Rows carry a task index into the list they were built from, so the check is
    on counts: every surviving train task contributes exactly its witness
    prefixes, and the total falls by exactly the dropped tasks' lengths.
    """
    paths, _ = two_corpora
    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"], paths["beta"]])
    loaded = driver.corpus()
    before_fit, before_val = driver.training_rows(loaded)

    dropped = {"beta-3", "beta-11", "alpha-19"}  # beta-11 and alpha-19 are held out
    monkeypatch.setattr(driver, "DROP_BEHAVIOURS", frozenset(dropped))
    after_fit, after_val = driver.training_rows(loaded)

    train = [t for t in loaded if t.split == "train"]
    gone = sum(t.length for t in train if t.behaviour in dropped)
    assert gone > 0
    assert len(after_fit) + len(after_val) == len(before_fit) + len(before_val) - gone
    assert len(after_fit) + len(after_val) == sum(
        t.length for t in train if t.behaviour not in dropped
    )
    # the drop reaches both halves, not just the fit rows
    assert len(after_fit) < len(before_fit)
    assert len(after_val) < len(before_val)


def test_validation_corpus_pins_the_holdout(two_corpora, monkeypatch) -> None:
    """Both arms must select the best epoch on the same rows.

    Without the pin, adding a corpus also changes the validation set, so the two
    arms' best_validation_cross_entropy are different quantities and the
    headline comparison is meaningless while looking plausible.
    """
    paths, _ = two_corpora
    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"]])
    alpha_tasks = driver.corpus()
    alpha_fit, alpha_only = driver.training_rows(alpha_tasks)

    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"], paths["beta"]])
    mixed = driver.corpus()
    naive_fit, naive = driver.training_rows(mixed)
    monkeypatch.setattr(driver, "VALIDATION_CORPUS", "alpha")
    pinned_fit, pinned = driver.training_rows(mixed)

    assert len(naive) > len(alpha_only)  # what the unpinned run would have scored
    assert len(pinned) == len(alpha_only)  # the same quantity as the baseline arm
    assert len(pinned_fit) == len(naive_fit)  # the pin does not touch the fit rows
    assert len(pinned_fit) > len(alpha_fit)  # the mixed arm does see the new data

    # beta's holdout tasks enter neither set: they stay a holdout
    beta_held = [
        t
        for t in mixed
        if t.split == "train"
        and driver.CORPUS_OF[t.behaviour] == "beta"
        and validation_skeleton(t)
    ]
    assert beta_held
    assert len(pinned_fit) + len(pinned) == sum(
        t.length for t in mixed if t.split == "train"
    ) - sum(t.length for t in beta_held)


def test_provenance_records_every_corpus(two_corpora, monkeypatch) -> None:
    """A mixed checkpoint must not claim it trained on the primary alone."""
    paths, _ = two_corpora
    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"], paths["beta"]])
    monkeypatch.setattr(driver, "DROP_BEHAVIOURS", frozenset({"beta-1"}))
    recorded = driver.provenance()
    assert [c["name"] for c in recorded["corpora"]] == ["alpha", "beta"]
    assert recorded["corpus"]["name"] == "alpha"
    assert recorded["dropped_behaviours"] == ["beta-1"]


def test_cell_path_separates_eval_corpora(two_corpora, monkeypatch, tmp_path) -> None:
    """A second corpus's timing cells must not overwrite the primary's."""
    paths, _ = two_corpora
    monkeypatch.setattr(driver, "CORPUS_MANIFESTS", [paths["alpha"], paths["beta"]])
    primary = driver.cell_path(tmp_path, "1m", 0, "cuda")
    monkeypatch.setattr(driver, "EVAL_CORPUS", "beta")
    other = driver.cell_path(tmp_path, "1m", 0, "cuda")
    assert primary.name == "1m-s0-cuda.json"
    assert other.name == "1m-s0-cuda-beta.json"
