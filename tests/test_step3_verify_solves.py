"""The step-3 solve check: verdict rules, dense draws, reproducible streams (#134, #142)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from experiments import step3_verify_solves as verify
from wandering_light.stack_pbe import CATALOGUE

ROOT = Path(__file__).resolve().parents[1]
RECORDED = (
    ROOT / "reports" / "world-plus-v4-policy-20260912" / "verified_solves_epochs.json"
)
BY_NAME = {b.name: b for b in CATALOGUE}


def _tally(tested: int, failed: int) -> dict:
    return {"tested": tested, "failed": failed, "counterexample": None}


def _recorded(behaviour: str, verdict: str) -> dict:
    programs = json.loads(RECORDED.read_text())["programs"]
    return next(
        p
        for p in programs
        if p["behaviour"] == behaviour and p["verdict"] == verdict and "draws" in p
    )


def test_a_mode_below_the_floor_is_inconclusive_not_genuine() -> None:
    assert (
        verify.verdict_of({"sampler": _tally(600, 0), "perturbed": _tally(700, 0)}, 500)
        == "genuine"
    )
    assert (
        verify.verdict_of({"sampler": _tally(600, 0), "perturbed": _tally(499, 0)}, 500)
        == "inconclusive"
    )
    assert (
        verify.verdict_of({"sampler": _tally(600, 1), "perturbed": _tally(10, 0)}, 500)
        == "shortcut"
    )


def test_a_catalogue_witness_is_its_own_verdict_not_a_verified_solve() -> None:
    """It gets no draws, so it must not be counted with the draw-tested solves."""
    behaviour = BY_NAME["keep_below"]
    result = verify.check(behaviour.program, behaviour, draws=10, seed=0)
    assert result == {
        "verdict": "equals_catalogue_witness",
        "why": "equals a catalogue witness",
    }
    assert verify.COLUMN_OF[result["verdict"]] == "witness"
    assert verify.COLUMN_OF["genuine"] == "verified"


def test_fewer_draws_than_the_floor_cannot_reach_genuine() -> None:
    """The floor used to be min(--min-tested, --draws): fewer draws, more genuine."""
    entry = _recorded("drop_last", "genuine")
    program, behaviour = tuple(entry["program"].split()), BY_NAME["drop_last"]
    result = verify.check(program, behaviour, draws=100, seed=134, min_tested=500)
    assert result["verdict"] == "inconclusive"
    assert all(t["failed"] == 0 for t in result["draws"].values())
    assert (
        verify.check(program, behaviour, draws=100, seed=134, min_tested=50)["verdict"]
        == "genuine"
    )


def test_every_mode_reports_what_it_skipped() -> None:
    entry = _recorded("drop_last", "genuine")
    program, behaviour = tuple(entry["program"].split()), BY_NAME["drop_last"]
    result = verify.check(program, behaviour, draws=200, seed=134)
    for tally in result["draws"].values():
        assert tally["skipped"] == 200 - tally["tested"]
    # The sampler stays in the corpus's domain; perturbed draws include the
    # empty list, which `drop_last`'s witness refuses.
    assert result["draws"]["sampler"]["skipped"] == 0
    assert result["draws"]["perturbed"]["skipped"] > 0


def test_the_recorded_shortcut_is_still_caught_in_every_mode() -> None:
    entry = _recorded("all_positive", "shortcut")
    result = verify.check(
        tuple(entry["program"].split()),
        BY_NAME["all_positive"],
        draws=300,
        seed=134,
        min_tested=50,
    )
    assert result["verdict"] == "shortcut"
    assert set(result["draws"]) == {"sampler", "perturbed", "dense"}


def test_dense_draws_use_the_small_pool_and_keep_booleans() -> None:
    rng = random.Random(0)
    lengths = set()
    for _ in range(300):
        values, number, flag = verify.densify(rng, ([5, 9], 7, True))
        assert all(-2 <= v <= 2 for v in values)
        assert -2 <= number <= 2
        assert flag is True
        lengths.add(len(values))
    assert min(lengths) == 0 and max(lengths) == 10


def test_adding_dense_moves_neither_recorded_stream() -> None:
    """sampler and perturbed tallies must reproduce with and without dense."""
    entry = _recorded("drop_last", "genuine")
    program, behaviour = tuple(entry["program"].split()), BY_NAME["drop_last"]
    plain = verify.check(program, behaviour, draws=200, seed=134, dense=False)
    with_dense = verify.check(program, behaviour, draws=200, seed=134)
    assert set(plain["draws"]) == {"sampler", "perturbed"}
    assert set(with_dense["draws"]) == {"sampler", "perturbed", "dense"}
    for mode in ("sampler", "perturbed"):
        assert plain["draws"][mode] == with_dense["draws"][mode]


def test_partition_reads_files_written_before_and_after_the_witness_verdict() -> None:
    """Report readers take the union; old files graded witnesses `genuine` with no draws."""
    old = [
        {"behaviour": "keep_below", "program": "w", "verdict": "genuine", "why": "x"},
        {
            "behaviour": "add_pair",
            "program": "swap add",
            "verdict": "genuine",
            "draws": {},
        },
        {
            "behaviour": "all_positive",
            "program": "s",
            "verdict": "shortcut",
            "draws": {},
        },
        {
            "behaviour": "drop_last",
            "program": "i",
            "verdict": "inconclusive",
            "draws": {},
        },
    ]
    new = [{**old[0], "verdict": "equals_catalogue_witness"}, *old[1:]]
    for programs in (old, new):
        genuine, witness = verify.partition(programs)
        assert genuine == {("add_pair", "swap add")}
        assert witness == {("keep_below", "w")}
    programs = json.loads(RECORDED.read_text())["programs"]
    genuine, witness = verify.partition(programs)
    assert len(witness) == sum(1 for p in programs if "why" in p) == 9
    assert genuine.isdisjoint(witness)


def test_cli_defaults_are_the_ones_the_reports_quote() -> None:
    args = verify.build_parser().parse_args(["--run-dir", "x"])
    assert (args.draws, args.min_tested, args.dense) == (10_000, 500, True)
    assert (args.beam_width, args.max_depth) == (64, 48)
    off = verify.build_parser().parse_args(["--run-dir", "x", "--no-dense"])
    assert off.dense is False


def _cell(rows: list[dict], *, beam_width: int = 64, seed: int = 0) -> dict:
    return {
        "arm": "arm",
        "seed": seed,
        "eval_corpus": "stack_pbe_v4",
        "policy_epoch": 3,
        "beam_width": beam_width,
        "max_depth": 48,
        "rows": rows,
    }


def _row(behaviour: str, program: str | None, bucket: str = "1-5") -> dict:
    return {
        "behaviour": behaviour,
        "program": program.split() if program else None,
        "solved": program is not None,
        "bucket": bucket,
    }


def test_main_counts_witness_solves_apart_and_rescores_only_the_budget_asked_for(
    tmp_path: Path, capsys
) -> None:
    rows = [
        _row("keep_below", BY_NAME["keep_below"].witness, bucket="6-10"),
        _row("add_pair", "swap add"),
        _row("all_positive", _recorded("all_positive", "shortcut")["program"], "21-40"),
        _row("drop_last", None),
    ]
    run_dir = tmp_path / "arm"
    (run_dir / "timing").mkdir(parents=True)
    (run_dir / "timing" / "a.json").write_text(json.dumps(_cell(rows)))
    (run_dir / "timing" / "b.json").write_text(
        json.dumps(_cell(rows, beam_width=32, seed=1))
    )
    out = tmp_path / "out.json"
    quick = ["--run-dir", str(run_dir), "--draws", "20", "--min-tested", "5"]

    assert verify.main([*quick, "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    (cell,) = written["cells"]
    assert cell["seed"] == 0
    assert (
        cell["graded_total"],
        cell["verified_total"],
        cell["witness_total"],
        cell["inconclusive_total"],
    ) == (3, 1, 1, 0)
    assert cell["witness"]["6-10"] == 1 and cell["verified"]["6-10"] == 0
    assert cell["verified"]["1-5"] == 1
    assert {p["behaviour"]: p["verdict"] for p in written["programs"]} == {
        "keep_below": "equals_catalogue_witness",
        "add_pair": "genuine",
        "all_positive": "shortcut",
    }
    assert (written["draws_per_mode"], written["min_tested"]) == (20, 5)
    assert written["modes"] == ["sampler", "perturbed", "dense"]
    assert (written["beam_width"], written["max_depth"]) == (64, 48)
    assert written["catalogue_digest"].startswith("sha256:")
    assert written["catalogue_source_digest"].startswith("sha256:")
    printed = capsys.readouterr().out
    assert "skipping" in printed and "b.json" in printed
    assert "witness 1" in printed

    assert verify.main([*quick, "--beam-width", "32", "--out", str(out)]) == 0
    assert [c["seed"] for c in json.loads(out.read_text())["cells"]] == [1]
