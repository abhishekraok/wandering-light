"""The short-segment-campaign (#168) search driver and the caps it added to the beams."""

import json
import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import experiments.step3_verify_solves as verify
from experiments.short_segment import run_cells
from experiments.short_segment import search as driver
from tests.test_segment_beam import WitnessScorer
from tests.test_segment_beam import task_for as witness_task
from tests.test_stack_beam_equivalence import _Deterministic
from wandering_light import solve
from wandering_light.rollout_policy import PolicyArchitecture
from wandering_light.stack_beam import (
    BUDGET_UNITS,
    beam_solve,
    segment_beam_solve,
    spent_in,
    widening_beam_solve,
)
from wandering_light.stack_pbe import BY_NAME, build_task, draw_task
from wandering_light.stack_policy import save_policy, train_policy, witness_rows


def drawn(name: str, seed: int):
    return build_task(
        BY_NAME[name], draw_task(BY_NAME[name], random.Random(seed), 4, 6)
    )


FIELDS = ("program", "termination", "executions", "fuel", "expansions", "depth")

# `beam_solve` before #168 added `deadline`, at 41761c0: guard both, state cap
# 16,384. Under `_Deterministic(seed)` no search fits, so the whole depth is
# walked; under `WitnessScorer(task, 2.0)` each fits its witness.
DETERMINISTIC_GOLDEN = [
    # name, seed, width, depth, (termination, executions, fuel, expansions, depth)
    ("reverse", 9, 8, 6, ("max_depth", 2196, 2278, 41, 6)),
    ("negate_each", 3, 16, 8, ("max_depth", 6827, 7281, 109, 8)),
    ("keep_below", 5, 8, 10, ("max_depth", 4440, 4621, 73, 10)),
    ("sum", 4, 16, 8, ("max_depth", 6728, 7140, 109, 8)),
    ("min_pair", 2, 4, 12, ("max_depth", 3202, 3258, 45, 12)),
]
WITNESS_GOLDEN = [
    # name, program, executions, fuel, expansions
    ("sum", "0 [ add ] fold", 1956, 2100, 26),
    ("negate_each", "[ 0 swap sub ] map", 3288, 3396, 34),
    ("keep_below", "[ lt? ] swap cons filter", 2836, 2926, 34),
    ("reverse", "nil [ cons ] fold", 2004, 2152, 26),
    ("min_pair", "[ over lt? ] [ swap drop ] [ drop ] ifte", 7908, 8035, 82),
]


@pytest.mark.parametrize(
    ("name", "seed", "width", "depth", "want"), DETERMINISTIC_GOLDEN
)
def test_flat_without_budget_or_deadline_is_the_search_before(
    name, seed, width, depth, want
):
    task = drawn(name, seed)
    far = time.perf_counter() + 3600
    for deadline in (None, far):
        got = beam_solve(
            task,
            _Deterministic(seed),
            beam_width=width,
            max_depth=depth,
            guard="both",
            max_state_atoms=16_384,
            deadline=deadline,
        )
        assert got.program is None
        assert (
            got.termination,
            got.executions,
            got.fuel,
            got.expansions,
            got.depth,
        ) == want


@pytest.mark.parametrize(
    ("name", "program", "executions", "fuel", "expansions"), WITNESS_GOLDEN
)
def test_flat_fits_are_the_fits_before(name, program, executions, fuel, expansions):
    task = witness_task(name, 4)
    got = beam_solve(
        task,
        WitnessScorer(task, 2.0),
        beam_width=8,
        max_depth=len(task.program) + 2,
        guard="both",
        max_state_atoms=16_384,
    )
    assert got.termination == "fit"
    assert " ".join(got.program) == program
    assert (got.executions, got.fuel, got.expansions) == (executions, fuel, expansions)


def test_the_flat_beam_honours_a_budget():
    task = drawn("negate_each", 3)
    free = beam_solve(task, _Deterministic(3), beam_width=16, max_depth=8)
    assert free.executions == 6827
    for budget in (0, 1, 500, 3000):
        capped = beam_solve(
            task, _Deterministic(3), beam_width=16, max_depth=8, budget=budget
        )
        assert capped.termination == "budget"
        assert capped.program is None
        # Checked before every attempt, and an attempt runs one op on each
        # shown example, so a search overshoots by less than one attempt.
        assert budget <= capped.executions < budget + task.examples
    # A budget the search never reaches changes nothing. Reaching it exactly is
    # reaching it: an attempt refused on the first example costs nothing, so
    # the check can still fire before the last few attempts.
    roomy = beam_solve(
        task, _Deterministic(3), beam_width=16, max_depth=8, budget=free.executions + 1
    )
    assert [getattr(roomy, f) for f in FIELDS] == [getattr(free, f) for f in FIELDS]


def test_a_passed_deadline_stops_both_beams_before_any_work():
    task = witness_task("sum", 4)
    past = time.perf_counter() - 1
    flat = beam_solve(task, WitnessScorer(task), deadline=past)
    segment = segment_beam_solve(task, WitnessScorer(task), deadline=past)
    for result in (flat, segment):
        assert result.termination == "wall_clock_cap"
        assert result.program is None
        assert result.executions == 0


def tiny_system(**changes) -> solve.System:
    base = solve.System(
        name="tiny",
        repo_id="nobody/nothing",
        revision="0" * 40,
        beam_width=8,
        max_depth=8,
        guard="both",
        library=True,
        measured="a test",
        max_state_atoms=16_384,
    )
    return solve.System(**{**base.__dict__, **changes})


def test_solve_tasks_budgets_count_library_replay_and_none_changes_nothing():
    first = witness_task("sum", 4)
    tasks = [first, drawn("negate_each", 3), drawn("reverse", 9)]
    scorer = WitnessScorer(first, 2.0)
    system = tiny_system()

    def records(rows):
        return [{k: v for k, v in r.record().items() if k != "seconds"} for r in rows]

    plain = solve.solve_tasks(tasks, scorer, system)
    assert plain[0].termination == "fit"
    same = solve.solve_tasks(tasks, scorer, system, budgets=[None] * 3)
    assert records(plain) == records(same)
    capped = solve.solve_tasks(tasks, scorer, system, budgets=[None, 400, None])
    # The second task replays `sum`'s program from the library first, and that
    # is charged against its 400 as well.
    assert capped[1].termination == "budget"
    assert 400 <= capped[1].executions < 400 + tasks[1].examples
    assert records(capped)[::2] == records(plain)[::2]
    with pytest.raises(ValueError, match="budgets"):
        solve.solve_tasks(tasks, scorer, system, budgets=[1])


def write_segment_cell(path: Path, spent: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"task_index": i, "executions": e} for i, e in spent.items()]
    path.write_text(json.dumps({"rows": rows}))


def test_the_matched_budget_is_read_per_task(tmp_path):
    cell = tmp_path / "segment.json"
    write_segment_cell(cell, {40: 900, 7: 12, 3: 5_000})
    assert driver.matched_budgets(cell, [7, 3, 40]) == [12, 5_000, 900]
    assert driver.matched_budgets(cell, [40]) == [900]
    with pytest.raises(ValueError, match="no row"):
        driver.matched_budgets(cell, [7, 8])
    with pytest.raises(FileNotFoundError, match="segment cell first"):
        driver.matched_budgets(tmp_path / "absent.json", [7])


def test_an_existing_cell_is_skipped_without_loading_anything(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    out = driver.cell_path(tmp_path / "out", run_dir, "staged", 0, "segment")
    out.parent.mkdir(parents=True)
    out.write_text('{"kept": true}\n')

    def refuse(*args, **kwargs):
        raise AssertionError("a skipped cell loaded something")

    monkeypatch.setattr(driver, "load_policy", refuse)
    monkeypatch.setattr(driver, "cohort_tasks", refuse)
    lines = []
    got = driver.run_cell(
        tmp_path / "out", run_dir, "staged", 0, "segment", log=lines.append
    )
    assert got == out
    assert out.read_text() == '{"kept": true}\n'
    assert lines == [f"skip {out}: exists"]


def test_cell_names_keep_searches_k_and_width_apart(tmp_path):
    run = tmp_path / "staged-arms-20260920"
    names = {
        driver.cell_path(
            tmp_path, run, "staged", 0, s, roots=k, width=w, segment_depth=d
        ).name
        for s in driver.SEARCHES
        for k in (8, 16)
        for w in (512, 1024)
        for d in (None, 12)
    }
    # flat depends on none, segment on K and depth, flat-matched on all three.
    assert len(names) == 1 + 2 * 2 + 4 * 2
    assert driver.cell_path(tmp_path, run, "capped", 0, "segment").name == (
        "capped-s0-e3-segment-k16-d8.json"
    )
    assert driver.cell_path(tmp_path, run, "fused", 0, "segment").name == (
        "fused-s0-e3-segment-k16-d48.json"
    )
    with pytest.raises(ValueError, match="segment depth"):
        driver.cell_path(tmp_path, run, "refused", 0, "segment")
    assert driver.cell_path(tmp_path, run, "staged", 0, "flat").name == (
        "staged-s0-e3-flat-w64.json"
    )


def test_the_searches_are_best_with_one_setting_changed():
    flat = driver.system_for("flat")
    segment = driver.system_for("segment", roots=8)
    matched = driver.system_for("flat-matched", width=256)
    for system in (flat, segment, matched):
        assert system.library and system.guard == "both"
        assert system.max_state_atoms == solve.BEST.max_state_atoms == 16_384
    assert (flat.beam_width, flat.max_depth, flat.search) == (64, 48, "flat")
    assert (segment.search, segment.roots, segment.beam_width) == ("segment", 8, 64)
    assert (segment.segment_depth, segment.max_segments) == (12, 12)
    assert (matched.beam_width, matched.max_depth, matched.search) == (
        256,
        48,
        "flat",
    )


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """A run dir holding one tiny epoch-3 policy, fit to `sum`'s witness prefixes."""
    rows = witness_rows([witness_task("sum", 4), witness_task("sum", 5)])
    model, state = train_policy(
        rows,
        rows,
        PolicyArchitecture(8, 16, 32),
        epochs=40,
        learning_rate=1e-2,
        batch_size=8,
        validation_batch_size=8,
        seed=0,
        device="cpu",
    )
    run_dir = tmp_path_factory.mktemp("runs") / "tiny-run"
    save_policy(
        run_dir / "policies" / "staged-s0-e3",
        model,
        trainer_state=state,
        manifest={"arm": "staged"},
        wandb_url=None,
    )
    return run_dir


def test_three_cells_run_and_the_matched_cell_spends_the_segment_budgets(
    tmp_path, tiny_run, monkeypatch
):
    cohort = [
        (11, witness_task("sum", 6)),
        (22, witness_task("negate_each", 4)),
        (33, witness_task("sum", 7)),
    ]
    monkeypatch.setattr(driver, "cohort_tasks", lambda positions=None: cohort)
    out_dir = tmp_path / "out"
    kwargs = {"roots": 2, "width": 16, "seconds_cap": 30.0, "threads": 1}
    with pytest.raises(FileNotFoundError, match="segment cell first"):
        driver.run_cell(out_dir, tiny_run, "staged", 0, "flat-matched", **kwargs)
    cells = {
        s: json.loads(
            driver.run_cell(
                out_dir, tiny_run, "staged", 0, s, log=lambda _: None, **kwargs
            ).read_text()
        )
        for s in driver.SEARCHES
    }
    for search, cell in cells.items():
        assert cell["search"] == search
        assert cell["encoding"] == "machine"
        assert cell["task_seconds_cap"] == 30.0
        assert cell["provenance"]["commit"]
        assert [r["task_index"] for r in cell["rows"]] == [11, 22, 33]
        for row in cell["rows"]:
            for field in ("solved", "shown_fit", "program", "executions", "seconds"):
                assert field in row
            assert row["termination"] and row["behaviour"]
    segment, matched = cells["segment"], cells["flat-matched"]
    assert segment["roots"] == 2 and matched["beam_width"] == 16
    # staged's registered depth, in the cell and its name, and the matched cell's.
    assert segment["segment_depth"] == matched["segment_depth"] == 16
    assert segment["system"]["segment_depth"] == 16
    assert segment["search_tag"] == "segment-k2-d16"
    assert cells["flat"]["segment_depth"] is None
    want = [r["executions"] for r in segment["rows"]]
    assert [r["budget"] for r in matched["rows"]] == want
    for row in matched["rows"]:
        assert row["executions"] < row["budget"] + 4 or row["budget"] == 0
    assert matched["budget_use"]["budget"] == sum(want)
    assert cells["flat"]["budget_use"] is None
    assert cells["flat"]["rows"][0]["budget"] is None


def test_the_report_counts_verified_behaviours_and_shortcuts(tmp_path, monkeypatch):
    def check(program, behaviour, draws, seed):
        return {"verdict": "genuine" if behaviour.name == "sum" else "shortcut"}

    monkeypatch.setattr(verify, "check", check)

    def row(behaviour, solved, program, shown_fit=None, executions=10, **extra):
        return {
            "behaviour": behaviour,
            "solved": solved,
            "shown_fit": solved if shown_fit is None else shown_fit,
            "program": program,
            "executions": executions,
            "seconds": 1.0,
            "termination": "fit" if program else "max_depth",
            **extra,
        }

    def cell(run, arm, seed, search, rows, encoding="machine"):
        path = tmp_path / run / f"{arm}-s{seed}-{search}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run": run,
                    "arm": arm,
                    "seed": seed,
                    "encoding": encoding,
                    "search": search.split("-w")[0].split("-k")[0],
                    "search_tag": search,
                    "cohort_positions": None,
                    "rows": rows,
                }
            )
        )

    run = driver.BEST_CELL[0]
    cell(
        run,
        "staged",
        0,
        "flat-w64",
        [
            row("sum", True, ["0", "[", "add", "]", "fold"]),
            row("add_pair", True, ["add"]),
            row("last", False, ["drop"], shown_fit=True),
            row("reverse", False, None, termination="wall_clock_cap"),
        ],
    )
    cell(
        run,
        "staged",
        0,
        "flat-matched-w1024-k16",
        [row("sum", True, ["0", "[", "add", "]", "fold"], budget=20, executions=19)],
    )
    driver.report(tmp_path, tmp_path / "report", log=lambda _: None)
    table = json.loads((tmp_path / "report" / "report.json").read_text())
    flat = next(t for t in table if t["search"] == "flat-w64")
    assert flat["is_best"]
    assert flat["verified_per_seed"] == [1]
    facts = flat["cells"][0]
    assert facts["graded"] == ["add_pair", "sum"]
    assert facts["heldout_fail_tasks"] == 1
    assert facts["unverified_solve_tasks"] == 1
    assert facts["unverified_behaviours"] == ["add_pair"]
    assert facts["executions"] == 40 and facts["capped"] == 1
    matched = next(t for t in table if t["search"].startswith("flat-matched"))
    assert matched["cells"][0]["budget_use"]["spent_fraction"] == pytest.approx(0.95)
    text = (tmp_path / "report" / "report.md").read_text()
    assert "verifies 1; best-system-command (#158) measured 16" in text
    assert "Not reproduced" in text


def test_the_queue_puts_every_segment_cell_before_any_matched_one(tmp_path):
    cells = run_cells.plan(tmp_path, list(driver.SEARCHES), None)
    assert len(cells) == 3 * 3 * (2 + 3 + 1)
    first_matched = next(i for i, c in enumerate(cells) if c.search == "flat-matched")
    assert all(c.search == "flat-matched" for c in cells[first_matched:])
    assert cells[0].run_dir.name == driver.BEST_CELL[0]
    assert (cells[0].arm, cells[0].seed, cells[0].search) == ("staged", 0, "flat")
    only = run_cells.plan(tmp_path, ["flat-matched"], {"staged-arms-20260920:fused:1"})
    assert [(c.arm, c.seed, c.search) for c in only] == [("fused", 1, "flat-matched")]


def test_a_matched_cell_starts_only_after_its_segment_cell_is_written(
    tmp_path, monkeypatch
):
    import argparse

    args = argparse.Namespace(
        out_dir=tmp_path / "out",
        roots=16,
        width=1024,
        segment_depth=None,
        jobs=3,
        poll=0.05,
        dry_run=False,
    )

    def fake(cell, args):
        out = cell.output(args.out_dir, args.roots, args.width)
        pause = 0.5 if cell.search == "segment" else 0.0
        return [
            sys.executable,
            "-c",
            f"import time, pathlib; time.sleep({pause}); "
            f"pathlib.Path({str(out)!r}).write_text('{{}}')",
        ]

    monkeypatch.setattr(run_cells, "command", fake)
    monkeypatch.setattr(run_cells.Cell, "trained", lambda self: True)
    cells = run_cells.plan(
        tmp_path, list(driver.SEARCHES), {"staged-arms-20260920:staged:0"}
    )
    lines = []
    assert run_cells.run_queue(cells, args, lines.append) == 0
    events = [line.split()[0] + " " + line.split()[1] for line in lines]
    segment_done = events.index("done staged-arms-20260920:staged:s0:segment")
    matched_start = events.index("start staged-arms-20260920:staged:s0:flat-matched")
    assert segment_done < matched_start
    # A rerun finds every output and starts nothing.
    again = []
    assert run_cells.run_queue(cells, args, again.append) == 0
    assert not any(line.startswith("start") for line in again)


# ---------------------------------------------------------------------------
# Budget units and the widening flat beam
# ---------------------------------------------------------------------------


class _Refusing:
    """A scorer that fails the test if the search ever asks it anything."""

    def log_probs(self, blocks, legal):
        raise AssertionError("the search did work before refusing")


def test_spent_in_counts_each_unit_and_refuses_others():
    assert BUDGET_UNITS == ("executions", "fuel", "executions+fuel")
    assert spent_in("executions", 7, 30) == 7
    assert spent_in("fuel", 7, 30) == 30
    assert spent_in("executions+fuel", 7, 30) == 37
    with pytest.raises(ValueError, match="unknown budget unit 'ops'"):
        spent_in("ops", 7, 30)


def test_an_unknown_unit_is_refused_before_any_work():
    task = drawn("negate_each", 3)
    for budget in (None, 100):
        with pytest.raises(ValueError, match="unknown budget unit"):
            beam_solve(task, _Refusing(), budget=budget, budget_unit="ops")
        with pytest.raises(ValueError, match="unknown budget unit"):
            widening_beam_solve(task, _Refusing(), budget=100, budget_unit="ops")


@pytest.mark.parametrize("unit", ["fuel", "executions+fuel"])
def test_the_flat_beam_counts_its_budget_in_the_unit(unit):
    task = drawn("negate_each", 3)
    free = beam_solve(task, _Deterministic(3), beam_width=16, max_depth=8)
    assert spent_in(unit, free.executions, free.fuel) > 3000
    for budget in (0, 1, 500, 3000):
        capped = beam_solve(
            task,
            _Deterministic(3),
            beam_width=16,
            max_depth=8,
            budget=budget,
            budget_unit=unit,
        )
        assert capped.termination == "budget"
        spent = spent_in(unit, capped.executions, capped.fuel)
        assert spent >= budget
        # Under one attempt over: the search stopped at the first check past
        # the budget, so it was under the budget one attempt earlier.
        assert capped.executions < budget + task.examples
    # In fuel a budget runs out sooner than the same number of executions.
    by_executions = beam_solve(
        task, _Deterministic(3), beam_width=16, max_depth=8, budget=3000
    )
    by_both = beam_solve(
        task,
        _Deterministic(3),
        beam_width=16,
        max_depth=8,
        budget=3000,
        budget_unit="executions+fuel",
    )
    assert by_both.executions < by_executions.executions
    roomy = beam_solve(
        task,
        _Deterministic(3),
        beam_width=16,
        max_depth=8,
        budget=spent_in(unit, free.executions, free.fuel) + 1,
        budget_unit=unit,
    )
    assert [getattr(roomy, f) for f in FIELDS] == [getattr(free, f) for f in FIELDS]


def test_widening_that_fits_at_the_first_width_is_one_flat_beam():
    task = witness_task("sum", 4)
    kwargs = {"max_depth": len(task.program) + 2, "max_state_atoms": 16_384}
    flat = beam_solve(task, WitnessScorer(task, 2.0), beam_width=8, **kwargs)
    got, info = widening_beam_solve(
        task, WitnessScorer(task, 2.0), budget=10**6, start_width=8, **kwargs
    )
    assert [getattr(got, f) for f in FIELDS] == [getattr(flat, f) for f in FIELDS]
    assert info == {
        "widths": [{"width": 8, "termination": "fit", "executions": flat.executions}],
        "largest_width_completed": 8,
        "fit_width": 8,
        "saturated": False,
    }


def test_widening_doubles_from_scratch_until_the_shared_budget_runs_out():
    task = drawn("negate_each", 3)
    budget = 40_000
    got, info = widening_beam_solve(
        task, _Deterministic(3), budget=budget, start_width=4, max_depth=8
    )
    widths = [w["width"] for w in info["widths"]]
    assert widths == [4 * 2**i for i in range(len(widths))]
    assert len(widths) >= 2
    # Every restart but the last walked its whole depth, each from scratch:
    # the same search as a lone flat beam at that width.
    for entry in info["widths"][:-1]:
        lone = beam_solve(
            task, _Deterministic(3), beam_width=entry["width"], max_depth=8
        )
        assert (entry["termination"], entry["executions"]) == (
            lone.termination,
            lone.executions,
        )
    assert info["widths"][-1]["termination"] == "budget"
    assert got.termination == "budget" and got.program is None
    assert got.executions == sum(w["executions"] for w in info["widths"])
    assert budget <= got.executions < budget + task.examples
    assert info["largest_width_completed"] == widths[-2]
    assert info["fit_width"] is None and not info["saturated"]
    assert got.depth == 8


def test_widening_with_a_tiny_budget_completes_no_width():
    task = drawn("negate_each", 3)
    got, info = widening_beam_solve(
        task, _Deterministic(3), budget=100, start_width=16, max_depth=8
    )
    assert info["widths"] == [
        {"width": 16, "termination": "budget", "executions": got.executions}
    ]
    assert info["largest_width_completed"] is None
    assert info["fit_width"] is None


def test_widening_stops_when_a_wider_beam_is_the_same_search():
    # One layer from the inputs has at most 28 successors, so no beam of 64 or
    # more ever fills and the second restart repeats the first.
    task = drawn("negate_each", 3)
    got, info = widening_beam_solve(
        task, _Deterministic(3), budget=10**6, start_width=64, max_depth=1
    )
    assert [w["width"] for w in info["widths"]] == [64, 128]
    first, second = info["widths"]
    assert first["executions"] == second["executions"]
    assert info["saturated"] and info["largest_width_completed"] == 128
    assert got.termination == first["termination"] == "max_depth"
    assert got.executions == 2 * first["executions"]


def test_widening_hands_the_unit_to_every_restart():
    task = drawn("negate_each", 3)
    budget = 20_000
    got, _info = widening_beam_solve(
        task,
        _Deterministic(3),
        budget=budget,
        start_width=4,
        max_depth=8,
        budget_unit="fuel",
    )
    assert got.termination == "budget"
    assert got.fuel >= budget
    plain, _ = widening_beam_solve(
        task, _Deterministic(3), budget=budget, start_width=4, max_depth=8
    )
    assert got.executions < plain.executions


def test_the_widening_system_describes_itself():
    system = tiny_system(search="widening")
    assert system.describe().endswith(
        ", search widening: the width doubles per restart within a budget"
    )
    assert "widening" not in tiny_system().describe()


def test_solve_tasks_refuses_bad_units_and_unbudgeted_widening_up_front():
    tasks = [witness_task("sum", 4), drawn("negate_each", 3)]
    refusing = _Refusing()
    with pytest.raises(ValueError, match="unknown budget unit"):
        solve.solve_tasks(tasks, refusing, tiny_system(), budget_unit="ops")
    with pytest.raises(ValueError, match="the segment search budgets executions only"):
        solve.solve_tasks(
            tasks, refusing, tiny_system(search="segment"), budget_unit="fuel"
        )
    widening = tiny_system(search="widening")
    for budgets in (None, [1000, None]):
        with pytest.raises(ValueError, match="a widening search needs a budget"):
            solve.solve_tasks(tasks, refusing, widening, budgets=budgets)


def test_solve_tasks_runs_widening_and_records_it():
    first = witness_task("sum", 4)
    tasks = [first, drawn("negate_each", 3), witness_task("sum", 4)]
    scorer = WitnessScorer(first, 2.0)
    rows = solve.solve_tasks(
        tasks, scorer, tiny_system(search="widening"), budgets=[10**6, 5_000, 10**6]
    )
    assert rows[0].termination == "fit"
    assert rows[0].widening["fit_width"] == 8
    assert rows[0].record()["widening"] == rows[0].widening
    assert rows[1].termination == "budget"
    assert 5_000 <= rows[1].executions < 5_000 + tasks[1].examples
    assert rows[1].widening["widths"][0]["width"] == 8
    # The third is solved by replaying the first's program; no search ran.
    assert rows[2].source == "library" and rows[2].widening is None
    assert "widening" not in rows[2].record()
    flat = solve.solve_tasks(tasks, scorer, tiny_system())
    assert all("widening" not in row.record() for row in flat)


def test_solve_tasks_charges_library_replay_in_the_budget_unit():
    first = witness_task("sum", 4)
    tasks = [first, drawn("negate_each", 3)]
    scorer = WitnessScorer(first, 2.0)
    rows = solve.solve_tasks(
        tasks, scorer, tiny_system(), budgets=[None, 900], budget_unit="fuel"
    )
    assert rows[1].termination == "budget"
    # Replay and search together: the fuel crosses the budget by under one
    # attempt, and the replay's fuel is part of it.
    assert rows[1].fuel >= 900
    assert rows[1].executions < 900
