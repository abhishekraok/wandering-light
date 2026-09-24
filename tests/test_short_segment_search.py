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
    beam_solve,
    segment_beam_solve,
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


def test_the_flat_beam_can_budget_fuel_or_both_counters():
    task = drawn("negate_each", 3)
    free = beam_solve(task, _Deterministic(3), beam_width=16, max_depth=8)
    assert free.fuel > free.executions  # fuel adds the quotation bodies' steps
    for unit, spent in (
        ("fuel", lambda r: r.fuel),
        ("executions+fuel", lambda r: r.executions + r.fuel),
    ):
        capped = beam_solve(
            task,
            _Deterministic(3),
            beam_width=16,
            max_depth=8,
            budget=1000,
            budget_unit=unit,
        )
        assert capped.termination == "budget"
        assert 1000 <= spent(capped) < 1000 + 50  # one attempt's worth over at most
    with pytest.raises(ValueError, match="budget unit"):
        beam_solve(task, _Deterministic(3), budget=10, budget_unit="seconds")


def test_widening_doubles_from_scratch_on_one_counter_and_stops_at_a_fit():
    task = witness_task("min_pair", 4)  # a 12-op witness: width 1 cannot keep it
    result, detail = widening_beam_solve(
        task,
        WitnessScorer(task, 0.5),
        budget=10**7,
        start_width=1,
        max_depth=14,
        max_state_atoms=16_384,
    )
    widths = [run["width"] for run in detail["widths"]]
    assert widths == [2**i for i in range(len(widths))]
    assert result.termination == "fit" and detail["fit_width"] == widths[-1]
    assert detail["largest_width_completed"] == widths[-1]
    assert result.executions == sum(run["executions"] for run in detail["widths"])
    # The fit is the one a single beam at that width finds, from scratch.
    alone = beam_solve(
        task,
        WitnessScorer(task, 0.5),
        beam_width=widths[-1],
        max_depth=14,
        max_state_atoms=16_384,
    )
    assert alone.program == result.program
    assert alone.executions == detail["widths"][-1]["executions"]


def test_widening_stops_on_the_shared_budget_and_charges_the_overshoot():
    task = drawn("negate_each", 3)
    for unit in ("executions", "executions+fuel"):
        result, detail = widening_beam_solve(
            task,
            _Deterministic(3),
            budget=20_000,
            start_width=4,
            max_depth=8,
            budget_unit=unit,
        )
        assert result.termination == "budget"
        assert len(detail["widths"]) > 1
        spent = result.executions + (result.fuel if unit != "executions" else 0)
        assert 20_000 <= spent < 20_000 + 50
        assert detail["fit_width"] is None
        assert detail["largest_width_completed"] == detail["widths"][-2]["width"]


def test_widening_stops_when_a_wider_beam_is_the_same_search():
    task = drawn("negate_each", 3)
    # A 2-deep search on four examples never fills a beam of 1,024, so the next
    # width repeats it exactly and the widening ends there, under budget.
    result, detail = widening_beam_solve(
        task, _Deterministic(3), budget=10**9, start_width=1024, max_depth=2
    )
    assert detail["saturated"]
    assert [run["width"] for run in detail["widths"]] == [1024, 2048]
    assert result.executions < 10**9


def test_a_widening_search_needs_budgets_and_a_segment_search_counts_executions():
    tasks = [drawn("negate_each", 3)]
    with pytest.raises(ValueError, match="needs a budget"):
        solve.solve_tasks(tasks, _Deterministic(3), tiny_system(search="widening"))
    with pytest.raises(ValueError, match="executions only"):
        solve.solve_tasks(
            tasks,
            _Deterministic(3),
            tiny_system(search="segment"),
            budgets=[100],
            budget_unit="fuel",
        )
    rows = solve.solve_tasks(
        tasks,
        _Deterministic(3),
        tiny_system(search="widening"),
        budgets=[5_000],
        budget_unit="fuel",
    )
    assert rows[0].termination == "budget"
    assert 5_000 <= rows[0].fuel < 5_050
    assert rows[0].record()["widening"]["widths"][0]["width"] == 8


def write_segment_cell(path: Path, spent: dict[int, tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"task_index": i, "executions": e, "fuel": f} for i, (e, f) in spent.items()
    ]
    path.write_text(json.dumps({"rows": rows}))


def test_the_matched_budget_is_read_per_task_in_its_unit(tmp_path):
    cell = tmp_path / "segment.json"
    write_segment_cell(cell, {40: (900, 1000), 7: (12, 20), 3: (5_000, 7_000)})
    assert driver.matched_budgets(cell, [7, 3, 40]) == [12, 5_000, 900]
    assert driver.matched_budgets(cell, [40], "fuel") == [1000]
    assert driver.matched_budgets(cell, [7, 40], "executions+fuel") == [32, 1900]
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


def test_cell_names_carry_k_depth_horizon_and_unit(tmp_path):
    run = tmp_path / "staged-arms-20260920"

    def name(arm, search, **kwargs):
        return driver.cell_path(tmp_path, run, arm, 0, search, **kwargs).name

    # The first wave's cells (eba479a) keep their names.
    assert name("staged", "flat") == "staged-s0-e3-flat-w64.json"
    assert name("capped", "segment") == "capped-s0-e3-segment-k16-d8.json"
    assert name("fused", "segment") == "fused-s0-e3-segment-k16-d48.json"
    assert name("capped", "flat-widening") == (
        "capped-s0-e3-flat-widening-k16-d8-h96-execfuel.json"
    )
    assert name("staged", "flat-widening", budget_unit="fuel") == (
        "staged-s0-e3-flat-widening-k16-d16-h192-fuel.json"
    )
    assert name("capped", "segment", roots=8, segment_depth=12) == (
        "capped-s0-e3-segment-k8-d12.json"
    )
    with pytest.raises(ValueError, match="segment depth"):
        name("refused", "segment")


def test_the_searches_are_best_with_the_registered_settings_changed():
    flat = driver.system_for("flat")
    segment = driver.system_for("segment", roots=8, segment_depth=16)
    widening = driver.system_for("flat-widening", segment_depth=16)
    for system in (flat, segment, widening):
        assert system.library and system.guard == "both"
        assert system.max_state_atoms == solve.BEST.max_state_atoms == 16_384
        assert system.beam_width == 64
    assert (flat.max_depth, flat.search) == (48, "flat")
    assert (segment.search, segment.roots) == ("segment", 8)
    assert (segment.segment_depth, segment.max_segments) == (16, 12)
    # The widening control runs at the segment cell's token horizon, 12 x depth.
    assert (widening.search, widening.max_depth) == ("widening", 192)
    assert [
        driver.system_for("flat-widening", segment_depth=d).max_depth
        for d in driver.SEGMENT_DEPTHS.values()
    ] == [96, 192, 576]


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


def test_three_cells_run_and_the_widening_cell_reads_the_segment_budgets(
    tmp_path, tiny_run, monkeypatch
):
    cohort = [
        (11, witness_task("sum", 6)),
        (22, witness_task("negate_each", 4)),
        (33, witness_task("sum", 7)),
    ]
    monkeypatch.setattr(driver, "cohort_tasks", lambda positions=None: cohort)
    out_dir = tmp_path / "out"
    kwargs = {"roots": 2, "segment_depth": 3, "seconds_cap": 30.0, "threads": 1}
    with pytest.raises(FileNotFoundError, match="segment cell first"):
        driver.run_cell(out_dir, tiny_run, "staged", 0, "flat-widening", **kwargs)
    cells = {
        s: json.loads(
            driver.run_cell(
                out_dir, tiny_run, "staged", 0, s, log=lambda _: None, lanes=2, **kwargs
            ).read_text()
        )
        for s in driver.SEARCHES
    }
    for search, cell in cells.items():
        assert cell["search"] == search
        assert cell["encoding"] == "machine"
        assert cell["task_seconds_cap"] == 30.0
        assert cell["provenance"]["commit"]
        assert cell["lanes"] == 2 and cell["tasks_per_hour"] > 0
        assert [r["task_index"] for r in cell["rows"]] == [11, 22, 33]
        for row in cell["rows"]:
            for field in ("solved", "shown_fit", "program", "executions", "fuel"):
                assert field in row
            assert row["termination"] and row["behaviour"]
            fit = row["program"] is not None
            assert (row["executions_at_fit"] == row["executions"]) is fit
    segment, widening = cells["segment"], cells["flat-widening"]
    assert segment["roots"] == 2 and segment["merge_rule"] == driver.MERGE_RULE
    assert segment["segment_depth"] == widening["segment_depth"] == 3
    assert segment["system"]["segment_depth"] == 3
    assert segment["search_tag"] == "segment-k2-d3"
    assert widening["max_depth"] == 36 and widening["budget_unit"] == "executions+fuel"
    assert cells["flat"]["segment_depth"] is None
    want = [r["executions"] + r["fuel"] for r in segment["rows"]]
    assert [r["budget"] for r in widening["rows"]] == want
    for row in widening["rows"]:
        assert row["budget_spent"] == row["executions"] + row["fuel"]
        assert row["overshoot"] == max(0, row["budget_spent"] - row["budget"])
        if row["source"] == "search":
            assert "widths" in row["widening"]
    use = widening["budget_use"]
    assert use["budget"] == sum(want) and use["unit"] == "executions+fuel"
    assert cells["flat"]["budget_use"] is None
    assert cells["flat"]["rows"][0]["budget"] is None


def test_the_report_counts_verified_behaviours_shortcuts_and_pairs(
    tmp_path, monkeypatch
):
    def check(program, behaviour, draws, seed):
        return {"verdict": "genuine" if behaviour.name == "sum" else "shortcut"}

    monkeypatch.setattr(verify, "check", check)
    sum_program = ["0", "[", "add", "]", "fold"]

    def row(index, behaviour, solved, program, shown_fit=None, executions=10, **extra):
        return {
            "task_index": index,
            "behaviour": behaviour,
            "solved": solved,
            "shown_fit": solved if shown_fit is None else shown_fit,
            "program": program,
            "executions": executions,
            "fuel": executions,
            "seconds": 1.0,
            "termination": "fit" if program else "max_depth",
            **extra,
        }

    def cell(run, arm, seed, search, tag, rows, **extra):
        path = tmp_path / run / f"{arm}-s{seed}-{tag}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "run": run,
                    "arm": arm,
                    "seed": seed,
                    "encoding": "machine",
                    "search": search,
                    "search_tag": tag,
                    "cohort_positions": None,
                    "rows": rows,
                    **extra,
                }
            )
        )

    run = driver.BEST_CELL[0]
    cell(
        run,
        "staged",
        0,
        "flat",
        "flat-w64",
        [
            row(1, "sum", True, sum_program, executions=10),
            row(2, "add_pair", True, ["add"]),
            row(3, "last", False, ["drop"], shown_fit=True),
            row(4, "reverse", False, None, termination="wall_clock_cap"),
            row(5, "product", False, None),
        ],
    )
    # The segment beam also solves `sum` (at 3x flat's cost, 60 in both
    # counters) and a `product` flat missed; it loses `add_pair`.
    cell(
        run,
        "staged",
        0,
        "segment",
        "segment-k16-d16",
        [
            row(1, "sum", True, sum_program, executions=30),
            row(5, "product", True, ["0", "[", "mul", "]", "fold"], executions=100),
        ],
    )
    budget_use = {
        "unit": "executions+fuel",
        "tasks": 1,
        "ended_by": {"budget": 1},
        "spent": 19,
        "budget": 20,
        "spent_fraction": 0.95,
        "overshoot": 0,
        "overshoot_tasks": 0,
        "max_largest_width_completed": 128,
    }
    cell(
        run,
        "staged",
        0,
        "flat-widening",
        "flat-widening-k16-d16-h192-execfuel",
        [row(1, "sum", True, sum_program, executions=19, budget=20)],
        budget_use=budget_use,
    )
    driver.report(tmp_path, tmp_path / "report", log=lambda _: None)
    result = json.loads((tmp_path / "report" / "report.json").read_text())
    flat = next(t for t in result["table"] if t["search"] == "flat-w64")
    assert flat["is_best"]
    assert flat["verified_per_seed"] == [1]
    facts = flat["cells"][0]
    assert facts["graded"] == ["add_pair", "sum"]
    assert facts["heldout_fail_tasks"] == 1
    assert facts["unverified_solve_tasks"] == 1
    assert facts["unverified_behaviours"] == ["add_pair"]
    assert facts["executions"] == 50 and facts["fuel"] == 50 and facts["capped"] == 1
    pairs = {p["search"]: p for p in result["paired"]}
    segment = pairs["segment-k16-d16"]
    # `product` is graded but the fake calls it a shortcut, so it is no gain.
    assert segment["gains"] == [] and segment["losses"] == []
    assert segment["curve"]["1"] == {"tasks": 0, "verified_behaviours": 0}
    assert segment["curve"]["4"]["verified_behaviours"] == 1
    # sum costs 60 against flat's 20; product 200 against flat's 20.
    assert segment["curve"]["4"]["tasks"] == 1
    assert segment["curve"]["16"]["tasks"] == 2
    text = (tmp_path / "report" / "report.md").read_text()
    assert "verifies 1; best-system-command (#158) measured 16" in text
    assert "Not reproduced" in text
    assert "0.95 of executions+fuel spent" in text
    for note in ("stop_at_fit=True", "moved == outputs", "Merge rule", "45x"):
        assert note in text


def test_the_queue_runs_the_registered_cells_in_the_advised_order(tmp_path):
    cells = run_cells.plan(tmp_path, list(driver.SEARCHES), None)
    assert len(cells) == 6 * 3 * 3
    labels = [(c.run_dir.name, c.arm, c.seed, c.search) for c in cells]
    difference = "short-segment-difference-20260923"
    assert labels[:4] == [
        (difference, "capped", 0, "flat"),
        (difference, "capped", 0, "segment"),
        (difference, "capped", 0, "flat-widening"),
        ("staged-arms-20260920", "staged", 0, "flat"),  # BEST, the equivalence check
    ]
    assert labels[4][:3] == ("short-segment-machine-20260923", "capped", 0)
    assert all(label[2] == 0 for label in labels[:18])
    assert all(label[2] > 0 for label in labels[18:])
    assert {c.arm for c in cells} == {"capped", "staged", "fused"}
    only = run_cells.plan(tmp_path, ["flat-widening"], {"staged-arms-20260920:fused:1"})
    assert [(c.arm, c.seed, c.search) for c in only] == [("fused", 1, "flat-widening")]


def test_a_widening_cell_starts_only_after_its_segment_cell_is_written(
    tmp_path, monkeypatch
):
    import argparse

    args = argparse.Namespace(
        out_dir=tmp_path / "out",
        roots=16,
        segment_depth=None,
        budget_unit="executions+fuel",
        jobs=2,
        poll=0.05,
        dry_run=False,
    )

    def fake(cell, args):
        out = cell.output(args)
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
        tmp_path, list(driver.SEARCHES), {"short-segment-machine-20260923:capped:0"}
    )
    lines = []
    assert run_cells.run_queue(cells, args, lines.append) == 0
    events = [line.split()[0] + " " + line.split()[1] for line in lines]
    label = "short-segment-machine-20260923:capped:s0"
    assert events.index(f"done {label}:segment") < events.index(
        f"start {label}:flat-widening"
    )
    # A rerun finds every output and starts nothing.
    again = []
    assert run_cells.run_queue(cells, args, again.append) == 0
    assert not any(line.startswith("start") for line in again)
