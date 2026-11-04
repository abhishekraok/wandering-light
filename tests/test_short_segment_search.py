"""The short-segment-campaign (#168) search driver and the caps it added to the beams."""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
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
    ArchivedSegment,
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
    # A 2-deep search on four examples never fills a beam of 1,024: nothing was
    # pruned, so any wider beam is this search, and the widening ends at once.
    result, detail = widening_beam_solve(
        task, _Deterministic(3), budget=10**9, start_width=1024, max_depth=2
    )
    assert detail["saturated"] and detail["stop"] == "saturated"
    assert [run["width"] for run in detail["widths"]] == [1024]
    assert detail["widths"][0]["pruned"] is False
    assert result.executions < 10**9


def test_equal_spend_at_two_widths_is_not_saturation(monkeypatch):
    """Codex's counterexample (#168 review 3): widths 1 and 2 fail at 7, width 4 fits.

    A finite graph with normalised, state-dependent policy probabilities, run
    through the real `beam_solve` with the machine patched out.
    """
    from types import SimpleNamespace

    from wandering_light import stack_beam as beam

    edges = {
        "R": [("A", 0.6), ("B", 0.3), ("F", 0.1)],
        "A": [("R", 0.99), ("C", 0.01)],
        "B": [("X", 0.5), ("Y", 0.5)],
        "C": [("D", 1.0)],
        "D": [("E", 1.0)],
        "F": [("T", 1.0)],
    }

    def step(product, op):
        options = edges.get(product[0][0], [])
        if op.index < len(options):
            return ((options[op.index][0],),), 1, 1
        return None, 0, 0

    class Graph:
        def log_probs(self, blocks, legal):
            scores = np.full(legal.shape, -np.inf)
            for row, state in enumerate(blocks):
                for col, (_, prob) in enumerate(edges.get(state, [])):
                    scores[row, col] = np.log(prob)
            return scores

    ops = tuple(SimpleNamespace(name=str(i), index=i) for i in range(3))
    monkeypatch.setattr(beam, "OPS", ops)
    monkeypatch.setattr(beam, "parse_stack", lambda text, **kw: (text,))
    monkeypatch.setattr(beam, "block_encoder", lambda *a: lambda p: p[0][0])
    monkeypatch.setattr(beam, "step_all_quiet", step)
    task = SimpleNamespace(inputs=("R",), outputs=("T",))
    flat = {
        w: beam.beam_solve(task, Graph(), beam_width=w, max_depth=8, guard="off")
        for w in (1, 2, 4)
    }
    assert flat[1].executions == flat[2].executions and flat[1].program is None
    assert flat[1].pruned and flat[2].pruned
    assert flat[4].termination == "fit"
    result, detail = beam.widening_beam_solve(
        task, Graph(), start_width=1, max_depth=8, guard="off", budget=100
    )
    assert not detail["saturated"] and detail["stop"] == "fit"
    assert result.program == flat[4].program and detail["fit_width"] == 4


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


def test_the_budget_is_the_larger_of_segment_and_flat_spend_per_task(tmp_path):
    segment = tmp_path / "segment.json"
    flat = tmp_path / "flat.json"
    write_segment_cell(segment, {40: (900, 1000), 7: (12, 20), 3: (5_000, 7_000)})
    write_segment_cell(flat, {3: (50, 60), 7: (30, 40), 40: (900, 1000)})
    budgets, terms = driver.matched_budgets(segment, flat, [7, 3, 40])
    assert budgets == [40, 7_000, 1000]  # fuel by default
    assert terms == ["flat", "segment", "segment"]  # a tie is the segment's
    budgets, terms = driver.matched_budgets(segment, flat, [7, 40], "executions+fuel")
    assert budgets == [70, 1900] and terms == ["flat", "segment"]
    with pytest.raises(ValueError, match="no row"):
        driver.matched_budgets(segment, flat, [7, 8])
    with pytest.raises(FileNotFoundError, match="segment and flat cells first"):
        driver.matched_budgets(segment, tmp_path / "absent.json", [7])


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
        "capped-s0-e3-flat-widening-k16-d8-h96-fuel.json"
    )
    assert name("staged", "flat-widening", budget_unit="executions+fuel") == (
        "staged-s0-e3-flat-widening-k16-d16-h192-execfuel.json"
    )
    assert name("capped", "segment", roots=8, segment_depth=12) == (
        "capped-s0-e3-segment-k8-d12.json"
    )
    with pytest.raises(ValueError, match="segment depth"):
        name("refused", "segment")
    # Every setting off its default is named; the defaults are the mapping.
    assert name("capped", "segment", scoring="local", roots=64) == (
        "capped-s0-e3-segment-local-k64-d8.json"
    )
    assert name("capped", "flat-widening", scoring="local", roots=64) == (
        "capped-s0-e3-flat-widening-local-k64-d8-h96-fuel.json"
    )
    assert name("staged", "flat", width=128) == "staged-s0-e3-flat-w128.json"
    assert name("capped", "segment", width=128) == (
        "capped-s0-e3-segment-k16-d8-w128.json"
    )
    assert name("capped", "flat-widening", width=32) == (
        "capped-s0-e3-flat-widening-k16-d8-h96-fuel-w32.json"
    )
    assert name("staged", "flat", seconds_cap=120) == "staged-s0-e3-flat-w64.json"
    assert name("staged", "flat", seconds_cap=300) == (
        "staged-s0-e3-flat-w64-cap300.json"
    )
    assert name("capped", "flat-widening", seconds_cap=300).endswith("-fuel.json")
    assert name("staged", "flat", device="cuda") == "staged-s0-e3-flat-w64-cuda.json"
    assert name("staged", "flat", threads=8) == "staged-s0-e3-flat-w64.json"
    assert name("staged", "flat", threads=1) == "staged-s0-e3-flat-w64-t1.json"
    # Clock caps keep every digit.
    assert name("staged", "flat", seconds_cap=1_000_000) != name(
        "staged", "flat", seconds_cap=1_000_001
    )
    assert name("staged", "flat", seconds_cap=1_000_001).endswith("-cap1000001.json")
    assert name("staged", "flat", seconds_cap=0.25).endswith("-cap0.25.json")
    # Subsets are named by their positions, not only their count.
    first = name("staged", "flat", positions=[0, 1])
    assert first != name("staged", "flat", positions=[2, 3])
    assert first.startswith("staged-s0-e3-flat-w64-p2-")


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
    kwargs = {"roots": 2, "segment_depth": 3, "threads": 1}
    with pytest.raises(FileNotFoundError, match="segment and flat cells first"):
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
        want_cap = 300.0 if search == "flat-widening" else 120.0
        assert cell["task_seconds_cap"] == want_cap
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
    assert segment["search_tag"] == "segment-k2-d3-t1"  # one thread, named
    assert widening["max_depth"] == 36 and widening["budget_unit"] == "fuel"
    assert cells["flat"]["segment_depth"] is None
    pairs = list(zip(segment["rows"], cells["flat"]["rows"], strict=True))
    want = [max(s["fuel"], f["fuel"]) for s, f in pairs]
    assert [r["budget"] for r in widening["rows"]] == want
    terms = ["segment" if s["fuel"] >= f["fuel"] else "flat" for s, f in pairs]
    assert [r["budget_term"] for r in widening["rows"]] == terms
    for row in widening["rows"]:
        assert row["budget_spent"] == row["fuel"]
        assert row["overshoot"] == max(0, row["budget_spent"] - row["budget"])
        if row["source"] == "search":
            assert "widths" in row["widening"]
    use = widening["budget_use"]
    assert use["budget"] == sum(want) and use["unit"] == "fuel"
    assert use["set_by_flat"] == terms.count("flat")
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
        path = tmp_path / run / f"{arm}-s{seed}-e3-{tag}.json"
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
            row(6, "second_largest", False, None, termination="wall_clock_cap"),
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
    assert "Count not reproduced" in text
    assert "0.95 of executions+fuel spent" in text
    assert "task 6 second_largest" in text
    for note in (
        "stop_at_fit=True",
        "moved == outputs",
        "Merge rule",
        "45x",
        "same order",
        "10,705",
        "max(the segment row's spend, flat-48's own spend",
    ):
        assert note in text


def test_the_queue_runs_the_registered_cells_in_the_advised_order(tmp_path):
    cells = run_cells.plan(tmp_path, list(driver.SEARCHES), None)
    assert len(cells) == 6 * 3 * 3 + 2
    labels = [(c.run_dir.name, c.arm, c.seed, c.search, c.scoring) for c in cells]
    difference = "short-segment-difference-20260923"
    assert labels[:6] == [
        (difference, "capped", 0, "flat", "cumulative"),
        (difference, "capped", 0, "segment", "cumulative"),
        (difference, "capped", 0, "flat-widening", "cumulative"),
        # BEST, the equivalence check
        ("staged-arms-20260920", "staged", 0, "flat", "cumulative"),
        # The exploratory pair, local scoring at K = 64.
        (difference, "capped", 0, "segment", "local"),
        (difference, "capped", 0, "flat-widening", "local"),
    ]
    assert {c.roots for c in cells[4:6]} == {64}
    assert labels[6][:3] == ("short-segment-machine-20260923", "capped", 0)
    assert all(label[2] == 0 for label in labels[:20])
    assert all(label[2] > 0 for label in labels[20:])
    assert len(run_cells.plan(tmp_path, list(driver.SEARCHES), None, False)) == 54
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
        budget_unit="fuel",
        seconds_cap=None,
        device="cpu",
        threads=8,
        jobs=2,
        poll=0.05,
        dry_run=False,
    )

    def fake(cell, args):
        out = cell.output(args)
        pause = {"segment": 0.5, "flat": 1.0}.get(cell.search, 0.0)
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
    start = events.index(f"start {label}:flat-widening")
    assert events.index(f"done {label}:segment") < start
    assert events.index(f"done {label}:flat") < start
    # A rerun finds every output and starts nothing.
    again = []
    assert run_cells.run_queue(cells, args, again.append) == 0
    assert not any(line.startswith("start") for line in again)


def test_library_replay_honours_the_budget_and_the_clock(monkeypatch):
    first = witness_task("sum", 4)
    scorer = WitnessScorer(first, 2.0)
    system = tiny_system(search="widening")
    rows = solve.solve_tasks(
        [first, first], scorer, system, budgets=[100_000, 0], budget_unit="fuel"
    )
    # Codex's case: a zero budget used to return the replayed fit anyway.
    assert rows[1].termination == "budget" and rows[1].program is None
    assert rows[1].fuel == 0 and not rows[1].solved
    roomy = solve.solve_tasks(
        [first, first], scorer, system, budgets=[100_000, 100_000], budget_unit="fuel"
    )
    assert roomy[1].source == "library" and roomy[1].solved
    # A budget that runs out part-way through a replay stops it there.
    part = solve.solve_tasks(
        [first, first], scorer, system, budgets=[100_000, 10], budget_unit="fuel"
    )
    assert part[1].termination == "budget" and 10 <= part[1].fuel < 10 + 20
    # The clock: a replay that finds the deadline passed stops, unsolved.
    clock = [0.0]
    real = solve.replay_capped

    def slow(*args, **kwargs):
        clock[0] += 10.0
        return real(*args, **kwargs)

    monkeypatch.setattr(solve.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        "wandering_light.stack_beam.time.perf_counter", lambda: clock[0]
    )
    monkeypatch.setattr(solve, "replay_capped", slow)
    rows = solve.solve_tasks([first, first], scorer, tiny_system(), seconds_cap=1.0)
    assert rows[1].termination == "wall_clock_cap" and not rows[1].solved


# segment_beam_solve at 6d19198, before scoring existed: cumulative is unchanged.
SEGMENT_GOLDEN = [
    # name, witness scorer?, (roots, segment_depth, max_segments, width), result
    (
        "sum",
        True,
        (4, 4, 4, 8),
        (("0", "nil", "dip", "[", "add", "]", "fold"), "fit", 7844, 8254, 118, 7),
    ),
    ("min_pair", True, (4, 4, 4, 8), (None, "frontier_exhausted", 2388, 2388, 25, 4)),
    ("keep_below", True, (4, 4, 4, 8), (None, "max_segments", 16344, 16518, 275, 13)),
]


@pytest.mark.parametrize(("name", "witness", "shape", "want"), SEGMENT_GOLDEN)
def test_cumulative_scoring_is_the_segment_beam_before(name, witness, shape, want):
    task = witness_task(name, 4)
    roots, depth, segments, width = shape
    kwargs = {
        "roots": roots,
        "segment_depth": depth,
        "max_segments": segments,
        "beam_width": width,
        "max_state_atoms": 16_384,
    }
    for extra in ({}, {"scoring": "cumulative"}):
        got = segment_beam_solve(task, WitnessScorer(task, 1.0), **kwargs, **extra)
        assert tuple(getattr(got, f) for f in FIELDS) == want


def test_local_scoring_picks_roots_by_segment_gain_and_restarts_them_at_zero(
    monkeypatch,
):
    from wandering_light import stack_beam as beam

    # What each root's segment closes, and the log-probability gained doing it.
    children = {
        (): [("A", -1.0), ("B", -3.0)],
        ("A",): [("A2", -2.0), ("C2", -2.5)],
        ("B",): [("B2", -1.0)],
    }

    def run(scoring):
        seen = []

        def fake(frontier, outputs, encode, scorer, tally, **kwargs):
            seen.append([(program, score) for _, program, score in frontier])
            closed = {}
            for _, program, score in frontier:
                for name, gain in children.get(program, []):
                    closed[((name,),)] = (score + gain, (*program, name))
            return None, None, closed

        monkeypatch.setattr(beam, "_run_segment", fake)
        task = witness_task("sum", 4)
        result = beam.segment_beam_solve(
            task, WitnessScorer(task), roots=2, max_segments=3, scoring=scoring
        )
        assert result.termination == "frontier_exhausted"
        return seen

    cumulative, local = run("cumulative"), run("local")
    assert cumulative[1] == [(("A",), -1.0), (("B",), -3.0)]
    assert local[1] == [(("A",), 0.0), (("B",), 0.0)]
    # Cumulatively A2 (-3) and C2 (-3.5) beat B2 (-4); by what each gained in
    # the segment, B2 (-1) and A2 (-2) beat C2 (-2.5), and restart at 0.
    assert cumulative[2] == [(("A", "A2"), -3.0), (("A", "C2"), -3.5)]
    assert local[2] == [(("B", "B2"), 0.0), (("A", "A2"), 0.0)]
    with pytest.raises(ValueError, match="scoring"):
        run("greedy")


def test_the_merge_keeps_the_best_gain_when_roots_restart_at_zero():
    """Two roots close into one state with `i`; the policy likes `i` at the second."""
    from wandering_light.stack_beam import OP_INDEX, _run_segment, _Tally
    from wandering_light.stack_machine import parse_stack
    from wandering_light.stack_pbe import BASIS, OP_BY_NAME
    from wandering_light.stack_policy import MACHINE, node_encoder

    one = (parse_stack("[ ( 1 2 ) ( ) ]", basis=BASIS),)
    two = (parse_stack("[ ( 1 2 ) ( dup drop ) ]", basis=BASIS),)
    closed_state = beam_step(one, OP_BY_NAME["i"])
    assert closed_state == beam_step(two, OP_BY_NAME["i"])
    outputs = (parse_stack("[ 0 0 0 0 ]", basis=BASIS),)
    encode = node_encoder(outputs, MACHINE)
    likes_i = encode(two).tobytes()

    class Scorer:
        def log_probs(self, blocks, legal):
            logits = np.zeros(legal.shape)
            for row, block in enumerate(blocks):
                if block.tobytes() == likes_i:
                    logits[row, OP_INDEX["i"]] = 5.0
            logits = np.where(legal, logits, -np.inf)
            peak = logits.max(axis=1, keepdims=True)
            shifted = logits - peak
            return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    def merged(first_score, second_score):
        _stop, _program, closed = _run_segment(
            [(one, ("a",), first_score), (two, ("b",), second_score)],
            outputs,
            encode,
            Scorer(),
            _Tally(),
            beam_width=4,
            segment_depth=1,
            horizon=10,
            guard="off",
            budget=None,
            max_state_atoms=None,
            stop_at_fit=False,
        )
        return closed[closed_state][1]

    # Local scoring: both roots at 0, so the higher gain, the second's, is kept.
    assert merged(0.0, 0.0) == ("b", "i")
    # Cumulative: the second root arrived far behind, so the first's is kept
    # (its `i` or its `dip`, which close into the same state at equal odds).
    assert merged(0.0, -50.0)[0] == "a"


def beam_step(product, op):
    from wandering_light.stack_beam import step_all_quiet

    return step_all_quiet(product, op)[0]


def test_witness_ranks_under_local_scoring_restart_every_root(monkeypatch):
    from wandering_light import stack_beam as beam

    task = witness_task("sum", 4)
    # A sum program with two cuts, after `dip` and after `fold`.
    kwargs = {
        "program": ("0", "nil", "dip", "[", "add", "]", "fold"),
        "roots": 2,
        "segment_depth": 4,
        "max_segments": 4,
        "beam_width": 8,
    }
    real = beam._run_segment
    frontiers = []

    def spy(frontier, *args, **kw):
        frontiers.append([score for _, _, score in frontier])
        return real(frontier, *args, **kw)

    monkeypatch.setattr(beam, "_run_segment", spy)
    cumulative = beam.segment_witness_ranks(task, WitnessScorer(task, 1.0), **kwargs)
    cumulative_scores = list(frontiers)
    frontiers.clear()
    local = beam.segment_witness_ranks(
        task, WitnessScorer(task, 1.0), scoring="local", **kwargs
    )
    assert len(local) == len(cumulative) > 1
    # Every root, forced witness roots included, starts each segment at 0.
    assert all(score == 0.0 for scores in frontiers for score in scores)
    assert any(score != 0.0 for scores in cumulative_scores[1:] for score in scores)


def test_a_second_report_does_not_read_the_first(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "check", lambda *a: {"verdict": "genuine"})
    driver.report(tmp_path, tmp_path / "report", log=lambda _: None)
    driver.report(tmp_path, tmp_path / "report", log=lambda _: None)
    assert (tmp_path / "report" / "report.md").exists()


def test_best_is_checked_task_by_task_against_158(tmp_path):
    reference = tmp_path / "b64.json"
    reference.write_text(
        json.dumps(
            {
                "rows": [
                    {"task_index": 1, "program": ["add"], "solved": True},
                    {"task_index": 2, "program": None, "solved": False},
                ]
            }
        )
    )
    cell = {
        "rows": [
            {
                "task_index": 1,
                "behaviour": "add_pair",
                "program": ["add"],
                "solved": True,
                "termination": "fit",
            },
            {
                "task_index": 2,
                "behaviour": "last",
                "program": ["drop"],
                "solved": False,
                "termination": "wall_clock_cap",
            },
        ]
    }
    check = driver.best_check(cell, reference)
    assert check["identical"] == 1 and check["capped"] == 1
    assert [d["task_index"] for d in check["differ"]] == [2]


def test_different_subsets_are_never_averaged_as_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "check", lambda *a: {"verdict": "genuine"})
    for positions, fuel in (([0, 1], 10), ([2, 3], 100)):
        path = driver.cell_path(
            tmp_path, tmp_path / "run", "staged", 0, "flat", positions=positions
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "task_index": p,
                "behaviour": "last",
                "solved": False,
                "shown_fit": False,
                "program": None,
                "executions": fuel,
                "fuel": fuel,
                "seconds": 1.0,
                "termination": "max_depth",
            }
            for p in positions
        ]
        path.write_text(
            json.dumps(
                {
                    "run": "run",
                    "arm": "staged",
                    "seed": 0,
                    "encoding": "machine",
                    "search": "flat",
                    "search_tag": "flat-w64",
                    "cohort_positions": positions,
                    "rows": rows,
                }
            )
        )
    driver.report(tmp_path, tmp_path / "report", log=lambda _: None)
    table = json.loads((tmp_path / "report" / "report.json").read_text())["table"]
    assert len(table) == 2
    assert sorted(t["cells"][0]["fuel"] for t in table) == [20, 200]
    assert all(t["subset"].startswith("p2-") for t in table)


def test_local_per_op_rank_divides_the_gain_by_the_segment_ops():
    from wandering_light.stack_beam import _rank_variants
    from wandering_light.stack_machine import Int

    def product(value):
        return ((Int(value),),)

    long_root = (product(9), tuple("abcdefghi"), 0.0)
    short_root = (product(1), ("a",), 0.0)
    # One-op gains -2 after a 9-op prefix and -1 after a 1-op prefix.
    worse = (product(20), (*long_root[1], "x"), -2.0)
    better = (product(30), (*short_root[1], "y"), -1.0)
    every = [better, worse]
    local = _rank_variants(
        every, [long_root, short_root], set(), product(30), local_scoring=True
    )
    assert local["per_op_rank"] == 1
    # Cumulative keeps the whole-program mean: -2 / 10 beats -1 / 2.
    cumulative = _rank_variants(every, [long_root, short_root], set(), product(30))
    assert cumulative["per_op_rank"] == 2


def segment_kwargs(shape):
    roots, depth, segments, width = shape
    return {
        "roots": roots,
        "segment_depth": depth,
        "max_segments": segments,
        "beam_width": width,
        "max_state_atoms": 16_384,
    }


@pytest.mark.parametrize("scoring", ["cumulative", "local"])
@pytest.mark.parametrize(("name", "witness", "shape", "want"), SEGMENT_GOLDEN)
def test_the_archive_keeps_every_segment_and_changes_nothing(
    name, witness, shape, want, scoring, monkeypatch
):
    from wandering_light import stack_beam as beam

    task = witness_task(name, 4)
    kwargs = {**segment_kwargs(shape), "scoring": scoring}
    plain = segment_beam_solve(task, WitnessScorer(task, 1.0), **kwargs)
    if scoring == "cumulative":
        assert tuple(getattr(plain, f) for f in FIELDS) == want

    returned = []
    real = beam._run_segment

    def spy(*args, **kw):
        out = real(*args, **kw)
        returned.append(out)
        return out

    monkeypatch.setattr(beam, "_run_segment", spy)
    archive = []
    got = segment_beam_solve(task, WitnessScorer(task, 1.0), archive=archive, **kwargs)
    assert tuple(getattr(got, f) for f in FIELDS) == tuple(
        getattr(plain, f) for f in FIELDS
    )

    # One entry per segment run, the one that stopped the search included.
    assert len(archive) == len(returned) >= 1
    assert all(isinstance(entry, ArchivedSegment) for entry in archive)
    assert [entry.segment for entry in archive] == list(range(len(archive)))
    assert archive[0].roots == (((), 0.0),)
    for entry, (stop, _program, closed) in zip(archive, returned, strict=True):
        assert entry.closed is closed  # a reference, not a copy
        assert entry.stop == stop
    last = archive[-1]
    if got.termination == "fit":
        assert last.stop == "fit"
        assert all(entry.stop is None for entry in archive[:-1])
    else:
        assert all(entry.stop is None for entry in archive)
    if got.termination == "max_segments":
        assert len(archive) == shape[2]

    # Each segment's roots are the previous segment's best unexpanded closed
    # states, in the order they ran, scored as the search scored them.
    for before, after in zip(archive, archive[1:]):
        ranked = sorted(before.closed.values(), key=lambda kept: kept[0], reverse=True)
        programs = [root for root, _ in after.roots]
        assert len(after.roots) <= shape[0]
        assert set(programs) <= {program for _, program in ranked}
        scores = [score for _, score in after.roots]
        if scoring == "local":
            assert scores == [0.0] * len(scores)
        else:
            by_program = {program: score for score, program in ranked}
            assert scores == [by_program[program] for program in programs]
            assert scores == sorted(scores, reverse=True)


def test_the_archive_records_a_budget_stop():
    task = witness_task("keep_below", 4)
    kwargs = segment_kwargs((4, 4, 4, 8))
    plain = segment_beam_solve(task, WitnessScorer(task, 1.0), budget=3_000, **kwargs)
    archive = []
    got = segment_beam_solve(
        task, WitnessScorer(task, 1.0), budget=3_000, archive=archive, **kwargs
    )
    assert got.termination == plain.termination == "budget"
    assert tuple(getattr(got, f) for f in FIELDS) == tuple(
        getattr(plain, f) for f in FIELDS
    )
    assert archive[-1].stop == "budget"
    assert all(entry.stop is None for entry in archive[:-1])


def test_solve_tasks_hands_each_row_its_archived_segments():
    first = witness_task("sum", 4)
    tasks = [first, first, drawn("negate_each", 3)]
    scorer = WitnessScorer(first, 1.0)
    system = tiny_system(
        search="segment", roots=4, segment_depth=4, max_segments=2
    )

    def records(rows):
        return [{k: v for k, v in r.record().items() if k != "seconds"} for r in rows]

    plain = solve.solve_tasks(tasks, scorer, system)
    calls = []
    archived = solve.solve_tasks(
        tasks, scorer, system, archive=lambda row, segments: calls.append((row, segments))
    )
    assert records(archived) == records(plain)
    assert [row for row, _ in calls] == archived
    assert archived[0].source == "search" and archived[0].termination == "fit"
    assert calls[0][1] and calls[0][1][-1].stop == "fit"
    # The second is the first again, fitted from the library: no segments ran.
    assert archived[1].source == "library"
    assert calls[1][1] == []
    assert archived[2].source == "search"
    assert len(calls[2][1]) >= 1
    assert all(isinstance(entry, ArchivedSegment) for entry in calls[2][1])

    flat = []
    solve.solve_tasks(
        tasks[2:], scorer, tiny_system(), archive=lambda row, segments: flat.append(segments)
    )
    assert flat == [[]]
