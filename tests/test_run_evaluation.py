import json
from pathlib import Path

import pytest

from wandering_light.evals import run_evaluation as run_eval
from wandering_light.evals.evaluate_solver import EvalResult
from wandering_light.function_def import FunctionDef, FunctionDefList
from wandering_light.trajectory import TrajectorySpec, TrajectorySpecList
from wandering_light.typed_list import TypedList


class DummySolver:
    def __init__(self):
        self.saved = []

    def save(self, directory: str):
        self.saved.append(directory)


@pytest.fixture
def sample_eval_file(tmp_path):
    inc = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    spec_list = TrajectorySpecList(
        [TrajectorySpec(TypedList([1], int), FunctionDefList([inc]))]
    )
    file_path = tmp_path / "eval_data.py"
    spec_list.to_py_file(str(file_path), variable_name="specs")
    return file_path


@pytest.fixture
def multi_eval_file(tmp_path):
    inc = FunctionDef(
        name="inc",
        input_type="builtins.int",
        output_type="builtins.int",
        code="return x + 1",
    )
    specs = TrajectorySpecList(
        [
            TrajectorySpec(TypedList([index], int), FunctionDefList([inc]))
            for index in range(6)
        ]
    )
    file_path = tmp_path / "multi_eval_data.py"
    specs.to_py_file(str(file_path), variable_name="specs")
    return file_path


def test_run_evaluation_creates_summary(monkeypatch, tmp_path, sample_eval_file):
    def fake_get_solver(name, budget=1, **kwargs):
        return DummySolver()

    def fake_eval(
        solver,
        trajectory_specs,
        available_functions,
        num_samples=None,
        save_results=True,
        output_dir=".",
    ):
        return EvalResult(
            total_samples=1,
            success_count=1,
            success_rate=1.0,
            avg_solution_length=1.0,
            failures=[],
            detailed_results=[],
        )

    monkeypatch.setattr(run_eval, "get_solver_by_name", fake_get_solver)
    monkeypatch.setattr(
        run_eval.EvaluateSolver,
        "evaluate_using_trajectories",
        staticmethod(fake_eval),
    )

    run_eval.run_evaluation(
        eval_file=str(sample_eval_file),
        solver_names=["dummy"],
        num_samples=1,
        budget=1,
        output_dir=str(tmp_path),
        variable_name="specs",
        command="custom command",
        basis_set_id="wl-core-v1",
        trusted_legacy_python=True,
    )

    summaries = list(Path(tmp_path).glob("*/summary.json"))
    assert len(summaries) == 1, "Summary file not created"
    summary = json.loads(summaries[0].read_text())
    assert summary["num_samples"] == 1
    assert "dummy" in summary["results"]
    assert summary["results"]["dummy"]["success_rate"] == 1.0
    assert summary["command"] == "custom command"


def test_legacy_eval_requires_trust_and_fixed_hash_seed(monkeypatch, sample_eval_file):
    with pytest.raises(ValueError, match="execute Python code"):
        run_eval.load_eval_data_as_trajectories(
            str(sample_eval_file),
            variable_name="specs",
            basis_set_id="wl-core-v1",
        )

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    with pytest.raises(RuntimeError, match="fixed seed"):
        run_eval.load_eval_data_as_trajectories(
            str(sample_eval_file),
            variable_name="specs",
            trusted_legacy_python=True,
        )


def test_bootstrap_ci_uses_seed_level_measurements():
    first = run_eval.bootstrap_mean_confidence_interval(
        [0.0, 0.5, 1.0],
        samples=500,
        seed=7,
    )
    second = run_eval.bootstrap_mean_confidence_interval(
        [0.0, 0.5, 1.0],
        samples=500,
        seed=7,
    )

    assert first == second
    assert first["unit"] == "seed"
    assert first["n"] == 3
    assert first["lower"] is not None
    assert first["upper"] is not None
    singleton = run_eval.bootstrap_mean_confidence_interval([0.5], samples=10)
    assert singleton["lower"] is None
    assert singleton["upper"] is None


def test_run_evaluation_freezes_cohort_and_aggregates_random_seeds(
    monkeypatch,
    tmp_path,
    multi_eval_file,
):
    calls = []
    factory_calls = []

    class SeedSolver(DummySolver):
        def __init__(self, name, seed, budget):
            super().__init__()
            self.name = name
            self.seed = seed
            self.budget = budget

    def fake_get_solver(name, budget=1, seed=None, **kwargs):
        factory_calls.append((name, budget, seed, kwargs))
        return SeedSolver(name, seed, budget)

    def fake_eval(solver, trajectories, available_functions, **kwargs):
        cohort = tuple(str(trajectory.input) for trajectory in trajectories)
        calls.append((solver.name, solver.seed, cohort))
        total = len(trajectories)
        successes = min(solver.seed, total)
        budget_unit = (
            "candidate_programs"
            if solver.name == "random"
            else "basis_function_executions"
        )
        return EvalResult(
            total_samples=total,
            success_count=successes,
            success_rate=successes / total,
            avg_solution_length=1.0 if successes else 0.0,
            failures=[],
            cost_unit="basis_function_executions",
            cost_definition="test cost",
            budget_unit=budget_unit,
            budget_value=solver.budget,
            costed_samples=total,
            total_charged_executions=total * (solver.seed + 1),
            mean_charged_executions=solver.seed + 1,
            total_candidate_programs=(total if solver.name == "random" else None),
            mean_candidate_programs=(1.0 if solver.name == "random" else None),
            solver_config={
                "cost_unit": "basis_function_executions",
                "budget_unit": budget_unit,
                "budget_value": solver.budget,
                "seed": solver.seed,
            },
        )

    monkeypatch.setattr(run_eval, "get_solver_by_name", fake_get_solver)
    monkeypatch.setattr(
        run_eval.EvaluateSolver,
        "evaluate_using_trajectories",
        staticmethod(fake_eval),
    )

    summary = run_eval.run_evaluation(
        eval_file=str(multi_eval_file),
        solver_names=["random", "bfs"],
        num_samples=3,
        budget=4,
        output_dir=str(tmp_path / "results"),
        variable_name="specs",
        basis_set_id="wl-core-v1",
        trusted_legacy_python=True,
        random_path_lengths=[1, 3],
        bfs_max_depth=5,
        seeds=[1, 2, 3],
        cohort_seed=11,
        bootstrap_samples=100,
    )

    assert len(calls) == 4
    assert [call[2] for call in factory_calls if call[0] == "random"] == [1, 2, 3]
    assert all(
        call[3]["path_lengths"] == (1, 3)
        for call in factory_calls
        if call[0] == "random"
    )
    bfs_factory_call = next(call for call in factory_calls if call[0] == "bfs")
    assert bfs_factory_call[3]["max_depth"] == 5
    assert {cohort for _, _, cohort in calls} == {calls[0][2]}
    assert summary["schema_version"] == run_eval.EVALUATION_SCHEMA_VERSION
    assert summary["cohort"]["size"] == 3
    assert len(summary["cohort"]["task_ids"]) == 3
    assert summary["random_path_lengths"] == [1, 3]
    assert summary["bfs_max_depth"] == 5

    random_result = summary["results"]["random"]
    assert random_result["seeds"] == [1, 2, 3]
    assert random_result["seed_effective"]
    assert random_result["total_samples"] == 9
    assert random_result["success_count"] == 6
    assert random_result["success_rate"] == pytest.approx(2 / 3)
    assert random_result["success_rate_ci"]["n"] == 3
    assert random_result["success_rate_ci"]["lower"] is not None
    assert random_result["mean_charged_executions_ci"]["n"] == 3
    assert random_result["mean_charged_executions_ci"]["lower"] is not None
    assert random_result["budget_unit"] == "candidate_programs"
    assert random_result["cost_unit"] == "basis_function_executions"

    bfs_result = summary["results"]["bfs"]
    assert bfs_result["seeds"] == [1]
    assert not bfs_result["seed_effective"]
    assert bfs_result["success_rate_ci"]["lower"] is None
    assert "reason" in bfs_result["success_rate_ci"]
    assert bfs_result["budget_unit"] == "basis_function_executions"

    run_dirs = list((tmp_path / "results").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "random" / "seed-1.json").is_file()
    assert (run_dirs[0] / "random" / "seed-3.json").is_file()
    assert (run_dirs[0] / "bfs" / "seed-1.json").is_file()


def test_empty_cohort_is_rejected():
    from wandering_light.trajectory import TrajectoryList

    with pytest.raises(ValueError, match="at least one trajectory"):
        run_eval._freeze_cohort(TrajectoryList(), num_samples=None, seed=0)


def test_default_baseline_seeds_are_five_replicates():
    assert run_eval.DEFAULT_BASELINE_SEEDS == (0, 1, 2, 3, 4)


def test_run_evaluation_propagates_invalid_corpus_errors(monkeypatch, tmp_path):
    def fail_load(*args, **kwargs):
        raise ValueError("invalid ground truth")

    monkeypatch.setattr(run_eval, "load_eval_data_as_trajectories", fail_load)

    with pytest.raises(ValueError, match="invalid ground truth"):
        run_eval.run_evaluation(
            eval_file="invalid.py",
            solver_names=["bfs"],
            output_dir=str(tmp_path),
            trusted_legacy_python=True,
        )
