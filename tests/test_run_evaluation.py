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
    )

    summaries = list(Path(tmp_path).glob("*/summary.json"))
    assert len(summaries) == 1, "Summary file not created"
    summary = json.loads(summaries[0].read_text())
    assert summary["num_samples"] == 1
    assert "dummy" in summary["results"]
    assert summary["results"]["dummy"]["success_rate"] == 1.0
    assert summary["command"] == "custom command"
