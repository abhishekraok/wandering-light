import json

from wandering_light.evals import explorer


def test_seeded_solver_results_are_discovered_without_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "20260828_120000_000001"
    result_path = run_dir / "random" / "seed-3.json"
    artifact_path = (
        run_dir / "solver_artifacts" / "random" / "seed-3" / "function_usage.json"
    )
    result_path.parent.mkdir(parents=True)
    artifact_path.parent.mkdir(parents=True)
    (run_dir / "summary.json").write_text(json.dumps({"results": {"random": {}}}))
    result_path.write_text(json.dumps({"detailed_results": []}))
    artifact_path.write_text("{}")
    monkeypatch.setattr(explorer, "RESULTS_ROOT", str(tmp_path))

    runs = explorer._find_solver_runs()

    assert runs == [
        (
            "20260828_120000_000001/random/seed-3.json",
            str(result_path),
        )
    ]
    assert explorer._find_summary_path(str(result_path)) == str(
        run_dir / "summary.json"
    )
