"""Replay preserves historical query sampling and incomplete execution grades."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "reports/program-embeddings-20261024/replay_execution_rankings.py"
)
spec = importlib.util.spec_from_file_location("rankings_replay", SCRIPT)
replay_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_module)


def test_query_views_follow_partition_sampling_then_all_shown_views():
    records = [
        SimpleNamespace(
            key=str(i),
            split="test",
            programs=["p"],
            examples=[(str(j), str(i)) for j in range(8)],
            provenance={"partition": "holdout"},
        )
        for i in range(7)
    ]
    rng = random.Random(0)
    selected = sorted(rng.sample(records, 4), key=lambda r: r.key)
    expected = [("holdout", r.key, rng.sample(r.examples, 3)) for r in selected]
    actual = list(
        replay_module.shown_queries(
            records,
            {
                "split": "test",
                "seed": 0,
                "io_examples": 3,
                "max_eval_records": 4,
            },
        )
    )
    assert actual == expected


def test_replay_retains_unknowns_and_reports_historical_deviations(monkeypatch):
    record = SimpleNamespace(
        key="task",
        split="test",
        programs=["p"],
        examples=[("given", "wanted")],
        provenance={"partition": "holdout"},
    )
    outcomes = {"first": None, "second": True, "third": False}
    monkeypatch.setattr(replay_module.ev, "_matches", lambda p, shown: outcomes[p])
    source = {
        "records": [
            {
                "partition": "holdout",
                "key": "task",
                "top10": list(outcomes),
                "execution": {"shown@1": False, "shown@10": True},
            }
        ]
    }
    report = replay_module.replay(
        [record],
        source,
        {
            "split": "test",
            "seed": 0,
            "io_examples": 5,
            "max_eval_records": 128,
        },
    )
    assert report["records"][0]["shown_pass"] == [None, True, False]
    assert report["partitions"]["holdout"]["shown_mrr@10_bounds"] == [0.5, 1]
    assert report["partitions"]["holdout"]["shown@1_bounds"] == [0, 1]
    assert report["validation_deviations"] == [
        {
            "partition": "holdout",
            "key": "task",
            "metric": "shown@1",
            "historical": False,
            "replayed": None,
        }
    ]
