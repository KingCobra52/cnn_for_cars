"""Regression tests for safe orchestration, provenance, and publication."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from carvision.cli import build_parser
from carvision.eval import compare_runs
from carvision.utils.artifacts import dependencies, fingerprint, publish, recover


def make_evaluation(root, name, ids, labels, classes=None):
    run = root / name
    run.mkdir()
    np.savez(
        run / "test_predictions.npz",
        logits=np.tile([1.0, 0.0], (len(ids), 1)),
        image_ids=np.array(ids),
        labels=np.array(labels),
    )
    (run / "checkpoint.pt").write_bytes(b"checkpoint")
    record = dependencies([run / "test_predictions.npz", run / "checkpoint.pt"], "carvision eval")
    record["predictions_sha256"] = fingerprint(run / "test_predictions.npz")
    from carvision.utils.artifacts import seal_evaluation

    data = {"class_names": classes or ["a", "b"], "provenance": record}
    seal_evaluation(data)
    (run / "evaluation.json").write_text(json.dumps(data))
    return run


def test_comparison_aligns_and_rejects_conflicting_labels(tmp_path):
    first = make_evaluation(tmp_path, "a", ["x", "y"], [0, 1])
    reordered = make_evaluation(tmp_path, "b", ["y", "x"], [1, 0])
    assert compare_runs(first, reordered).point == 0
    conflicting = make_evaluation(tmp_path, "c", ["x", "y"], [1, 0])
    with pytest.raises(ValueError, match="conflicting labels"):
        compare_runs(first, conflicting)


@pytest.mark.parametrize("kind", ["duplicate", "classes", "checkpoint", "predictions"])
def test_comparison_rejects_incompatible_artifacts(tmp_path, kind):
    a = make_evaluation(tmp_path, "a", ["x", "y"], [0, 1])
    b = make_evaluation(
        tmp_path,
        "b",
        ["x", "x"] if kind == "duplicate" else ["x", "y"],
        [0, 1],
        ["b", "a"] if kind == "classes" else None,
    )
    if kind == "checkpoint":
        (b / "checkpoint.pt").write_bytes(b"changed")
    elif kind == "predictions":
        with (b / "test_predictions.npz").open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        compare_runs(a, b)


@pytest.mark.parametrize("failure", range(4))
def test_publication_restores_all_files_on_each_replace_failure(tmp_path, monkeypatch, failure):
    files = {tmp_path / str(i): b"new" for i in range(4)}
    for path in files:
        path.write_bytes(b"old")
    original = Path.replace
    calls = 0

    def replace(path, target):
        nonlocal calls
        index = calls
        calls += 1
        if index == failure:
            raise OSError("injected")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="injected"):
        publish(files, tmp_path / "journal.json")
    assert all(path.read_bytes() == b"old" for path in files)
    assert not (tmp_path / "journal.json").exists()


def test_recovery_restores_interrupted_publication(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    target, backup = tmp_path / "target", stage / "old"
    target.write_bytes(b"new")
    backup.write_bytes(b"old")
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "staging": str(stage),
                "entries": [{"target": str(target), "backup": str(backup), "existed": True}],
            }
        )
    )
    recover(journal)
    assert target.read_bytes() == b"old"
    assert not journal.exists()


def test_eval_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "--all", "--run", "anything"])


@pytest.mark.parametrize("target", ["report", "figures", "export"])
def test_individual_make_targets_do_not_train(target):
    result = subprocess.run(["make", "-n", target], capture_output=True, text=True, check=True)
    assert "carvision sweep" not in result.stdout
    assert "carvision train" not in result.stdout


@pytest.mark.parametrize("fail", ["", "eval"])
def test_parallel_make_all_orders_stages_and_stops(tmp_path, fail):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "Makefile").write_bytes((repo / "Makefile").read_bytes())
    bin_dir = tmp_path / "env/bin"
    bin_dir.mkdir(parents=True)
    script = bin_dir / "carvision"
    script.write_text(
        '#!/bin/sh\necho "$*" >> "$TASK_LOG"\nif [ "$1" = "$TASK_FAIL" ]; then exit 9; fi\n'
    )
    script.chmod(0o755)
    log = tmp_path / "commands"
    env = {**os.environ, "TASK_LOG": str(log), "TASK_FAIL": fail}
    result = subprocess.run(
        ["make", "-j", "4", "all", "VENV=env"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    commands = log.read_text().splitlines()
    assert commands[:3] == ["data download", "data split", "data verify"]
    assert commands[6:9] == ["sweep", "zeroshot", "eval --all"]
    if fail:
        assert result.returncode != 0
        assert commands[-1] == "eval --all"
    else:
        assert result.returncode == 0
        assert commands[9:] == [
            "compare-all",
            "export --run best",
            "bench",
            "figures",
            "report --strict",
        ]


def test_sweep_clock_excludes_cache_preparation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import carvision.train as training

    monkeypatch.setenv("CARVISION_ROOT", str(tmp_path))
    now = 0.0

    def prepare(*args):
        nonlocal now
        now += 100

    def train(config):
        nonlocal now
        now += 3
        run = tmp_path / "artifacts/runs" / f"{config.backbone}-{config.head}-seed{config.seed}"
        run.mkdir(parents=True)
        (run / "metrics.json").write_text("{}")
        return SimpleNamespace(run_dir=run, best_val_top1=0.5)

    monkeypatch.setattr(training.cache, "build", prepare)
    monkeypatch.setattr(training, "train", train)
    monkeypatch.setattr(training, "time", SimpleNamespace(monotonic=lambda: now))
    training.sweep(["tiny"], ["linear"], [0, 1])
    assert json.loads((tmp_path / "artifacts/sweep_timing.json").read_text())["seconds"] == 6


def test_representative_uses_seed_tie_breaker():
    from carvision.utils.artifacts import representative

    candidates = [(0.9, 4), (0.9, 0), (0.9, 3), (0.9, 1), (0.9, 2)]
    assert representative(candidates, key=lambda row: row) == (0.9, 2)


def test_benchmark_clock_excludes_warmup(monkeypatch):
    import time

    import torch

    from carvision.export import benchmark_pytorch

    ticks = iter([1.0, 1.01, 2.0, 2.02])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    calls = []
    model = torch.nn.Identity()
    handle = model.register_forward_hook(lambda *_args: calls.append(1))
    try:
        result = benchmark_pytorch(model, torch.zeros(1, 3), runs=2, warmup=3)
    finally:
        handle.remove()
    assert len(calls) == 5
    assert result.p50_ms == pytest.approx(15)
    assert result.p95_ms == pytest.approx(19.5)


def test_modified_evaluation_settings_are_rejected(tmp_path):
    from carvision.utils.artifacts import validate_evaluation

    run = make_evaluation(tmp_path, "run", ["x", "y"], [0, 1])
    data = json.loads((run / "evaluation.json").read_text())
    data["settings"] = {"resamples": 1}
    (run / "evaluation.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="metadata changed"):
        validate_evaluation(run)
