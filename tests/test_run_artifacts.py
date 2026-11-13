"""SCRUM-59: run ID, metadata, separate latest/best checkpoints and failure records."""

import json
import sys
from unittest.mock import patch

import pytest

from app.checkpoint import sha256_file
from app.dataset.manifest import build_manifest
from app.dataset.rfdetr_export import SplitAssignment, export_rfdetr_dataset
from app.run_artifacts import (
    RunArtifactError, capture_console, finalize_run, new_run_id, select_best, start_run,
)
from app.train_rfdetr import load_config
from tests.checkpoint_fakes import load_fake, write_fake_checkpoint, write_metrics_log
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


@pytest.fixture
def run(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    export_rfdetr_dataset(
        dataset, build_manifest(dataset, "v1"),
        [SplitAssignment("fix-001", "train", "g1"), SplitAssignment("fix-002", "train", "g1"),
         SplitAssignment("fix-003", "valid", "g2")],
        {1: "A", 2: "B"}, tmp_path / "dataset",
    )
    (tmp_path / "initial.pth").write_bytes(b"initial weights")
    config = tmp_path / "train.json"
    config.write_text(json.dumps(dict(
        variant="nano", dataset_dir="dataset", initial_checkpoint="initial.pth",
        device="cpu", seed=7, output_dir="run", training=dict(epochs=3))))
    resolved = load_config(config)
    resolved.update(run_id=new_run_id(), installed_dependencies={"rfdetr": "1.3.0"},
                    train_kwargs={"run_test": False},
                    test_loader_policy="validation-only; frozen test never opened")
    output = tmp_path / "run"
    output.mkdir()
    start_run(resolved)
    return resolved, output


def complete_outputs(output, *, values=(0.1, 0.4, 0.3), ema=None, best_names=("A", "B")):
    write_metrics_log(output / "log.txt", values, ema)
    write_fake_checkpoint(output / "checkpoint.pth", epoch=len(values) - 1)
    write_fake_checkpoint(output / "checkpoint_best_total.pth", class_names=best_names, stripped=True)


def record(output):
    return json.loads((output / "run.json").read_text())


def test_run_record_holds_identity_parameters_dependencies_and_dataset_hashes(run):
    resolved, output = run
    saved = record(output)
    assert saved["run_id"] == resolved["run_id"] and saved["status"] == "running"
    assert saved["seed"] == 7
    assert saved["parameters"]["epochs"] == 3 and saved["parameters"]["lr"] == 0.0001
    assert saved["dependencies"]["pinned"]["rfdetr"] == "1.3.0"
    assert saved["dependencies"]["installed"] == {"rfdetr": "1.3.0"}
    dataset = saved["dataset"]
    assert dataset["dataset_version"] == "v1" and len(dataset["dataset_content_hash"]) == 64
    root = output.parent / "dataset"
    assert dataset["file_sha256"]["train/_annotations.coco.json"] == sha256_file(
        root / "train" / "_annotations.coco.json")
    assert set(dataset["image_set_sha256"]) == {"train", "valid"}
    assert dataset["image_counts"] == {"train": 2, "valid": 1}
    assert dataset["class_map"] == {"1": "A", "2": "B"}
    assert saved["initial_checkpoint"]["sha256"] == sha256_file(output.parent / "initial.pth")
    assert saved["checkpoints"] == {"latest": None, "best": None}


def test_run_ids_are_unique():
    assert len({new_run_id() for _ in range(50)}) == 50


def test_finalize_publishes_separate_verified_latest_and_best(run):
    resolved, output = run
    complete_outputs(output)
    (output / "console.log").write_text("training output\n")
    finalize_run(output, loader=load_fake)
    saved = record(output)
    assert saved["status"] == "completed" and saved["failure"] is None
    latest, best = saved["checkpoints"]["latest"], saved["checkpoints"]["best"]
    assert latest == {"path": "checkpoint.pth", "epoch": 2, "sha256": sha256_file(output / "checkpoint.pth"),
                      "size_bytes": (output / "checkpoint.pth").stat().st_size}
    assert best["path"] == "checkpoint_best_total.pth"
    assert best["sha256"] == sha256_file(output / "checkpoint_best_total.pth")
    assert best["selected_by"]["split"] == "valid"
    assert (best["selected_by"]["epoch"], best["selected_by"]["value"]) == (1, 0.4)
    for name in (latest["path"], best["path"]):
        assert (output / name).is_file()
    assert saved["logs"]["log.txt"]["sha256"] == sha256_file(output / "log.txt")
    assert saved["logs"]["console.log"]["sha256"] == sha256_file(output / "console.log")
    assert saved["train_kwargs"]["run_test"] is False


def test_best_selection_uses_validation_metric_only_and_matches_rfdetr_rules():
    entries = [
        {"epoch": 0, "test_coco_eval_bbox": [0.5], "ema_test_coco_eval_bbox": [0.5]},
        {"epoch": 1, "test_coco_eval_bbox": [0.5], "ema_test_coco_eval_bbox": [0.4],
         "final_test_map": 0.99},
    ]
    chosen = select_best(entries, "log.txt")
    assert (chosen["source"], chosen["epoch"], chosen["value"]) == ("regular", 0, 0.5)
    entries[1]["ema_test_coco_eval_bbox"] = [0.6]
    chosen = select_best(entries, "log.txt")
    assert (chosen["source"], chosen["epoch"], chosen["log_key"]) == ("ema", 1, "ema_test_coco_eval_bbox")


def assert_failed(output, operation, path):
    saved = record(output)
    assert saved["status"] == "failed"
    assert saved["checkpoints"] == {"latest": None, "best": None}
    assert saved["failure"]["operation"] == operation
    assert saved["failure"]["path"] == str(path)


def test_truncated_latest_checkpoint_is_a_recorded_failure(run):
    _, output = run
    complete_outputs(output)
    (output / "checkpoint.pth").write_text('{"model": {"class_embed')  # interrupted write
    with pytest.raises(RunArtifactError, match="verify latest checkpoint"):
        finalize_run(output, loader=load_fake)
    assert_failed(output, "verify latest checkpoint", output / "checkpoint.pth")


def test_latest_checkpoint_must_match_last_completed_epoch(run):
    _, output = run
    complete_outputs(output)
    write_fake_checkpoint(output / "checkpoint.pth", epoch=1)
    with pytest.raises(RunArtifactError, match="epoch 1 does not match last logged epoch 2"):
        finalize_run(output, loader=load_fake)
    assert_failed(output, "verify latest checkpoint", output / "checkpoint.pth")


@pytest.mark.parametrize("change", ["missing", "empty", "wrong classes"])
def test_unusable_best_checkpoint_is_a_recorded_failure(run, change):
    _, output = run
    complete_outputs(output, best_names=("B", "A") if change == "wrong classes" else ("A", "B"))
    best = output / "checkpoint_best_total.pth"
    if change == "missing":
        best.unlink()
    elif change == "empty":
        best.write_bytes(b"")
    with pytest.raises(RunArtifactError, match="verify best checkpoint"):
        finalize_run(output, loader=load_fake)
    assert_failed(output, "verify best checkpoint", best)


def test_log_without_validation_metric_is_a_recorded_failure(run):
    _, output = run
    complete_outputs(output)
    (output / "log.txt").write_text(json.dumps({"epoch": 0, "train_loss": 1.0}) + "\n")
    write_fake_checkpoint(output / "checkpoint.pth", epoch=0)
    with pytest.raises(RunArtifactError, match="no validation mAP"):
        finalize_run(output, loader=load_fake)
    assert_failed(output, "select best checkpoint", output / "log.txt")


def test_metadata_write_failure_names_path_and_reports_no_checkpoints(run):
    _, output = run
    complete_outputs(output)
    with patch("app.run_artifacts.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(RunArtifactError) as raised:
            finalize_run(output, loader=load_fake)
    assert raised.value.operation == "write run record"
    assert raised.value.path == output / "run.json"
    saved = record(output)  # previous complete file survives; no partial JSON
    assert saved["status"] == "running"
    assert saved["checkpoints"] == {"latest": None, "best": None}
    assert not list(output.glob(".run.json-*"))


def test_console_output_is_copied_to_log(tmp_path, capsys):
    with capture_console(tmp_path / "console.log"):
        print("epoch 1")
        print("warning", file=sys.stderr)
    assert "epoch 1" in capsys.readouterr().out
    assert (tmp_path / "console.log").read_text() == "epoch 1\nwarning\n"
