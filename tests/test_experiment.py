import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.dataset.manifest import build_manifest
from app.dataset.rfdetr_export import SplitAssignment, export_rfdetr_dataset
from app.experiment import (
    ExperimentError, launch_experiment, load_experiment, main, save_experiment,
)
from app.train_rfdetr import TrainingConfigError, launch
from tests.checkpoint_fakes import load_fake, write_fake_checkpoint, write_metrics_log
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class FakeRfdetr:
    """Stand-in for the RF-DETR package that writes real run outputs on train()."""

    def __init__(self):
        self.models = []
        self.builder = Mock(return_value="validation dataset")
        self.main = SimpleNamespace(build_dataset=self.builder)

    def factory(self, **kwargs):
        model = Mock()
        model.kwargs = kwargs

        def train(**train_kwargs):
            # The runner must redirect the upstream test loader to validation.
            self.main.build_dataset("test", args="inputs")
            output = Path(train_kwargs["output_dir"])
            write_metrics_log(output / "log.txt", [0.2, 0.4])
            write_fake_checkpoint(output / "checkpoint.pth", epoch=1)
            write_fake_checkpoint(output / "checkpoint_best_total.pth", stripped=True)

        model.train.side_effect = train
        self.models.append(model)
        return model

    def modules(self):
        return {"torch": Mock(), "numpy": Mock(),
                "rfdetr": SimpleNamespace(RFDETRSmall=self.factory), "rfdetr.main": self.main}


def run_with(fake, function, *args, **kwargs):
    with patch("app.train_rfdetr.check_dependencies", return_value={"rfdetr": "1.3.0"}), \
         patch("app.train_rfdetr.importlib.import_module", side_effect=fake.modules().__getitem__):
        return function(*args, checkpoint_loader=load_fake, **kwargs)


def tree_digest(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture
def parent(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    export_rfdetr_dataset(
        dataset, build_manifest(dataset, "v1"),
        [SplitAssignment("fix-001", "train", "g1"),
         SplitAssignment("fix-002", "train", "g1"),
         SplitAssignment("fix-003", "valid", "g2")],
        {1: "A", 2: "B"}, tmp_path / "dataset",
    )
    (tmp_path / "initial.pth").write_bytes(b"pretrained weights stub")
    config = tmp_path / "train.json"
    config.write_text(json.dumps({
        "variant": "small", "dataset_dir": "dataset", "initial_checkpoint": "initial.pth",
        "device": "cpu", "seed": 17, "output_dir": "runs/parent",
        "training": {"epochs": 2, "batch_size": 2, "lr": 0.002, "use_ema": False},
    }))
    resolved = run_with(FakeRfdetr(), launch, config)
    return tmp_path / "runs" / "parent", resolved


def save(tmp_path, parent_dir, **options):
    options.setdefault("output_dir", tmp_path / "runs" / "child")
    options.setdefault("changes", {"lr": 0.0005, "epochs": 4})
    return save_experiment(parent_dir, tmp_path / "experiments" / "lower-lr.json", **options)


def test_saving_preserves_parent_dataset_checkpoint_and_resolved_changes(tmp_path, parent):
    parent_dir, parent_config = parent
    document = save(tmp_path, parent_dir)
    record = json.loads((parent_dir / "run.json").read_text())

    assert document["parent_run"]["run_id"] == parent_config["run_id"] == record["run_id"]
    assert document["parent_run"]["path"] == "../runs/parent"
    assert document["dataset"]["path"] == "../dataset"
    assert document["dataset"]["dataset_version"] == "v1"
    for key in ("dataset_content_hash", "file_sha256", "image_set_sha256", "class_map"):
        assert document["dataset"][key] == record["dataset"][key]
    assert document["initial_checkpoint"] == {
        "source": "best", "path": "../runs/parent/checkpoint_best_total.pth",
        "sha256": record["checkpoints"]["best"]["sha256"],
    }
    assert document["changes"] == {"lr": 0.0005, "epochs": 4}
    assert document["resolved"]["training"] == {**parent_config["training"], "lr": 0.0005, "epochs": 4}
    assert (document["resolved"]["variant"], document["resolved"]["seed"]) == ("small", 17)
    saved = tmp_path / "experiments" / "lower-lr.json"
    assert json.loads(saved.read_text()) == document


def test_reloading_restores_values_through_normal_training_validation(tmp_path, parent):
    parent_dir, parent_config = parent
    save(tmp_path, parent_dir, changes={"lr": 0.0005, "seed": 99, "use_ema": True})
    config = load_experiment(tmp_path / "experiments" / "lower-lr.json")

    assert config["training"] == {**parent_config["training"], "lr": 0.0005, "use_ema": True}
    assert type(config["training"]["epochs"]) is int and type(config["training"]["lr"]) is float
    assert config["seed"] == 99 and config["variant"] == "small"
    assert config["initial_checkpoint"] == str(parent_dir / "checkpoint_best_total.pth")
    assert config["dataset_dir"] == str(tmp_path / "dataset")
    assert config["output_dir"] == str(tmp_path / "runs" / "child")
    assert config["dependencies"] == parent_config["dependencies"]
    assert config["lineage"]["parent_run_id"] == parent_config["run_id"]


@pytest.mark.parametrize("changes,field", [
    ({"lr": "0.001"}, "training.lr"),
    ({"epochs": 0}, "training.epochs"),
    ({"use_ema": 1}, "training.use_ema"),
    ({"device": "auto"}, "device"),
    ({"seed": -1}, "seed"),
])
def test_invalid_changes_fail_normal_training_validation(tmp_path, parent, changes, field):
    parent_dir, _ = parent
    with pytest.raises(TrainingConfigError, match=field):
        save(tmp_path, parent_dir, changes=changes)
    assert not (tmp_path / "experiments" / "lower-lr.json").exists()


def test_hand_edited_parameter_is_still_normally_validated(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    path = tmp_path / "experiments" / "lower-lr.json"
    document = json.loads(path.read_text())
    document["changes"]["batch_size"] = "4"
    document["resolved"]["training"]["batch_size"] = "4"
    path.write_text(json.dumps(document))
    with pytest.raises(TrainingConfigError, match="training.batch_size"):
        load_experiment(path)


@pytest.mark.parametrize("changes,field", [
    ({"variant": "nano"}, "changes.variant"),
    ({"dataset_dir": "elsewhere"}, "changes.dataset_dir"),
    ({"learning_rate": 0.1}, "changes.learning_rate"),
])
def test_unsupported_changes_are_rejected(tmp_path, parent, changes, field):
    parent_dir, _ = parent
    with pytest.raises(ExperimentError, match=field):
        save(tmp_path, parent_dir, changes=changes)


def test_launch_creates_new_linked_run_and_leaves_parent_intact(tmp_path, parent):
    parent_dir, parent_config = parent
    before = tree_digest(parent_dir)
    save(tmp_path, parent_dir)
    fake = FakeRfdetr()
    experiment = tmp_path / "experiments" / "lower-lr.json"
    resolved = run_with(fake, launch_experiment, experiment)

    child = tmp_path / "runs" / "child"
    record = json.loads((child / "run.json").read_text())
    assert record["status"] == "completed"
    assert record["run_id"] == resolved["run_id"] != parent_config["run_id"]
    assert record["lineage"] == {
        "parent_run_id": parent_config["run_id"],
        "parent_run_dir": str(parent_dir),
        "initial_checkpoint_source": "best",
        "initial_checkpoint_sha256": hashlib.sha256(
            (parent_dir / "checkpoint_best_total.pth").read_bytes()).hexdigest(),
        "changes": {"lr": 0.0005, "epochs": 4},
        "splits": ["train", "valid"],
        "experiment_path": str(experiment),
        "experiment_sha256": hashlib.sha256(experiment.read_bytes()).hexdigest(),
    }
    assert record["initial_checkpoint"]["path"] == str(parent_dir / "checkpoint_best_total.pth")
    parent_record = json.loads((parent_dir / "run.json").read_text())
    for key in ("dataset_version", "dataset_content_hash", "file_sha256", "image_set_sha256"):
        assert record["dataset"][key] == parent_record["dataset"][key]
    assert json.loads((child / "resolved_config.json").read_text())["lineage"] == record["lineage"]

    (model,) = fake.models
    assert model.kwargs["pretrain_weights"] == str(parent_dir / "checkpoint_best_total.pth")
    kwargs = model.train.call_args.kwargs
    assert (kwargs["lr"], kwargs["epochs"], kwargs["batch_size"]) == (0.0005, 4, 2)
    assert kwargs["output_dir"] == str(child) and kwargs["run_test"] is False
    assert kwargs["dataset_dir"] == str(tmp_path / "dataset")
    # The upstream test loader was redirected to validation; test is never opened.
    fake.builder.assert_called_once_with(image_set="val", args="inputs")
    assert tree_digest(parent_dir) == before


def test_child_run_can_be_the_parent_of_another_experiment(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    child = run_with(FakeRfdetr(), launch_experiment, tmp_path / "experiments" / "lower-lr.json")
    grandchild = save_experiment(tmp_path / "runs" / "child", tmp_path / "experiments" / "next.json",
                                 output_dir=tmp_path / "runs" / "grandchild", changes={"lr": 0.0001},
                                 checkpoint="latest")
    assert grandchild["parent_run"]["run_id"] == child["run_id"]
    assert grandchild["resolved"]["training"]["epochs"] == 4
    assert grandchild["initial_checkpoint"]["source"] == "latest"


def test_dry_run_validates_without_writing(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    fake = FakeRfdetr()
    resolved = run_with(fake, launch_experiment, tmp_path / "experiments" / "lower-lr.json", dry_run=True)
    assert resolved["train_kwargs"]["lr"] == 0.0005
    assert fake.models == [] and not (tmp_path / "runs" / "child").exists()


def test_missing_parent_run_is_identified(tmp_path):
    with pytest.raises(ExperimentError, match="parent_run: run directory is missing"):
        save(tmp_path, tmp_path / "runs" / "absent")


def test_parent_must_have_completed(tmp_path, parent):
    parent_dir, _ = parent
    record = json.loads((parent_dir / "run.json").read_text())
    record["status"] = "failed"
    (parent_dir / "run.json").write_text(json.dumps(record))
    with pytest.raises(ExperimentError, match="parent_run: .*'failed'"):
        save(tmp_path, parent_dir)


def test_moved_parent_is_identified_on_launch(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    parent_dir.rename(tmp_path / "runs" / "moved")
    with pytest.raises(ExperimentError, match="parent_run: run directory is missing"):
        load_experiment(tmp_path / "experiments" / "lower-lr.json")


def test_different_run_in_parent_location_is_identified(tmp_path, parent):
    parent_dir, _ = parent
    document = save(tmp_path, parent_dir)
    path = tmp_path / "experiments" / "lower-lr.json"
    document["parent_run"]["run_id"] = "20260101T000000Z-00000000"
    path.write_text(json.dumps(document))
    with pytest.raises(ExperimentError, match="parent_run.run_id: expected 20260101T000000Z-00000000"):
        load_experiment(path)


def test_missing_checkpoint_is_identified(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    (parent_dir / "checkpoint_best_total.pth").unlink()
    with pytest.raises(ExperimentError, match="initial_checkpoint: file is missing"):
        load_experiment(tmp_path / "experiments" / "lower-lr.json")


def test_replaced_checkpoint_is_identified(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    write_fake_checkpoint(parent_dir / "checkpoint_best_total.pth", epoch=7)
    with pytest.raises(ExperimentError, match="initial_checkpoint: .* does not match its saved SHA-256"):
        load_experiment(tmp_path / "experiments" / "lower-lr.json")


def test_checkpoint_changed_after_parent_run_cannot_be_saved(tmp_path, parent):
    parent_dir, _ = parent
    write_fake_checkpoint(parent_dir / "checkpoint.pth", epoch=9)
    with pytest.raises(ExperimentError, match="no longer matches the SHA-256 recorded by run"):
        save(tmp_path, parent_dir, checkpoint="latest")


@pytest.mark.parametrize("relative,field", [
    ("valid/_annotations.coco.json", "dataset.file_sha256"),
    ("train/images/two_a.png", "dataset.image_set_sha256"),
])
def test_changed_dataset_hash_is_identified(tmp_path, parent, relative, field):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    target = next((tmp_path / "dataset").glob(relative.replace("images/", "**/")))
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ExperimentError, match=f"{field}: changed since parent run"):
        load_experiment(tmp_path / "experiments" / "lower-lr.json")


def test_dataset_with_test_split_is_rejected(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    (tmp_path / "dataset" / "test").mkdir()
    with pytest.raises(ExperimentError, match="contains a test split"):
        load_experiment(tmp_path / "experiments" / "lower-lr.json")


def test_output_inside_parent_or_existing_is_rejected(tmp_path, parent):
    parent_dir, _ = parent
    with pytest.raises(ExperimentError, match="output_dir: must be outside"):
        save(tmp_path, parent_dir, output_dir=parent_dir / "child")
    with pytest.raises(ExperimentError, match="output_dir: must be outside"):
        save(tmp_path, parent_dir, output_dir=parent_dir)
    (tmp_path / "runs" / "taken").mkdir()
    with pytest.raises(TrainingConfigError, match="output_dir: must be a new directory"):
        save(tmp_path, parent_dir, output_dir=tmp_path / "runs" / "taken")


def test_edited_resolved_values_must_match_changes(tmp_path, parent):
    parent_dir, _ = parent
    document = save(tmp_path, parent_dir)
    path = tmp_path / "experiments" / "lower-lr.json"
    document["resolved"]["training"]["lr"] = 0.9
    path.write_text(json.dumps(document))
    with pytest.raises(ExperimentError, match="resolved: does not equal"):
        load_experiment(path)


def test_existing_experiment_file_is_not_overwritten(tmp_path, parent):
    parent_dir, _ = parent
    save(tmp_path, parent_dir)
    with pytest.raises(ExperimentError, match="experiment: file already exists"):
        save(tmp_path, parent_dir)


def test_command_line_save_and_dry_run(tmp_path, parent, capsys):
    parent_dir, _ = parent
    experiment = tmp_path / "experiments" / "cli.json"
    assert main(["save", str(parent_dir), str(experiment), "--output-dir", str(tmp_path / "runs" / "cli"),
                 "--set", "lr=0.0003", "--set", "use_ema=true", "--set", "device=cpu",
                 "--checkpoint", "initial"]) == 0
    saved = json.loads(capsys.readouterr().out)
    assert saved["changes"] == {"lr": 0.0003, "use_ema": True, "device": "cpu"}
    assert saved["initial_checkpoint"]["source"] == "initial"
    with patch("app.train_rfdetr.check_dependencies", return_value={}), \
         patch("app.train_rfdetr.importlib.import_module", side_effect=FakeRfdetr().modules().__getitem__):
        assert main(["launch", str(experiment), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["training"]["lr"] == 0.0003

    with pytest.raises(SystemExit) as stopped:
        main(["save", str(tmp_path / "missing"), str(tmp_path / "x.json"), "--output-dir", "out"])
    assert stopped.value.code == 2
    assert "parent_run: run directory is missing" in capsys.readouterr().err
