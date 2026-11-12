import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.train_rfdetr import TrainingConfigError, check_dependencies, launch, load_config
from app.dataset.manifest import build_manifest
from app.dataset.rfdetr_export import SplitAssignment, export_rfdetr_dataset
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


@pytest.fixture
def config(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    export_rfdetr_dataset(
        dataset, build_manifest(dataset, "v1"),
        [SplitAssignment("fix-001", "train", "g1"),
         SplitAssignment("fix-002", "train", "g1"),
         SplitAssignment("fix-003", "valid", "g2")],
        {1: "A", 2: "B"}, tmp_path / "dataset",
    )
    (tmp_path / "initial.pth").write_bytes(b"checkpoint stub for adapter test")
    raw = dict(variant="small", dataset_dir="dataset", initial_checkpoint="initial.pth",
               device="cpu", seed=17, output_dir="run",
               training=dict(epochs=3, batch_size=2, lr=0.002, use_ema=False))
    path = tmp_path / "train.json"
    path.write_text(json.dumps(raw))
    return path, raw


@pytest.mark.parametrize("field,value", [
    ("variant", "unknown"), ("device", "auto"), ("seed", True),
    ("seed", -1), ("initial_checkpoint", "absent.pth"),
])
def test_invalid_setting_fails_before_constructing_model(config, field, value):
    path, raw = config
    raw[field] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(TrainingConfigError, match=field):
        load_config(path)


@pytest.mark.parametrize("field,value", [
    ("epochs", 0), ("batch_size", "4"), ("lr", -0.1), ("lr", float("nan")),
    ("num_workers", -1), ("use_ema", "false"), ("unknown", 3),
])
def test_invalid_training_parameter_identifies_field(config, field, value):
    path, raw = config
    raw["training"][field] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(TrainingConfigError, match=f"training.{field}"):
        load_config(path)


def test_launch_dispatches_typed_parameters_and_exposes_resolved_config(config):
    path, _ = config
    model = Mock()
    factory = Mock(return_value=model)
    original_builder = Mock(return_value="validation dataset")
    upstream = SimpleNamespace(build_dataset=original_builder)
    torch = Mock()
    modules = {"torch": torch, "numpy": Mock(), "rfdetr": SimpleNamespace(RFDETRSmall=factory),
               "rfdetr.main": upstream}
    writer = Mock()

    def train(**kwargs):
        assert upstream.build_dataset("test", args="inputs", resolution=512) == "validation dataset"
        original_builder.assert_called_once_with(image_set="val", args="inputs", resolution=512)

    model.train.side_effect = train
    with patch("app.train_rfdetr.check_dependencies", return_value={"rfdetr": "1.3.0"}), \
         patch("app.train_rfdetr.importlib.import_module", side_effect=modules.__getitem__):
        resolved = launch(path, artifact_writer=writer)
    factory.assert_called_once_with(pretrain_weights=str(path.parent / "initial.pth"), device="cpu")
    kwargs = model.train.call_args.kwargs
    assert kwargs["epochs"] == 3 and type(kwargs["epochs"]) is int
    assert kwargs["lr"] == 0.002 and type(kwargs["lr"]) is float
    assert kwargs["batch_size"] == 2
    assert kwargs["seed"] == 17
    assert kwargs["run_test"] is False
    assert kwargs["dataset_dir"] == str(path.parent / "dataset")
    writer.assert_called_once_with(resolved)
    assert json.loads((path.parent / "run" / "resolved_config.json").read_text()) == resolved
    assert upstream.build_dataset is original_builder
    torch.manual_seed.assert_called_once_with(17)


def test_unavailable_cuda_stops_before_model_import(config):
    path, raw = config
    raw["device"] = "cuda"
    path.write_text(json.dumps(raw))
    torch = Mock()
    torch.cuda.is_available.return_value = False
    with patch("app.train_rfdetr.check_dependencies", return_value={}), \
         patch("app.train_rfdetr.importlib.import_module", return_value=torch) as importer:
        with pytest.raises(TrainingConfigError, match="device: CUDA"):
            launch(path)
    importer.assert_called_once_with("torch")
    assert not (path.parent / "run").exists()


def test_dependency_version_mismatch_is_rejected():
    with patch("app.train_rfdetr.version", return_value="0.0.0"):
        with pytest.raises(TrainingConfigError, match="dependencies.rfdetr"):
            check_dependencies()


def test_dry_run_never_constructs_model_or_writes(config):
    path, _ = config
    with patch("app.train_rfdetr.check_dependencies", return_value={}), \
         patch("app.train_rfdetr.importlib.import_module", return_value=Mock()) as importer:
        resolved = launch(path, dry_run=True)
    importer.assert_called_once_with("torch")
    assert resolved["train_kwargs"]["run_test"] is False
    assert not (path.parent / "run").exists()
