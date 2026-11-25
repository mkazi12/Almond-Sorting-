"""Transactional model transfer, activation and desktop recovery contracts."""

import json
from pathlib import Path
from unittest.mock import Mock, patch
import zipfile

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from app.inference import Detection, InferenceResult, RGBImage
from app.model_registry import read_registry
from app.model_workflow import ModelError, ModelWorkflow, sha256
from app.rfdetr_detector import RFDETRDetector
from app.ui.main_window import MainWindow


def write_model(root, model_id="a", **changes):
    folder = root / model_id
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "weights.pth"
    checkpoint.write_bytes(b"test weights " + model_id.encode())
    record = {
        "schema_version": 1, "model_id": model_id, "variant": "nano",
        "model_version": "v1", "class_map": {"17": "Almond"},
        "dataset_version": "dataset-v4", "checkpoint": "weights.pth",
        "checkpoint_sha256": sha256(checkpoint), "format": "rfdetr-state-dict-v1",
        "rfdetr_version": "1.3.0", "output_class_map": [None, 17],
        "training_provenance": {"seed": 42, "run": "run-123", "config": {"epochs": 3}},
        "validation_summary": {"mAP": 0.63, "dataset_version": "validation-v4"},
        **changes,
    }
    (folder / "model.json").write_text(json.dumps(record), encoding="utf-8")
    return folder


def make_package(folder, path, *, omit=None, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("model.json", "weights.pth"):
            if name != omit:
                archive.write(folder / name, name)
        if extra:
            archive.writestr(extra, "bad")
    return path


def workflow(root):
    return ModelWorkflow(root, loader=Mock(side_effect=lambda model, record: Mock()))


def test_import_once_and_export_preserve_checkpoint_and_all_metadata(tmp_path):
    folder = write_model(tmp_path / "source")
    original = make_package(folder, tmp_path / "original.zip")
    service = workflow(tmp_path / "registry")
    imported = service.import_package(original)
    assert service.active is None
    assert imported.model_id == "a"
    assert imported.checkpoint.read_bytes() == (folder / "weights.pth").read_bytes()
    assert imported.metadata.read_bytes() == (folder / "model.json").read_bytes()
    with pytest.raises(ModelError, match="already exists"):
        service.import_package(original)
    assert len(read_registry(service.root).models) == 1
    assert service.loader.call_count == 1
    exported = service.export_package("a", tmp_path / "export.zip")
    second = workflow(tmp_path / "another-registry").import_package(exported)
    assert sha256(second.checkpoint) == sha256(imported.checkpoint)
    assert second.class_map == imported.class_map
    assert second.metadata.read_bytes() == imported.metadata.read_bytes()
    with pytest.raises(ModelError, match="already exists"):
        service.export_package("a", exported)


@pytest.mark.parametrize(("changes", "omit", "extra", "message"), [
    ({}, "model.json", None, "missing model.json"),
    ({}, "weights.pth", None, "missing checkpoint"),
    ({"format": "unknown"}, None, None, "Incompatible format"),
    ({"variant": "unsupported"}, None, None, "Unsupported RF-DETR variant"),
    ({"checkpoint_sha256": "0" * 64}, None, None, "SHA-256 mismatch"),
    ({"class_map": {}}, None, None, "class_map"),
    ({"output_class_map": [None, 99]}, None, None, "output_class_map"),
    ({"training_provenance": None}, None, None, "training_provenance"),
    ({"validation_summary": {"mAP": float("nan")}}, None, None, "validation_summary"),
    ({"checkpoint": "../weights.pth"}, None, None, "Invalid checkpoint path"),
    ({"checkpoint": "C:/weights.pth"}, None, None, "Invalid checkpoint path"),
    ({}, None, "../escape", "Unexpected package members"),
])
def test_invalid_import_preserves_registry_and_active(tmp_path, changes, omit, extra, message):
    service = workflow(tmp_path / "registry")
    write_model(service.root, "working")
    previous = service.activate("working")
    selected = (service.root / "active_model.json").read_bytes()
    folder = write_model(tmp_path / "source", **changes)
    package = make_package(folder, tmp_path / "package.zip", omit=omit, extra=extra)
    with pytest.raises(ModelError, match=message):
        service.import_package(package)
    assert service.active is previous
    assert (service.root / "active_model.json").read_bytes() == selected
    assert [model.model_id for model in read_registry(service.root).models] == ["working"]
    assert not list(service.root.glob(".import-*"))


def test_corrupt_zip_and_load_failure_do_not_publish(tmp_path):
    service = workflow(tmp_path / "registry")
    package = tmp_path / "broken.zip"
    package.write_bytes(b"not zip")
    with pytest.raises(ModelError, match="not a zip"):
        service.import_package(package)
    folder = write_model(tmp_path / "source")
    make_package(folder, package)
    service.loader.side_effect = RuntimeError("tensor shape mismatch")
    with pytest.raises(ModelError, match="tensor shape mismatch"):
        service.import_package(package)
    assert not read_registry(service.root).models
    assert not read_registry(service.root).errors


def test_selection_restart_and_failure_keep_last_working_model(tmp_path):
    service = workflow(tmp_path)
    write_model(tmp_path, "a")
    folder = write_model(tmp_path, "b", model_version="v2")
    first = service.activate("a")
    saved = (tmp_path / "active_model.json").read_bytes()
    service.loader.side_effect = RuntimeError("cannot load b")
    with pytest.raises(ModelError, match="cannot load b"):
        service.activate("b")
    assert service.active is first
    assert (tmp_path / "active_model.json").read_bytes() == saved
    service.loader.side_effect = lambda model, record: Mock()
    with patch("app.model_workflow.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(ModelError, match="disk full"):
            service.activate("b")
    assert service.active is first
    assert (tmp_path / "active_model.json").read_bytes() == saved
    assert not list(tmp_path.glob(".selection-*"))
    second = service.activate("b")
    restarted = workflow(tmp_path)
    restored = restarted.restore()
    assert restored.model_id == "b"
    assert restored.model_version == "v2"
    assert restored.checkpoint_sha256 == second.checkpoint_sha256
    assert dict(restored.class_map) == {17: "Almond"}
    (folder / "weights.pth").write_bytes(b"changed")
    with pytest.raises(ModelError, match="SHA-256 mismatch"):
        restarted.restore()
    assert restarted.active is restored


def test_restore_rejects_changed_metadata(tmp_path):
    service = workflow(tmp_path)
    folder = write_model(tmp_path)
    service.activate("a")
    record = json.loads((folder / "model.json").read_text())
    record["class_map"] = {"17": "Different label"}
    (folder / "model.json").write_text(json.dumps(record))
    restarted = workflow(tmp_path)
    with pytest.raises(ModelError, match="metadata changed"):
        restarted.restore()
    assert restarted.active is None


def test_failed_export_does_not_publish_partial_package(tmp_path):
    service = workflow(tmp_path / "registry")
    write_model(service.root)
    with patch("app.model_workflow.os.link", side_effect=OSError("disk full")):
        with pytest.raises(ModelError, match="disk full"):
            service.export_package("a", tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()
    assert not list(tmp_path.glob(".export-*"))
    with pytest.raises(ModelError, match="outside the registry"):
        service.export_package("a", service.root / "a" / "weights.pth")


def test_desktop_import_activate_export_and_restart(tmp_path, qt_application):
    folder = write_model(tmp_path / "source")
    package = make_package(folder, tmp_path / "package.zip")
    service = workflow(tmp_path / "registry")
    window = MainWindow(model_workflow=service)
    restarted = None
    try:
        window.open_model_registry()
        dialog = window.registry_dialog
        assert not dialog.activate_button.isEnabled()
        with patch("app.ui.model_registry.QFileDialog.getOpenFileName", return_value=(str(package), "")):
            dialog.import_button.click()
        assert dialog.table.rowCount() == 1
        assert window.detector is None
        dialog.table.selectRow(0)
        dialog.activate_button.click()
        assert window.detector is service.active.detector
        assert window.class_names == {17: "Almond"}
        output = tmp_path / "export.zip"
        with patch("app.ui.model_registry.QFileDialog.getSaveFileName", return_value=(str(output), "")):
            dialog.export_button.click()
        assert output.is_file()
        restarted = MainWindow(model_workflow=workflow(service.root))
        assert restarted.model_workflow.active.model_id == "a"
        assert restarted.model_label == window.model_label
        previous = window.detector
        service.loader.side_effect = RuntimeError("bad checkpoint")
        dialog.activate_button.click()
        assert "bad checkpoint" in dialog.status.text()
        assert window.detector is previous
    finally:
        window.close()
        if restarted:
            restarted.close()


def test_inflight_prediction_uses_original_model_and_classes(tmp_path, qt_application):
    service = workflow(tmp_path / "registry")
    write_model(service.root, "a")
    write_model(service.root, "b", class_map={"8": "New label"}, output_class_map=[None, 8])
    window = MainWindow(model_workflow=service)
    try:
        window.activate_model("a")
        original_label = window.model_label
        original = window.detector
        image = QImage(20, 20, QImage.Format.Format_RGB32)
        image.fill(0)
        path = tmp_path / "image.png"
        image.save(str(path))

        def predict(pixels, *, confidence_floor):
            window.activate_model("b")
            return InferenceResult(20, 20, confidence_floor, (Detection(17, .9, (1, 1, 10, 10)),))

        original.predict.side_effect = predict
        window.display_image_path(path)
        assert service.active.model_id == "b"
        assert window.current_analysis.model_label == original_label
        assert dict(window.current_analysis.class_names) == {17: "Almond"}
        assert window.current_analysis.result.detections[0].category_id == 17
        window.detector.predict.assert_not_called()
    finally:
        window.close()


def test_adapter_maps_outputs_and_preserves_original_dimensions():
    backend = Mock()
    backend.predict.return_value = Mock(
        class_id=[0, 1], confidence=[.9, .8], xyxy=[(1, 1, 3, 3), (-2, -1, 30, 40)],
    )
    detector = RFDETRDetector(backend, [None, 17], {17: "Almond"})
    result = detector.predict(RGBImage(20, 10, bytes(600)), confidence_floor=.5)
    assert result.detections == (Detection(17, .8, (0, 0, 20, 10)),)
    assert (result.image_width, result.image_height) == (20, 10)
    backend.predict.assert_called_once()
