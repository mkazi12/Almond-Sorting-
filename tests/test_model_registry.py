"""Persisted identity, failure states and read-only desktop browsing."""

import json
from unittest.mock import Mock

import pytest
from PySide6.QtWidgets import QApplication

from app.model_registry import read_registry
from app.ui.main_window import MainWindow
from app.ui.model_registry import ModelRegistryDialog


def save_model(root, directory_name="first", **changes):
    folder = root / directory_name
    folder.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1, "model_id": directory_name, "variant": "nano",
        "model_version": "v1", "class_map": {"17": "Almond"},
        "dataset_version": "dataset-1", "checkpoint": "weights.pth",
        "validation_summary": {"mAP": 0.51},
        **changes,
    }
    (folder / "model.json").write_text(json.dumps(record), encoding="utf-8")
    return folder


def test_records_resolve_matching_metadata_and_checkpoint(tmp_path):
    first = save_model(tmp_path)
    (first / "weights.pth").write_bytes(b"first")
    second = save_model(tmp_path, "second", variant="small", model_version="v2",
                        class_map={"3": "Broken"}, dataset_version="dataset-2",
                        validation_summary={"mAP": 0.72})
    snapshot = read_registry(tmp_path)
    assert not snapshot.errors
    a, b = snapshot.models
    assert a.checkpoint == first / "weights.pth"
    assert a.metadata == first / "model.json"
    assert a.checkpoint_available
    assert b.metadata == second / "model.json"
    assert (b.variant, b.model_version, b.class_map, b.dataset_version) == (
        "small", "v2", {3: "Broken"}, "dataset-2")
    assert b.validation_summary == {"mAP": 0.72}
    assert not b.checkpoint_available


@pytest.mark.parametrize("summary", [None, {}, [], "missing", {"mAP": float("nan")}])
def test_unavailable_validation_never_falls_back(tmp_path, summary):
    save_model(tmp_path)
    save_model(tmp_path, "second", validation_summary=summary)
    assert read_registry(tmp_path).models[1].validation_summary is None


@pytest.mark.parametrize("changes", [
    {"model_id": "other"}, {"checkpoint": "../first/weights.pth"},
    {"schema_version": 2}, {"class_map": {"01": "Almond"}},
    {"class_map": {}}, {"dataset_version": ""},
])
def test_bad_record_is_reported_without_hiding_good_record(tmp_path, changes):
    save_model(tmp_path)
    save_model(tmp_path, "second", **changes)
    snapshot = read_registry(tmp_path)
    assert [model.model_id for model in snapshot.models] == ["first"]
    assert len(snapshot.errors) == 1
    assert snapshot.errors[0].startswith("second:")


def test_empty_corrupt_and_unreadable_registry(tmp_path):
    assert read_registry(tmp_path / "absent").models == ()
    folder = save_model(tmp_path)
    (folder / "model.json").write_text("{")
    assert read_registry(tmp_path).errors
    assert read_registry(folder / "model.json").errors


def test_view_refresh_reopen_and_selection_preserve_active_model(tmp_path, qt_application):
    detector = Mock()
    window = MainWindow(detector=detector, class_names={17: "Active"},
                        model_label="active-version", registry_root=tmp_path)
    try:
        window.open_model_registry()
        dialog = window.registry_dialog
        assert dialog.status.text() == "No saved models yet."
        first = save_model(tmp_path)
        dialog.refresh_button.click()
        assert dialog.table.rowCount() == 1
        assert dialog.table.item(0, 5).text() == "Missing checkpoint file"
        dialog.table.selectRow(0)
        assert str(first / "model.json") in dialog.details.toPlainText()
        save_model(tmp_path, "second", variant="small", model_version="v2",
                   class_map={"3": "Broken"}, dataset_version="dataset-2",
                   validation_summary=None)
        dialog.close()
        window.open_model_registry()
        assert dialog.table.rowCount() == 2
        assert [dialog.table.item(1, col).text() for col in (0, 1, 2, 4, 6)] == [
            "second", "small", "v2", "dataset-2", "Validation unavailable"]
        dialog.table.selectRow(1)
        assert "0.51" not in dialog.details.toPlainText()
        assert '"3": "Broken"' in dialog.details.toPlainText()
        assert window.detector is detector
        assert window.class_names == {17: "Active"}
        assert window.model_label == "active-version"
        assert not detector.mock_calls
        (first / "weights.pth").write_bytes(b"checkpoint")
        dialog.refresh()
        assert dialog.table.item(0, 5).text() == "Available"
        (tmp_path / "second" / "model.json").write_text("invalid")
        dialog.refresh()
        assert dialog.table.rowCount() == 1
        assert dialog.details.toPlainText() == ""
        assert "second:" in dialog.errors.toPlainText()
    finally:
        window.close()


def test_new_dialog_reads_disk_again(tmp_path, qt_application):
    save_model(tmp_path)
    first = ModelRegistryDialog(tmp_path)
    first.close()
    save_model(tmp_path, model_version="updated")
    second = ModelRegistryDialog(tmp_path)
    try:
        assert second.table.item(0, 2).text() == "updated"
    finally:
        second.close()
