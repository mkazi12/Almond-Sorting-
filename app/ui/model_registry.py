"""Saved-model browser with explicit import, activate and export actions."""

import json

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QDialog, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from app.model_registry import read_registry
from app.model_workflow import ModelError


class ModelRegistryDialog(QDialog):
    def __init__(self, root, parent=None, *, workflow=None, activate=None):
        super().__init__(parent)
        self.root = root
        self.models = ()
        self.workflow = workflow
        self.activate = activate
        self.setWindowTitle("Saved models")
        self.resize(1000, 650)
        layout = QVBoxLayout(self)
        location = QLabel(f"Registry: {root}\nSelect a row to inspect it. Browsing does not activate a model.")
        location.setWordWrap(True)
        layout.addWidget(location)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.active_label = QLabel()
        layout.addWidget(self.active_label)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Model ID", "Variant", "Model version", "Class map", "Dataset version",
            "Checkpoint", "Validation summary",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.show_details)
        layout.addWidget(self.table, 2)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        layout.addWidget(self.details, 1)
        self.errors = QPlainTextEdit()
        self.errors.setReadOnly(True)
        layout.addWidget(self.errors)
        buttons = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh)
        buttons.addWidget(self.refresh_button)
        self.import_button = QPushButton("Import package")
        self.import_button.clicked.connect(self.import_package)
        self.import_button.setEnabled(workflow is not None)
        buttons.addWidget(self.import_button)
        self.activate_button = QPushButton("Use selected model")
        self.activate_button.clicked.connect(self.activate_selected)
        buttons.addWidget(self.activate_button)
        self.export_button = QPushButton("Export selected model")
        self.export_button.clicked.connect(self.export_selected)
        buttons.addWidget(self.export_button)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.refresh()

    def refresh(self):
        selected = self.table.currentRow()
        model_id = self.models[selected].model_id if 0 <= selected < len(self.models) else None
        snapshot = read_registry(self.root)
        self.table.clearSelection()
        self.table.setRowCount(0)
        self.details.clear()
        self.models = snapshot.models
        self.table.setRowCount(len(self.models))
        for row, model in enumerate(self.models):
            values = [
                model.model_id, model.variant, model.model_version,
                json.dumps(model.class_map, ensure_ascii=False), model.dataset_version,
                "Available" if model.checkpoint_available else "Missing checkpoint file",
                json.dumps(model.validation_summary, ensure_ascii=False)
                if model.validation_summary is not None else "Validation unavailable",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, model.model_id)
                item.setToolTip(str(model.checkpoint) if column == 5 else value)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self.status.setText(
            f"{len(self.models)} saved model(s)." if self.models else
            "No readable saved models. See registry errors below." if snapshot.errors else
            "No saved models yet."
        )
        self.errors.setPlainText("\n".join(snapshot.errors))
        self.errors.setVisible(bool(snapshot.errors))
        active = self.workflow.active if self.workflow else None
        self.active_label.setText(
            f"Active model: {active.model_id} / {active.model_version}" if active else "No registry model active."
        )
        self._update_actions()
        for row, model in enumerate(self.models):
            if model.model_id == model_id:
                self.table.selectRow(row)
                break

    def show_details(self):
        self._update_actions()
        row = self.table.currentRow()
        if not self.table.selectedItems() or not 0 <= row < len(self.models):
            self.details.clear()
            return
        model = self.models[row]
        self.details.setPlainText(
            f"Model ID: {model.model_id}\nMetadata: {model.metadata}\n"
            f"Checkpoint: {model.checkpoint}\n\nClass map:\n"
            + json.dumps(model.class_map, indent=2, ensure_ascii=False)
            + "\n\nValidation summary:\n"
            + (json.dumps(model.validation_summary, indent=2, ensure_ascii=False)
               if model.validation_summary is not None else "Validation unavailable")
            + "\n\nTraining provenance:\n"
            + (json.dumps(model.training_provenance, indent=2, ensure_ascii=False)
               if model.training_provenance else "Training provenance unavailable")
        )

    def _selected_model(self):
        row = self.table.currentRow()
        return self.models[row] if self.table.selectedItems() and 0 <= row < len(self.models) else None

    def _update_actions(self):
        selected = self._selected_model()
        available = selected is not None and selected.checkpoint_available
        self.activate_button.setEnabled(available and self.activate is not None)
        self.export_button.setEnabled(available and self.workflow is not None)

    def import_package(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import model package", "", "Model package (*.zip)")
        if not path:
            return
        try:
            model = self.workflow.import_package(path)
            self.refresh()
            self.status.setText(f"Imported {model.model_id}. Select it and use 'Use selected model' to activate it.")
        except ModelError as error:
            self.status.setText(str(error))

    def activate_selected(self):
        model = self._selected_model()
        if model is None:
            return
        try:
            self.activate(model.model_id)
            self.refresh()
            self.status.setText(f"Active model saved: {model.model_id} / {model.model_version}")
        except ModelError as error:
            self.status.setText(str(error))

    def export_selected(self):
        model = self._selected_model()
        if model is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export model package (new filename)", model.model_id + ".zip", "Model package (*.zip)",
        )
        if not path:
            return
        try:
            destination = self.workflow.export_package(model.model_id, path)
            self.status.setText(f"Exported {model.model_id}: {destination}")
        except ModelError as error:
            self.status.setText(str(error))
