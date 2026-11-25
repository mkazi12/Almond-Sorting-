# Saved model registry (SCRUM-95)

Open **Saved models** to inspect persisted model records. Selecting a row,
refreshing, and reopening the view do not change the active detector.

Each model has a `models/<model_id>/model.json` record and a checkpoint within
that directory. The model ID must match the directory name. The view displays
variant, model version, class map, dataset version and validation summary from
that record, with the exact metadata and checkpoint paths in the details pane.

Records use schema_version 1 and fields model_id, variant, model_version,
class_map (canonical nonnegative integer string keys), dataset_version,
checkpoint (relative path), and validation_summary (a nonempty object or null).
Optional training_provenance appears in the details pane.

An empty registry displays **No saved models yet.** Missing checkpoint files
and unavailable validation results are explicitly identified. Invalid records
are reported without hiding valid entries or borrowing another model's data.
Refresh and reopening reread the saved records.

Run `python -m pytest -q tests/test_model_registry.py` to verify record identity,
metadata display, empty/error states, and preservation of the active model.
