# Saved model workflow (Sprint 4)

Open **Saved models** in the desktop window. Selecting a row only inspects it;
refresh and reopening read disk without changing the active model.

- **Import package** accepts a ZIP with `model.json` and its referenced checkpoint.
  It validates metadata, SHA-256, tensor shapes and a real forward pass before
  publishing the model. Import does not activate it. Reimporting an existing model
  ID reports that it already exists; it never replaces saved weights or metadata.
- **Use selected model** loads a separate candidate and saves the active selection
  atomically before switching the detector. A load or persistence failure retains
  the previous detector and saved selection. The active label identifies the model.
- **Export selected model** creates a new ZIP with exact metadata bytes and
  checkpoint. Choose a new filename outside the registry; existing destinations
  are not overwritten. The exported checkpoint hash is checked before publication.

Startup restores `models/active_model.json`, checking the recorded metadata and
checkpoint hashes. If the saved model is missing or changed, the window displays
an error and remains in preview mode. No other model is silently substituted.
Each analysis captures its detector, class map, confidence and model identity at
its start; replacing the active model cannot relabel an in-flight result. Existing
saved analyses keep the model label and class map used to produce them.

## Runtime

Use Python 3.12 and install `requirements-models.txt` in the desktop environment:

```powershell
python -m pip install -r requirements-models.txt
python -m app
```

This combines the pinned training runtime with Qt and Pillow. Use it **instead of**
`requirements.txt`, whose NumPy 2.x pin conflicts with the RF-DETR NumPy 1.x pin.
The loader supports RF-DETR 1.3.0 / torch 2.8.0, default nano/small/medium/base/large
object-detection architectures and CPU execution. Custom architectures and
segmentation models are rejected. Loading and prediction currently run
synchronously in the desktop, so large models may take time to open.

The adapter uses restricted `torch.load(weights_only=True)` with the passive
`argparse.Namespace` container allowed for training metadata, then a strict tensor
state-dictionary load. It does not use RF-DETR's permissive pretrained loader or
download replacement weights. Implementation was checked against the
[pinned RF-DETR source](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/detr.py).

## Registry and package contract

The default registry is `models/` at the project root, independent of the launch
directory. Each saved model occupies `models/<model_id>/`, containing `model.json`
and its checkpoint. Model IDs use 1-100 letters, digits, underscores or hyphens,
starting with a letter or digit. They must match their directory names.

The package ZIP contains exactly two files: `model.json` at the ZIP root, and the
checkpoint at the relative path named in that record. Nested checkpoint paths use
forward slashes and cannot escape the model directory. Metadata is limited to
1 MiB and checkpoints to 4 GiB. Import stages files in a hidden temporary directory
and publishes the complete model only after validation succeeds.

Example metadata (replace illustrative labels, provenance, metrics and hash
with actual training records):

```json
{
  "schema_version": 1,
  "format": "rfdetr-state-dict-v1",
  "rfdetr_version": "1.3.0",
  "model_id": "almond-nano-001",
  "variant": "nano",
  "model_version": "v1",
  "class_map": {"1": "Almond", "2": "Broken almond"},
  "output_class_map": [null, 1, 2],
  "dataset_version": "dataset-v3",
  "training_provenance": {"run_id": "training-001", "seed": 42},
  "checkpoint": "checkpoint_best.pth",
  "checkpoint_sha256": "<64 lowercase hex characters from the checkpoint SHA-256>",
  "validation_summary": {"mAP_50_95": 0.61, "mAP_50": 0.84}
}
```

The checkpoint must contain a `model` tensor state dictionary. The full classifier
output count must equal the length of `output_class_map`; each element maps that
output index to an approved class ID or `null` for an unused/background output.
Every ID in `class_map` must appear exactly once. Never infer the output mapping
from the order of a JSON class map. Record the actual mapping used during training.
For the exporter's COCO IDs 1 and 2 with a three-output head, use `[null, 1, 2]`.

Class IDs are canonical nonnegative integer strings in `class_map`. Keep the real
training dataset version and provenance. Validation must be a nonempty JSON object
or explicit `null` when unavailable; import rejects malformed results. Additional
JSON metadata is preserved verbatim by export and import, including training
configuration and validation dataset identifiers if recorded.

Legacy listing-only records remain visible. Missing checkpoints display **Missing
checkpoint file**, and absent or invalid validation displays **Validation
unavailable**. Malformed records are reported without hiding valid entries. The
details pane shows metadata/checkpoint paths, classes, validation and provenance.
Legacy records need the package compatibility fields above before activation or
export; the application never guesses missing model metadata.

Training does not yet automatically publish model records. A training-artifact
producer must supply this record and checkpoint together (or the two-file ZIP).
This workflow does not fabricate production records or validation metrics.

## Verification

`python -m pytest -q` covers import rejection, duplicate IDs, round-trip metadata
and hashes, failed-load rollback, failed-persistence rollback, restart restoration,
and inference identity during replacement. These tests inject small test detectors.

For the real pinned CPU runtime and a local official nano checkpoint:

```powershell
python -m tests.run_model_workflow_integration PATH_TO_NANO_CHECKPOINT
```

The opt-in test runs real loading/forward passes, import, activation, restart and
export/reimport. It uses an explicitly synthetic label/provenance and makes no
model-quality claims.
