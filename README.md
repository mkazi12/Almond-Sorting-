# Camera Based Almond Detection

A local desktop application for inspecting almond images with RF-DETR.
The application uses PySide6 for the UI and regular files for models, results and
training artifacts. Start it from the project root with Python 3.12.

## Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -r requirements.txt
python -m app
```

On macOS/Linux, activate with `source .venv/bin/activate`.
`requirements.txt` installs the application and dataset tools. The supplied direct dependency pins are retained; see [versions.md](versions.md)
for the selected versions and installation verification limits.

## Desktop workflow

- Open a JPG/PNG or load the included sample image.
- With a configured detector, inspect the rendered boxes, labels and confidence values.
- Save successful analysis as JSON, or export its detections as CSV.
- Open a saved JSON result to restore its image and overlays without running inference.

The current baseline previews images immediately. It shows a model-unconfigured state
until the Sprint 3 RF-DETR checkpoint adapter is connected. Real model training/loading
remains ST-05 work. A preview alone cannot be saved as a successful analysis. A completed
inference with zero detections can be saved and exported.

JSON contains the source-image reference and SHA-256, model label when available, class
map, original image dimensions, prediction confidence floor and detections. Reopening
validates the saved record and the source image before displaying it. Keep the source
image with your saved results; it is referenced rather than embedded. CSV is for analysis
and sharing; reopen results through JSON. See [local_files.md](docs/local_files.md).

## Code layout

| Location | Responsibility |
| --- | --- |
| `app/__main__.py` | Desktop startup |
| `app/ui/` | Image/result dialogs, preview and overlays |
| `app/inference.py` | Validated RGB inputs, detector protocol and prediction records |
| `app/rfdetr_adapter.py` | RF-DETR checkpoint loading and prediction; see [rfdetr-inference.md](docs/rfdetr-inference.md) |
| `app/results.py` | Validated local JSON save/reopen and CSV export |
| `app/image_loader.py` | Qt image loading and actionable errors |
| `app/dataset/` | Headless import, labels, grouping, split manifests, audit and frozen-test guards; separate Qt ground-truth preview |
| `scripts/create_demo_data.py` | Explicit local synthetic image generator |
| `tests/` | Regression checks and authored fixtures |

The detector accepts RGB pixels and returns category IDs, confidence scores and original
pixel `xyxy` boxes. Model preprocessing and output-index/category mapping belong in the
RF-DETR adapter. Keep training, metrics and filesystem operations outside Qt widgets.

## Dataset tools

Use the labeled importer with the approved production class mapping and real provenance.
Read [dataset_import.md](docs/dataset_import.md), [dataset_grouping.md](docs/dataset_grouping.md)
and [sprint_3.md](docs/sprint_3.md) for the checked labeled handoff.

To exercise raw image preparation locally:

```bash
python scripts/create_demo_data.py data/source_data
python -m app.split_dataset data/source_data data/dataset --seed 1
python -m app.audit_dataset data/dataset
```

The unlabeled demo audit reports pending labels and exits nonzero. Artificial group
names do not establish independent real sources. The raw splitter copies images only;
its output is not the labeled RF-DETR COCO exporter.

Keep frozen test images out of fitting, validation, confidence tuning and model selection;
see [frozen_test_set.md](docs/frozen_test_set.md). Final locked testing belongs to Sprint 8.

## Tests and Sprint 3

```bash
python -m pytest -q
```

Desktop tests use Qt's offscreen platform. Install the declared dependencies before
running the complete suite. Source image/result tests and dataset checks run headlessly.
The earlier database services, database tests, migrations and server setup were removed.

See [RF-DETR dataset export](docs/rfdetr-dataset-export.md) for converting the
versioned manifests and approved splits into validated COCO train/valid inputs.

See [Configured RF-DETR training](docs/rfdetr-training.md) for pinned training
dependencies, JSON configuration, validation and the launch command.
