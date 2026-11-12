# Project versions

## Python and platform

- Target Python version: 3.12
- Target platform: Windows
- Python package versions are pinned in `requirements.txt`.

Install dependencies:

    python -m pip install -r requirements.txt

## Model

- Architecture: RF-DETR
- Task: bounding-box object detection
- Model variant: to be selected in Sprint 3
- RF-DETR version: to be selected and pinned
- PyTorch version: to be selected and pinned
- Execution device: to be verified on the target workstation
- Checkpoint: not selected yet
- Production class map: to be confirmed

RF-DETR and PyTorch dependencies will be added when the
training runner and checkpoint adapter are implemented.

## Checkpoint information

When a checkpoint is selected, record:

- Model name and version
- Checkpoint filename
- Source or training run
- SHA-256 checksum
- License
- Class-map filename
- Input resolution and preprocessing

## Updating dependencies

1. Update the relevant pin in `requirements.txt`.
2. Install into a clean Python environment.
3. Run `python -m pytest -q` and check the desktop app.
4. Update the Python or model information above if it changed.