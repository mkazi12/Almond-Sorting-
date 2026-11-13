# Configured RF-DETR training

Create a separate Python 3.11 or 3.12 training environment; the application's
existing environment does not require PyTorch. Install the pinned training
stack:

```powershell
py -3.11 -m venv .venv-training
.\.venv-training\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv-training\Scripts\python.exe -m pip check
```

The six direct training dependencies are pinned and checked at launch.
Transitive dependencies are resolved by pip; this is not a complete platform
lock file. Exact installed direct versions, including PyTorch wheel build
suffixes, are included in the resolved configuration. Use the same Python
and wheel platform when reconstructing a run.

Edit `configs/training.example.json`. Paths are relative to the configuration
file, independent of the shell working directory. Supply an existing local
initial checkpoint, the dataset produced by the RF-DETR exporter, and a new
output directory. Variants: `nano`, `small`, `medium`, `base`, `large`.
Devices: `cpu`, `cuda`, `mps`; unavailable accelerators cause an error rather
than silently switching to CPU. Initial weights are used for fine-tuning;
this command does not resume optimizer/epoch state.

```powershell
.\.venv-training\Scripts\python.exe -m app.train_rfdetr configs/training.example.json --dry-run
.\.venv-training\Scripts\python.exe -m app.train_rfdetr configs/training.example.json
```

Dry-run checks settings, source paths, direct dependency pins and device
availability without constructing the model. Actual launch loads the
checkpoint; a corrupt or incompatible checkpoint can still fail then.
`training` accepts the explicit fields shown in `app/train_rfdetr.py:DEFAULTS`.
Unknown fields, numeric strings, boolean integers and invalid numeric ranges
are rejected. Defaults are expanded in `resolved_config.json` before fitting.

Each real launch records a run ID, metadata, separate latest and best
checkpoints and logs in `output_dir/run.json`; see
[run-artifacts.md](run-artifacts.md).

`launch(config_path, artifact_writer=callback)` supplies that same resolved
dictionary to a later run-artifact writer. It includes constructor arguments,
training arguments, seed, absolute paths, safety settings and dependency pins.
RF-DETR's remaining internal defaults are defined by its pinned release.
The seed is set before model construction and passed to RF-DETR; bitwise
identity across hardware/backends is not guaranteed.

The compatibility code targets RF-DETR 1.3.0's public classes and `train()`
interface. Its upstream trainer constructs a test loader unconditionally,
so a scoped shim redirects that loader to validation while `run_test=False`.
It never reads the frozen test directory. Run this launcher in a dedicated
process; the scoped upstream function patch is not intended for concurrent
training threads. Train/valid data must be approved through the exporter,
including the frozen-lock guard, before launch.

Adapter tests use a fake model and device backend, with real exported image
fixtures. They verify dispatch and failures without downloading weights or
running an expensive training job:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_train_rfdetr.py tests/test_rfdetr_export.py -q
```

Upstream API references:
[model and training implementation](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/detr.py),
[configuration](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/config.py),
[trainer](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/main.py).
