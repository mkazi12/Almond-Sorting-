# Training run artifacts

Each real launch of `python -m app.train_rfdetr CONFIG.json` gets one run ID
(`YYYYMMDDTHHMMSSZ-<8 hex>`) and keeps everything for that run in its
`output_dir`:

```text
runs/almond-001/
  run.json                    run ID, status, metadata and checkpoint references
  resolved_config.json        resolved parameters (includes run_id)
  checkpoint.pth              latest completed save (RF-DETR, every epoch)
  checkpoint_best_total.pth   best by validation mAP@[.50:.95]
  log.txt                     per-epoch metrics written by RF-DETR
  console.log                 copy of stdout/stderr during fitting
```

## run.json

| Field | Contents |
| --- | --- |
| `run_id`, `status` | `running` -> `completed` or `failed` |
| `parameters`, `seed`, `variant`, `device` | Resolved training settings |
| `model_kwargs`, `train_kwargs` | Exact arguments passed to RF-DETR |
| `dependencies` | Pinned and installed versions; Python and platform in `environment` |
| `initial_checkpoint` | Path and SHA-256 of the starting weights |
| `dataset` | Version, content hash, class map, SHA-256 of both COCO files and the provenance file, one SHA-256 per split over every image |
| `logs` | SHA-256 of `log.txt` and `console.log` after fitting |
| `checkpoints.latest` | `checkpoint.pth`, its SHA-256, size and epoch |
| `checkpoints.best` | `checkpoint_best_total.pth`, its SHA-256, size and the selection (metric, split, source, epoch, value) |
| `failure` | Operation, path and error when the run failed |

## Latest and best checkpoints

`checkpoint.pth` is overwritten by RF-DETR after every epoch. It is reported
as latest only if it reloads and its epoch equals the last epoch in `log.txt`.

The best checkpoint is RF-DETR's `checkpoint_best_total.pth`: the higher of
regular and EMA validation mAP@[.50:.95] (EMA only when strictly higher,
earliest epoch on ties). The selection is recomputed from `log.txt` and
recorded. RF-DETR labels these validation values `test_coco_eval_bbox` in
`log.txt` because its evaluator is shared; the launcher runs with
`run_test=False` and redirects the upstream test loader to `valid`, so the
frozen test set is never an input to selection.

Before it is reported, the best checkpoint must reload and match the run's
dataset class map, the same check the inference adapter applies.

## Failures

`run.json` is written atomically (temporary file, fsync, replace), so it is
either the previous complete version or the new one. Checkpoint references
are only added after verification. If fitting, a checkpoint, a log or a
metadata write fails, `status` becomes `failed`, `failure` names the
operation and path, and both checkpoint references are `null`. If `run.json`
itself cannot be written, the command exits with an error naming the
operation and path, and the last complete `run.json` still has no checkpoint
references.

Run the tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_run_artifacts.py tests/test_train_rfdetr.py -q
```
