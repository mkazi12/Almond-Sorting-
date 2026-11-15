# Validation inference benchmarking (SCRUM-63)

Use the RF-DETR environment described in [rfdetr-training.md](rfdetr-training.md).
Supply a trusted compatible checkpoint, approved class map, versioned dataset
manifest, split manifest and frozen test lock. No weights or production dataset
are included in this benchmark implementation.

From PowerShell, with that environment activated:

```powershell
python -m app.evaluate DATASET_ROOT MANIFEST SPLIT_MANIFEST TEST_LOCK `
  CHECKPOINT CLASS_MAP results/evaluation.json --variant nano --device cuda `
  --confidence-floor 0.5 --iou-threshold 0.5 `
  --benchmark-output results/benchmark.json --warmup-runs 5
```

Replace the placeholders with actual paths and choose the checkpoint's variant.
Use `--device cpu` for a CPU workstation or `mps` for an available Apple accelerator.
Both output files must be new and distinct when benchmarking is enabled. Parent
directories are created. Without benchmark flags, evaluation behavior is unchanged.
Benchmark-enabled JSON publication atomically refuses to overwrite even a file
created during inference. It uses same-directory hard links and fails closed on
filesystems without hard-link support. Evaluation-only commands retain their
existing overwrite behavior.
`--warmup-runs` requires `--benchmark-output`; its default is 5 and zero is allowed.

The evaluator runs normally, saves evaluation JSON, and rechecks the same
validation membership/content guard before benchmarking. It loads one image at
a time in a second pass. Test images are never selected for inference. Image
decoding and dimension checks happen before timing; no decoded dataset is cached.
The already-loaded detector is reused. Therefore these are steady-state timings
after evaluation, not cold startup measurements. Additional warm-ups use the
first validation image and are excluded from statistics.

Each validation image contributes one batch-size-one timed sample. The interval
covers `Detector.predict`, including conversion, preprocessing, model execution,
and output conversion. CUDA/MPS completion is synchronized around each sample.
Image loading, metric calculation and artifact writes are excluded. Throughput
is `samples / summed timed seconds`: it is sequential inference images/second,
not the full desktop application's processing rate.

Benchmark JSON schema version 1 copies `evaluation_id`, checkpoint identity,
dataset identity (including lock hashes and ordered validation IDs), and settings
from evaluation JSON. `evaluation_sha256` identifies the exact saved evaluation
file; `evaluation_created_at` preserves its timestamp. Evaluation schema remains
version 1 and gains no benchmark fields.

The `benchmark` object contains warm-up/sample counts, per-image milliseconds,
original dimensions and dimension frequencies, model input resolution (square
side length), p50/p95, total timed seconds, throughput, and runtime metadata.
Percentiles use linear interpolation at `(n - 1) * q` over sorted samples.
A single image gives identical p50 and p95; small sets provide limited evidence
about tail latency. Missing descriptive metadata is `null`, not invented.
RF-DETR input size is read from the loaded model's `model.resolution`, as used by
the [pinned 1.3.0 predict implementation](https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/detr.py).

Keep checkpoint, dataset, thresholds, device, resolution and runtime versions
consistent when comparing runs. Close unrelated heavy workloads and record
repeated real runs separately. This command does not optimize or compile the
model or decide whether it is ready for operator trials.

If benchmarking fails, the command exits nonzero and reports the location of the
valid evaluation JSON retained on disk. No partial benchmark is published. Each
JSON is written atomically, but the two files are not a single transaction. Use
new output paths for a retry.

## Markdown evaluation reporting (SCRUM-64)

Add `--report-output results/report.md` to the evaluation command above to
generate evaluation JSON, benchmark JSON and Markdown in one invocation.
`--report-output` requires `--benchmark-output`. All three output paths must
be new, distinct files, separate from the input files. Without reporting flags,
evaluation and benchmarking retain their existing behavior.

To create a report later from existing saved JSON, without loading a model or
reading any images:

```powershell
python -m app.evaluation_report results/evaluation.json results/benchmark.json results/report.md
```

The generator reads both files from disk and verifies schema versions, evaluation
ID/timestamp, checkpoint, dataset and lock identities, settings, validation image
membership, sample count and device. It checks `evaluation_sha256` against the
exact bytes read from evaluation JSON, so even edits to that file's whitespace
require a matching benchmark artifact. This is an integrity check, not a digital
signature or an independent verification of model performance.
Image dimensions and resolution-summary counts must be positive integers, and
the summary must match the saved samples without duplicate sizes. Model input
resolution must be a positive integer or `null` when unavailable. These checks
validate metadata only; they do not recalculate numerical performance metrics.

Markdown presents overall and per-class AP, operating precision/recall, labeled
object counts, checkpoint and dataset identities, thresholds, latency/throughput,
warm-up/sample counts, resolutions, and hardware/runtime details. Values are
copied from the saved JSON without recalculating metrics or percentiles. Accuracy
values remain on the 0–1 scale; `N/A` means undefined or unavailable, not zero.
The report records the AP prediction confidence floor because AP is calculated
from predictions retained at that floor.

Invalid or mismatched inputs fail before a report is published. Existing reports
are never overwritten. The generator writes and flushes a temporary file beside
the destination, then publishes it atomically using a hard link; filesystems that
do not support hard links fail with a clear error. Temporary files are cleaned up.
If reporting fails after evaluation and benchmarking succeed, their JSON files
are retained and the combined command exits nonzero. Fix the report destination
or inputs and retry the standalone reporting command with a new report path.

Reporting tests create temporary JSON fixtures using fake-detector evaluation,
then exercise saved-value rendering, mismatches, malformed inputs and publication
failures. Run them with:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_evaluation_report.py tests/test_evaluate.py -q
```

Tests use fake detectors, clocks and runtime providers to verify calculations,
streaming and failure handling without GPU hardware or weights:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_benchmark.py tests/test_evaluate.py tests/test_rfdetr_adapter.py -q
```

These checks establish implementation correctness, not real model speed. When
weights and approved validation inputs become available, run the command above,
check the recorded identities, sample count and hardware, and retain both JSON
files as the real integration benchmark evidence.
