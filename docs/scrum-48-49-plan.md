# Implementation plan: SCRUM-48 and SCRUM-49

Reviewed November 1, 2026 against the current local `main` checkout and
`C:/Users/Mikun/Downloads/scrum-48-49.md`. The comparison below records the
pre-implementation baseline. Backend implementation and Python 3.12/PostgreSQL
18 acceptance checks are complete; see
[Storage operations](storage-operations.md) for implemented contracts and
individually testable acceptance work items.

## Current project compared with the tasks

- **Database foundation exists.** `app/storage/models.py`, migration `0001`,
  configuration, engine/session helpers, and schema checks cover recordings,
  model versions, inspections, datasets, training, and evaluation. UUIDv7 IDs
  and creation timestamps already have PostgreSQL defaults. Do not rebuild
  this layer or edit `0001_initial_schema.py`.
- **Media foundation exists.** `app/storage/media.py` copies files atomically,
  records size and SHA-256, confines paths to the media root, and exposes
  `media_path`. It removes a copied file when its row insert fails; it does
  not currently undo successful copies if the enclosing transaction later
  rolls back.
- **SCRUM-48 service layer is missing.** None of `recordings.py`,
  `model_registry.py`, `inspections.py`, `datasets.py`, or `training.py`
  exists under `app/storage`.
- **Dataset inputs already exist.** `app/dataset/records.py` supplies accepted
  and rejected images, COCO objects, and provenance. `manifest.py` and
  `manifest_save.py` build and save manifests. They do not import database
  rows. `split_dataset.py` has a separate split manifest; the dataset records
  and version manifest do not currently carry actual split assignments.
- **The desktop app is a prototype.** `app/__main__.py` opens source images
  and displays mock results. `app/inference.py` returns named classes and
  pixel `xywh`, then draws overlays. It does not return RF-DETR class indices
  and `xyxy`, persist results, load ONNX models, or replay saved sessions.
  `requirements.txt` has no real RF-DETR/ONNX inference runtime.
- **SCRUM-49 commands are missing.** The storage CLI only supports `migrate`
  and `check`; settings have no `DINOS_PG_BIN`. The sample JPG files exist.
  Compose already declares a persistent PostgreSQL volume.
- **Tests provide a good starting point.** Storage tests cover schema,
  constraints, migrations, connection errors, and media relocation. The
  `db_session` fixture uses an outer rollback transaction: its commits are
  savepoints, so it cannot prove persistence through a new connection or
  engine. Use dedicated throwaway databases for restart/restore tests.

## Scope and working agreements

Deliver the storage APIs and recovery CLI described in the open Markdown
file. Reuse the current schema unless implementation demonstrates a gap.
Real model training, RF-DETR inference, a background worker, login,
evaluation workflows, and a full inspection-history UI are separate work.
For these tasks, saved result links mean resolving stored IDs to the correct
model and readable media; provide automated checks and a CLI demonstration.

If the sprint requires clickable links inside the desktop app, add a small
saved-session browser/reopen action as explicit integration scope. The current
UI has no result-link feature to test across restart.

All public storage functions take a SQLAlchemy `Session` first, including
summary, list, and replay reads whose abbreviated signatures omit it in the
task notes. Callers own commits; storage functions may flush but never commit.
Use PostgreSQL-generated IDs and lifecycle times. Recording capture time is
different: `create_recording(..., started_at=...)` accepts the supplied,
timezone-aware source capture time. Return ORM rows for entity operations;
use a defined summary result for aggregate counts.

## 1. Confirm the foundation and define input contracts

1. Branch from the current foundation after confirming it contains the
   SCRUM-46/47 changes. The local `main` already contains their storage files;
   remote merge status has not been checked.
2. Set up Python 3.12, project dependencies, PostgreSQL 18, and a writable
   media root using `docs/database.md`. Run migration and storage checks.
3. Run the existing suite with `DINOS_REQUIRE_DB=1` and Qt offscreen so missing
   database coverage fails instead of silently skipping.
4. Define frame/detection input records: media ID, nullable frame index,
   recording-relative timestamp, class index, confidence, and pixel `xyxy`.
   Define missing-ID errors, legal state transitions, category selection
   semantics, replay interval boundaries, and list pagination bounds.
5. Define still-image behavior: preserve `frame_index=None` and
   `sample_fps=None`; do not fabricate video FPS/duration. Still images may
   share timestamp zero in the sample recording. Replay and resume must
   distinguish them by media ID, rather than timestamp alone.

**Exit:** existing checks pass and the contracts are documented before the
services are used by each other.

## 2. Implement recording import and the model registry

Create `app/storage/recordings.py`:

- Implement `create_recording` and `get_recording` with the signatures in the
  task notes. Probe images with image decoding and videos with OpenCV;
  validate dimensions, finite FPS, frame count, and usable duration.
- Import through `add_media_file`; populate ordered `RecordingFile` rows.
  Calculate video offsets from cumulative durations using a documented
  rounding rule. Reject unreadable media before leaving partial records.
- Load ordered file/media relationships explicitly for detached callers.

Create `app/storage/model_registry.py`:

- Implement `register_model_version`, `list_model_versions(status=READY)`,
  and `get_model_version`. Validate unique known `class_keys`, input size,
  optional dataset reference, and at least one weights file.
- Create/reuse the model family, copy weights, and map zero-based output
  positions to seeded category IDs. Imported versions use imported provenance.
- Serialize version allocation with a model-row lock, including safe handling
  of concurrent creation of the same model name. Do not use an unprotected
  `max(version_number) + 1`.
- Set READY explicitly only when the registration contract is satisfied;
  the schema defaults to DRAFT. Checkpoint-only versions must not be treated
  as executable ONNX models. Keep the sample placeholder clearly identified
  and outside real inference loading.

**Exit:** IDs reopen with the same ordered media and class maps; concurrent
registration cannot allocate duplicate version numbers.

## 3. Implement inspection lifecycle, queries, and bulk saves

Create `app/storage/inspections.py`:

- Implement `create_session`, `start_session`, `complete_session`,
  `fail_session`, and `cancel_session`. Validate recording/model existence,
  intervals, cadence, threshold, and selected categories. Require READY on
  creation and recheck on start. Define empty category selection explicitly.
- Enforce legal transitions using locked rows or conditional updates; update
  status and database-generated timestamps together. Failed sessions require
  an actionable error. Terminal sessions must not accidentally restart.
- Implement newest-first `list_sessions` with search, model and date filters,
  validated limit/offset, and deterministic ID ordering for timestamp ties.
- Implement `get_session_summary` with model label, recording, interval,
  analyzed-frame count, and category totals, including empty frames and zero
  counts. Count frames separately from detections to avoid join inflation.
- Implement `frames_between` with a documented boundary convention, stable
  ordering, and explicit loading of detections/media for replay. Avoid
  implicitly loading all frames from the write-only relationship.

Implement `save_frame_results`:

1. Load session settings, recording membership/media dimensions, and the
   model class map once per call. Reject foreign media, invalid frame indices,
   invalid times, unknown classes, nonfinite numbers, and invalid confidence.
   Apply the documented category/threshold policy consistently.
2. Normalize `xyxy` with actual stored width/height. Clamp only edge errors
   within `1e-6` in normalized coordinates; reject larger excursions and
   degenerate boxes with useful messages.
3. Use PostgreSQL Core frame inserts in batches of about 500 with conflict
   handling on the existing `(session_id, media_file_id, frame_index)` key.
   Its NULLS NOT DISTINCT behavior already protects still-image retries.
4. Map returned IDs by the unique frame key, never by RETURNING row order.
   Insert detections only for newly returned frames, in the same transaction.
   Subdivide detection inserts as necessary to stay below driver bind limits:
   500 frames times 20 detections is already 10,000 detection rows.
5. Keep one caller-owned transaction per batch. The caller invokes the saver
   with bounded batches; internal chunking alone does not create commits.
   Persist empty frames and make repeated batches harmless.
6. Resume from the last committed position. A timestamp-only `>` cursor can
   lose distinct frames sharing a timestamp; use a deterministic media/frame
   tie-breaker or safely resend the boundary frames using idempotent inserts.

**Exit:** replay and counts match saved results; duplicate saves, including
empty/still frames, produce no duplicate detections or frames.

## 4. Implement dataset and training persistence

Create `app/storage/datasets.py`:

- Implement `import_dataset_version` and `freeze_dataset_version`. Verify the
  supplied manifest matches the input records and label/hash/source policy.
  Save the manifest and accepted image bytes in the media store.
- Preserve stable image keys, provenance, annotation IDs, and rejection
  reasons. Join COCO annotations by annotation-file identity plus image ID,
  using existing helpers. Normalize COCO pixel `xywh` boxes after conversion
  to corners; preserve accepted images with no objects.
- Define `category_map` by annotation-file identity and local category ID,
  or reject conflicting local-ID meanings across files. Never assume a COCO
  category ID is a database category ID.
- Leave split unset when no explicit split assignments are supplied. Do not
  infer a split from eligibility; the existing split script is a separate
  integration. Freeze status and database time together; reject changes via
  these services to frozen versions and duplicate version labels.
- Do not describe the current manifest hash as a hash of file bytes: it
  covers selected record/annotation content. Media SHA-256 covers copied bytes.

Create `app/storage/training.py`:

- Implement `create_training_run`, `record_epoch`, `request_stop`, and
  `finish_training_run`. Validate frozen dataset/base-model references,
  epoch bounds, metrics, terminal errors, and allowed state changes.
- Add an explicit start operation so queued runs can enter RUNNING with a
  database-generated `started_at`; the task's function list omits this step.
  Define queued-stop behavior and duplicate epoch behavior. Update epoch
  metrics and current/best epoch atomically.
- These APIs persist progress; a training worker and crash recovery policy
  remain separate integration work.

For multi-file operations in steps 2 and 4, add transaction-aware cleanup of
newly copied media on rollback, including tests with the existing savepoint
fixture. Never delete previously committed files. A database rollback by
itself cannot reverse filesystem writes.

**Exit:** datasets and training progress round-trip through the APIs with
provenance intact and valid lifecycle states.

## 5. Add sample records and media verification

Create `app/storage/sample_data.py` and `app/storage/verification.py`; extend
the CLI with argparse subcommands while preserving `migrate` and `check`.

- `sample-data` calls the SCRUM-48 functions to build one clearly labeled
  placeholder model, a recording containing the two supplied JPGs, and a
  completed session with sample detections. Each image has one frame row;
  use an empty frame among the sample results to demonstrate zero detections.
- Keep `SAMPLE-not-a-real-model.onnx` as the original filename and clearly
  label model metadata; the media store intentionally randomizes stored names.
  Do not attempt to run the placeholder through an inference runtime.
- Use a reserved sample identity and transaction/lock so repeat or concurrent
  invocations are no-ops after successful creation. Print session/model IDs
  and resolved links suitable for the restart demonstration.
- `verify-media` checks all registered files for existence/readability and
  size; `--full` streams SHA-256. Report missing, unreadable, and corrupt
  files with their owners: models, recordings/sessions, datasets, logs,
  and exports where referenced. Return nonzero on verification failure.
- Reuse the verification report for future UI warnings. The current
  `media_path` exception covers an individual missing file; it does not
  replace verification of the whole store.

**Exit:** sample creation is repeatable and all saved links pass verification.

## 6. Implement backup and restore

Create `app/storage/backup.py`; extend `config.py`, `.env.example`, and CLI:

- Add optional `DINOS_PG_BIN` with existing environment/file precedence.
  Discover `pg_dump` and `pg_restore` from that directory or PATH, inspect
  tool/server versions, and reject unsupported combinations with actionable
  messages. Use matching PostgreSQL 18 tools for the acceptance demonstration.
- Run subprocess argument lists without shell interpolation. Supply credentials
  through the subprocess environment or a temporary password file, avoiding
  password-bearing arguments/logs. Write custom dumps directly to a file,
  avoiding PowerShell binary redirection problems.
- `backup DEST`: require a new destination outside the source media root.
  Reject active analysis and require a maintenance window with no writers or
  media deletion. An active-job check alone cannot prevent a new job starting
  halfway through a backup; document exclusive operation or enforce it with
  a shared maintenance lock respected by writing services.
- Dump first, copy media second, excluding transient partial files. Write
  `backup.json` last only after successful checks, recording manifest format,
  schema revision, server version, UTC creation time, file count, relative
  paths, sizes, and SHA-256. Distinguish registered files from harmless extras.
  Leave interrupted output clearly incomplete and refuse to overwrite backups.
- `restore SOURCE`: preflight the manifest, dump, paths, hashes, compatibility,
  tools, target database, and empty destination media folder before mutation.
  The target must have no application objects: even a migrated database with
  only seeded categories is not empty. Do not run `migrate` before restoring.
- Run `pg_restore --no-owner --exit-on-error` against the empty target; use a
  single transaction where compatible. Preserve relative media paths when
  copying into the different configured root. Validate the restored schema
  and run full media verification before reporting success.
- Database and filesystem restoration are not a single atomic transaction.
  Stage/verify copied media and define failure reporting/cleanup so a failed
  restore never appears usable. Do not clear a non-empty target automatically.

**Exit:** a backup restores into a new database and new media root with stable
IDs, equal data, and valid file hashes.

## 7. Acceptance tests and sprint demonstration

Add focused tests alongside existing storage tests:

- Service tests for recordings, model registry, inspections, datasets, and
  training. Use `db_session` and builders for ordinary transactional tests.
- A committed round-trip test on a dedicated migrated `dinos_test_*` database:
  create entities through services, commit, close sessions/dispose engines,
  then use a new engine to reopen IDs and read all linked files. Assert
  timezone-aware timestamps, compare them in UTC, and allow equality because
  PostgreSQL `now()` is transaction-scoped.
- Bulk tests for duplicate/mixed old-and-new batches, empty frames, nullable
  image indices, invalid boxes/classes/media, boundary timestamps, and failure
  between frame and detection inserts. Assert the failed batch is atomic.
- A registered `slow` benchmark for 28,800 frames and 576,000 detections,
  measuring the real caller batch/commit strategy. Record hardware and elapsed
  time; the target is well under 60 seconds, to be demonstrated rather than
  assumed. Verify exact counts and measure peak memory as useful context.
- Sample idempotency and verification tests for missing files, wrong sizes,
  same-size corruption detectable only by `--full`, and owner diagnostics.
- A real dump/restore integration test using disposable source/target
  databases and separate temporary media roots. Compare every application
  table's counts and saved IDs, hash every linked file, and reopen
  session -> model -> ONNX and session -> frame -> image/video paths.
  Include a tiny generated video fixture because the JPG sample alone cannot
  prove video reopening. Check restored identity sequences by inserting new
  detail rows after restore.
- Rejection/failure tests for non-empty restore targets, invalid manifests,
  bad hashes, absent/wrong client tools, interrupted copying, and failed
  PostgreSQL commands. Ensure backups are never reported complete prematurely.
- Required final run with `DINOS_REQUIRE_DB=1`, the existing suite, new storage
  tests, migration-drift checks, and the slow benchmark. Missing PostgreSQL
  client tools must fail the acceptance run instead of quietly skipping it.

Manual demonstration on the Windows PC:

1. Migrate/check; create samples; save the printed IDs; run full verification.
2. Restart the database service (or Compose) and the app/process. Reopen those
   IDs and read their model/image files; verify no sample records duplicate.
3. Back up during the maintenance window using the installed PostgreSQL 18
   tools. Restore into a fresh database and different empty media root.
4. Repeat link/file checks and compare counts and hashes. Temporarily move
   one copied sample file to demonstrate an actionable missing-media warning,
   then put it back and verify again.
5. Record commands, tool versions, benchmark timing, and restart/restore
   evidence in `docs/database.md` or a linked acceptance record.

## Suggested delivery order

1. **SCRUM-48 / foundation services:** contracts, errors, recording import,
   registry, transaction cleanup, and their tests.
2. **SCRUM-48 / inspections:** lifecycle, reads, bulk saves, restart round-trip,
   and performance evidence.
3. **SCRUM-48 / datasets and training:** import/freeze, progress/state APIs,
   and their tests. All three increments are needed to finish task 48.
4. **SCRUM-49 / recovery:** sample/verify commands, backup/restore commands,
   committed integration tests, documentation, and Windows demonstration.

The dependency path is recording/registry -> inspection persistence ->
sample/verification -> backup/restore -> restart/restore acceptance. Dataset
and training services must also be completed for the full task-48 scope.
