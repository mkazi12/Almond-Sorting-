# SCRUM-48/49 storage operations and acceptance

The service APIs and CLI are implemented against the schema documented in
[database.md](database.md). They do not add a real inference/training runtime
or an inspection-history screen to the prototype UI. The sample model is an
explicitly labeled placeholder and must never be executed as ONNX.

**Current acceptance status: SCRUM-48 and SCRUM-49 backend checks passed on
Python 3.12.10 and PostgreSQL 18.6 on this Windows workstation.** See the final
acceptance record below; earlier partial runs are retained as history.

## Prerequisites

Use the project Python environment and PostgreSQL 18 or newer. Configure the
`DINOS_DB_*` settings and `DINOS_MEDIA_ROOT` as in `database.md`. Install the
PostgreSQL client tools; the binary psycopg wheel does not include `pg_dump`
or `pg_restore` executables. Optionally set:

```dotenv
DINOS_PG_BIN=C:\Program Files\PostgreSQL\18\bin
```

Otherwise the tools are discovered on PATH. Each tool must be at least the
server's major version; matching server/client majors are recommended.
Relative `DINOS_PG_BIN` values resolve like `DINOS_MEDIA_ROOT`, against the
file defining the setting (or the working directory for environment values).

## SCRUM-48 dependency

These commands use the create/read APIs from the preceding SCRUM-48 branch.
See [Storage services](storage-services.md) for their contracts and tests.

## Sample and verification

```powershell
python -m app.storage migrate
python -m app.storage check
python -m app.storage sample-data
python -m app.storage verify-media
python -m app.storage verify-media --full
```

`sample-data` prints the stable session ID and absolute links for the model
and both supplied JPGs. One frame has a sample detection and one has none.
Repeated/concurrent invocations reuse the existing reserved sample session.
The placeholder's original filename is `SAMPLE-not-a-real-model.onnx`; stored
filenames are randomized by the media store, as for other imports.

Verification checks every registered file, reports affected owners, and exits
nonzero when any file is missing, unreadable or the wrong size. `--full` also
streams SHA-256, detecting same-size corruption. Run it after moving the media
root, restoring backups, or investigating unavailable results.

## Backup and restore

Close the app and stop analysis, training, evaluation, and external media
writers/deletion before backup. Service writes share a maintenance lock;
backup takes it exclusively and rejects persisted active jobs. Direct SQL
and arbitrary filesystem changes do not participate in this lock.

```powershell
python -m app.storage backup D:\Backups\2026-11-01
```

Choose a **new** directory outside the media root. Backup verifies registered
media, dumps PostgreSQL first, copies media second, and publishes `backup.json`
last. The manifest records schema, server version, UTC creation time, count,
paths, sizes, hashes, and the dump hash. Partial files are excluded. Existing
backups are never overwritten. An interrupted backup without a valid manifest
cannot be restored. Retry into a new directory after fixing the error.

To restore, create a **fresh empty database** and select a different empty
media directory in your settings. Do not migrate the restore target first:
even a migrated database containing only seeded categories is non-empty.
Keep the original database and backup intact.

```powershell
python -m app.storage restore D:\Backups\2026-11-01
python -m app.storage verify-media --full
python -m app.storage sample-data
```

Restore validates the complete backup and empty targets, copies/verifies media,
then restores SQL with `--no-owner --no-acl --exit-on-error --single-transaction`.
The existing schema and IDs come from the dump. Final schema and full media
verification must succeed before success is reported. Copy/SQL failures remove
newly copied files and empty directories. Database and filesystem operations
cannot form one atomic transaction; if final verification fails after SQL
restore, keep the app stopped, inspect the diagnostic, and use fresh targets
for another attempt. Restore never clears an existing database or media folder.

## Individually testable work items

Run all commands from the repository root with settings pointing at a
PostgreSQL 18 test server. Tests create/drop only their own `dinos_test_*`
databases. The database role needs permission to create those databases.

```powershell
$env:DINOS_REQUIRE_DB = '1'
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest -q tests/test_storage_services.py
python -m pytest -q tests/test_storage_recovery.py
python -m pytest -q --run-slow
```

1. **Recording create/read:** service tests reopen ordered media and source
   capture time; decode video metadata and cumulative offsets; reject bad
   files before copying. Pass when IDs/paths match and offsets are correct.
2. **Model registry:** service tests verify class order, READY/DRAFT filtering,
   bad-category rejection, version increments, eager reads, and concurrent
   allocation across two connections. Pass when two registrations get 1/2.
3. **Inspection lifecycle/read:** service tests cover category selection,
   archived models, terminal transitions, filtering, counts and replay bounds.
   Pass when empty frames count and detection joins do not inflate totals.
4. **Bulk save:** service tests cover box round-trip/tolerance, duplicates,
   threshold filtering, still/video keys and an invalid second chunk that
   rolls back the first chunk. Pass when retries create no extra rows and
   failed calls leave no partial frames/detections.
5. **Datasets/training:** service tests verify normalized annotation boxes,
   image/annotation identities, provenance, rejection rows, freeze behavior,
   epoch ordering and completed/stopped/failed states. Pass when DB records
   match source records and illegal changes are rejected.
6. **Restart/sample:** recovery tests commit, dispose the engine, reopen via
   another engine, and assert sample idempotency, timestamps, file bytes and
   hashes. Pass when every original ID/link resolves after reopening.
7. **Verify/restore:** recovery tests use actual `pg_dump`/`pg_restore` with
   a different database/root, compare every application's table counts,
   reopen images/model and a decoded video, and insert after restore to check
   identity sequences. They also test corrupt media/dumps, missing tools,
   active jobs, failed SQL cleanup and non-empty target rejection. Tools
   missing from the configured acceptance environment cause failure.
8. **Performance:** `python -m pytest -q -s --run-slow
   tests/test_storage_performance.py::test_full_shift_bulk_save_under_a_minute`
   saves 28,800 frames/576,000 detections, commits every 500 frames, checks
   exact counts and requires elapsed save time below 60 seconds. The video
   bytes in this benchmark are a metadata fixture; actual decode/restore is
   checked separately with a generated AVI.

Before the sprint demo, also restart the actual database service and app,
then reopen the printed sample links and restore into fresh targets on the
delivery PC. Automated engine recreation proves committed storage independence;
it does not itself restart the PostgreSQL process. The prototype UI currently
has no saved-result browser, so opening results inside that UI is future
integration work.

## Validation recorded November 1, 2026

- Full suite with `DINOS_REQUIRE_DB=1`, Qt offscreen, and `--run-slow`:
  **208 passed, 2 skipped, 35 subtests passed** in 97 seconds. The two skips
  are existing Windows-inapplicable symlink/permission tests; database and
  backup/restore tests ran against a real server with real client tools.
- Full-shift benchmark measured **34.44 seconds** in its isolated run for
  28,800 frames and 576,000 detections, including a commit every 500 frames.
  It also passed its under-60-second assertion in the final full-suite run.
- Separate actual PostgreSQL process restart: committed a sample, restarted
  the isolated cluster, reopened the same session/model/image IDs and paths,
  and verified all SHA-256 hashes. The temporary database was dropped and
  the isolated server was stopped afterward. Existing PostgreSQL 16 was
  not changed.
- Environment: Windows 11 build 26200, AMD64 Family 25 Model 80, Python
  **3.14.6**, PostgreSQL/client tools **18.6**, SQLAlchemy **2.1.6**, psycopg
  **3.3.6**. All packages came from the existing pinned requirements.
  Python 3.12 was not installed for that initial run. The required Python
  3.12 verification was subsequently completed as recorded below. These runs
  do not change the project's supported Python version.

### Initial Python 3.12 attempt, November 1, 2026 (superseded below)

- Prepared a project-local Python **3.12.10** runtime from the
  [official Windows embeddable package](https://www.python.org/downloads/release/python-31210/)
  under ignored `.local-test/python312`, with every dependency from the
  existing `requirements.txt`. Default/system Python was not changed.
- The normal workspace run completed with **98 passed, 112 skipped,
  35 subtests passed** in **13.26 seconds**. Two skips are the Windows
  symlink/permission checks; the others require PostgreSQL or the explicitly
  enabled slow benchmark. This is a partial check, not full database acceptance.
- Strengthened restart acceptance to compare exact model/recording IDs, media
  paths, frame IDs, timestamps, and explicit UTC offsets across engine
  recreation. These assertions were pending at this stage and passed in the
  final run below.
- Full-suite attempts through an elevated runner failed to import
  `pytest.__main__`; running pytest in the normal workspace succeeds.
  A subsequent request to start only the isolated PostgreSQL test server
  for that normal run was declined. At that point the Python
  3.12 database/restore/restart checks and performance gate were pending.
  The prior Python 3.14 PostgreSQL results above are separate evidence.

The database-required acceptance command, run from the normal workspace:

```powershell
$env:DINOS_ENV_FILE = '.local-test/test.env'
$env:DINOS_REQUIRE_DB = '1'
$env:QT_QPA_PLATFORM = 'offscreen'
.local-test\python312\python.exe -m pytest -q -rs -s --run-slow
```

### Final Python 3.12 acceptance, November 1, 2026 — passed

- Runtime: Python **3.12.10** with the existing pinned dependencies, Windows
  11, PostgreSQL server and native `pg_dump`/`pg_restore` tools **18.6**.
  The isolated test server listened only on `127.0.0.1:55489`.
- Full database-required suite, including the slow benchmark:
  **208 passed, 2 skipped, 35 subtests passed in 105.88 seconds**. The only
  skipped checks concern Windows symlink privileges and POSIX permission
  bits; no database acceptance checks were skipped.
- **SCRUM-48:** service create/read and lifecycle checks, database-generated
  IDs and UTC timestamps, exact reopen comparisons, category/model validation,
  duplicate-safe and atomic frame saves, normalized boxes, dataset import,
  training persistence, and concurrent version allocation passed.
- Performance: **28,800 frames and 576,000 detections saved in 48.19 seconds**,
  including a commit every 500 frames, with exact final row counts. The
  under-60-second acceptance gate passed under Python 3.12.
- **SCRUM-49:** sample idempotency, separate-process CLI checks, real custom
  dump/restore into a different database and media folder, every table's row
  counts, model/image/video links and hashes, media diagnostics, failed restore
  cleanup, and non-empty target rejection passed.
- After the suite, a separate committed Python 3.12 sample survived an actual
  PostgreSQL process restart. Exact session/model/recording/frame IDs, UTC
  timestamps, relative paths, readable linked files, and full media hashes
  matched the snapshot taken before restarting.
- Test databases were removed; the isolated PostgreSQL server was stopped.
  Existing PostgreSQL 16 and application data were not changed. Scratch runtime,
  client binaries, and test media remain under the ignored `.local-test/` folder.

No backend acceptance checks from the supplied SCRUM-48/49 task document remain
pending. Real inference/training execution and a desktop saved-result browser
remain outside these two storage tasks.

### Separate PR branch validation

SCRUM-49 is stacked on `feature/scrum-48-storage-operations`; merge that PR
first and then retarget the SCRUM-49 PR to `main` before merging it.

After updating the base to include the newly merged dataset grouping/frozen
test work, the combined branch passed **234 tests and 35 subtests** in
78.83 seconds under Python 3.12.10/PostgreSQL 18.6. The two Windows-only checks
were skipped, and the inherited benchmark was verified separately on the
SCRUM-48 branch. Its measured result and timing variability are recorded in
[Storage services](storage-services.md).

The service benchmark now lives in `tests/test_storage_performance.py`; this
PR adds `tests/test_storage_recovery.py` for its sample, verification, CLI,
backup/restore, relocation, failure, and restart checks.
