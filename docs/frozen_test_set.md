# Frozen test set (SCRUM-53)

The test split is frozen once, hashed, and then kept out of training, threshold
tuning and checkpoint selection. Freeze it in Sprint 2; the exclusion holds through Sprint 7.

Implementation: `app/dataset/frozen_test_set.py`.

## Freeze it (once)

```bash
python -m app.split_dataset RAW_FOLDER dataset_demo          # SCRUM-51
python -m app.audit_dataset dataset_demo                     # SCRUM-52 (recommended first)
python -m app.dataset.frozen_test_set freeze dataset_demo manifests/test_lock --version test-v1
git add manifests/test_lock/test-v1.json && git commit
```

`freeze` refuses, and writes nothing, if:

- a test group also appears in train or valid,
- a test file has the same bytes as a train or valid file,
- test images on disk are missing from the manifest, or listed ones are missing from disk,
- `audit_report.json` exists and its status is `BLOCKED`.

The lock `manifests/test_lock/test-v1.json` holds, per test image: output path, source
path, group ID and SHA-256. It also records the group list, split counts, seed, audit
status, and the test split's class coverage and rare-class warnings (copied from the audit
report when present). `lock_hash` is a SHA-256 over the files and groups only, so it is the
same on any machine and unaffected by the version label or timestamp.

The file is write-once: freezing the same version again fails, and the saved file is
read-only. To change the test set, freeze a new version (`test-v2`) and say why in the commit.

## Check it (any time)

```bash
python -m app.dataset.frozen_test_set verify manifests/test_lock/test-v1.json dataset_demo
```

Reports files that were changed, went missing or were added after the freeze, and any
locked group that now also appears in train or valid. It also fails if the lock file itself
was edited. Exit code 0 means identical to the freeze.

## Keep it out of training and tuning

In any training, threshold-tuning or checkpoint-selection code, call the guard on the
files and groups being used:

```python
from app.dataset.frozen_test_set import load_test_lock, assert_excluded

lock = load_test_lock("manifests/test_lock/test-v1.json")
assert_excluded(lock, paths=training_files, group_ids=training_group_ids)
```

It raises `FrozenSetLeakError` if anything matches. Matching is by file **content**, so a
renamed or copied test image is still caught. Command-line version:

```bash
python -m app.dataset.frozen_test_set check manifests/test_lock/test-v1.json FILE [FILE ...]
```

## Limits

- The lock proves the test files are unchanged and kept apart. It cannot prove that two
  *different* photos of the same almond are independent; that is what grouping
  (`docs/dataset_grouping.md`) and the audit's near-duplicate review are for.
- The guard only works where it is called. Add it to the training and evaluation entry points.