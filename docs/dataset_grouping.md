# Dataset grouping (SCRUM-50)

Training, validation and test data must come from different physical sources, so
the final quality number reflects almonds the model has never seen. To enforce
that, related frames are put in one **group**, and a group is never split across
partitions.

## What counts as one group

Images belong to the same group if they share any of:

| Field | Meaning | Example |
|---|---|---|
| `source_lot` | The batch of almonds the frame came from | `lot-2026-10-A` |
| `source_session` | One capture session, camera run or recording | `run-0412` |
| `near_duplicate_group_id` | A cluster of near-identical frames from one recording | `dup-17` |

These links are **transitive**. If frame A and B share a lot and B and C share a
session, then A, B and C are one group. This is deliberate: it is the only way
to guarantee that no lot or session ID appears in more than one split.

Implementation: `app/dataset/grouping.py`, `assign_groups(images, policy)`.

## The group ID

`group_id` looks like `grp-3f9a1c0b22de`. It is a SHA-256 prefix of the sorted
lot/session/near-duplicate keys in the group, so:

- the same records always give the same ID, regardless of read order;
- adding a frame to an existing lot keeps the ID;
- adding a frame that links two groups merges them, and the merged group gets a
  new ID. Group before you split, and freeze the test set only after grouping is final.

Keys are namespaced (`lot:A`, `session:A`) so a lot and a session with the same
name do not get merged by accident.

## Images that cannot be grouped

An accepted image is **held out of splitting** (returned in `GroupingResult.ungrouped`)
when:

- a field required by the dataset's `SourcePolicy` is blank, or
- it has no lot, session or near-duplicate ID at all.

These are never assigned to a guessed group. Fix the metadata, then regroup.

## How it connects to the rest of the pipeline

1. Import accepted images with their lot/session metadata (`app/dataset/records.py`).
2. `assign_groups` produces one `group_id` per image.
3. The splitter (SCRUM-51) assigns whole groups to train, valid or test.
4. The audit (SCRUM-52) re-checks that no group crosses a split.
5. The test split is frozen and hashed (SCRUM-53, `docs/frozen_test_set.md`).

## Open point

`split_dataset.py` currently uses the source **folder name** as `group_id`. That is
correct only if each folder is exactly one lot or recording. If a lot spans several
folders, or one folder mixes lots, grouping from folder names will leak. Switching
the splitter to `assign_groups` removes that risk.