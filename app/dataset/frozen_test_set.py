"""Version and lock the independent test set (SCRUM-53).

Run from the project root, after ``app.split_dataset`` (and ideally
``app.audit_dataset``) has produced a dataset folder::

    python -m app.dataset.frozen_test_set freeze  DATASET manifests/test_lock --version test-v1
    python -m app.dataset.frozen_test_set verify  manifests/test_lock/test-v1.json DATASET
    python -m app.dataset.frozen_test_set check   manifests/test_lock/test-v1.json FILE [FILE ...]

``freeze`` writes ``{version}.json``: the test images, the group each came from,
and the SHA-256 of every file's bytes. The file is write-once. An existing
version is never overwritten, and the saved file is made read-only. Commit it,
so the freeze is visible in git history.

``verify`` re-hashes the files on disk and reports anything that changed, went
missing, appeared, or now shares a group with train or valid.

``check`` (and ``assert_excluded`` from Python) is the guard for training,
threshold tuning and checkpoint selection (frozen in Sprint 2, in force through
Sprint 7). It matches by file *content*, so a renamed or copied test image is
still caught.

Exit codes: 0 = ok, 1 = findings, 2 = bad input.

This module does not split data, audit similarity, or label anything.
"""

import argparse
import hashlib
import json
import os
import re
import stat
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "valid", "test")
DEFAULT_FROZEN_IN = "Sprint 2"
DEFAULT_LOCKED_THROUGH = "Sprint 7"
_VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class FrozenSetError(ValueError):
    """The test set cannot be frozen, loaded or verified."""


class FrozenSetExistsError(FrozenSetError):
    """A lock for this version already exists."""


class FrozenSetLeakError(FrozenSetError):
    """Something that must stay out of training or tuning is in the frozen test set."""


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _lock_hash(files: list[dict], groups: list[str]) -> str:
    """Hash of the locked content only: files and groups.

    Labels (version, sprint names) and the timestamp are not part of it, so
    renaming a label does not look like a change to the data.
    """
    payload = {"files": files, "groups": groups, "schema_version": SCHEMA_VERSION}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _read_split_manifest(dataset: Path) -> dict:
    path = dataset / "split_manifest.json"
    if not path.is_file():
        raise FrozenSetError(f"No split_manifest.json in {dataset}.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        rows = manifest["images"]
    except (ValueError, KeyError, TypeError) as error:
        raise FrozenSetError(f"split_manifest.json is not usable: {error}") from error
    if not rows:
        raise FrozenSetError("The split manifest contains no images.")
    for row in rows:
        if row.get("split") not in SPLITS or not row.get("group_id") or not row.get("output"):
            raise FrozenSetError("Each image needs a valid split, group_id and output path.")
    return manifest


def _inside(root: Path, split: str, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root / split):
        raise FrozenSetError(f"Image path is outside its recorded partition: {relative}")
    return path


def _images_on_disk(root: Path, split: str) -> set[Path]:
    folder = root / split
    if not folder.is_dir():
        return set()
    return {
        p.resolve()
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }


def _group_splits(rows: list[dict]) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[row["group_id"]].add(row["split"])
    return seen


def _audit_summary(dataset: Path) -> tuple[str | None, dict | None, list[str]]:
    """Return (status, test-split class coverage, test-split warnings) from audit_report.json."""
    path = dataset / "audit_report.json"
    if not path.is_file():
        return None, None, []
    try:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        status = report["status"]
    except (ValueError, KeyError, TypeError) as error:
        raise FrozenSetError(f"audit_report.json is not usable: {error}") from error
    counts = (report.get("labels") or {}).get("counts", {}).get("test")
    warnings = [w for w in report.get("warnings", []) if str(w).startswith("test:")]
    return status, counts, warnings


def freeze_test_set(
    dataset_dir,
    version: str,
    frozen_in: str = DEFAULT_FROZEN_IN,
    locked_through: str = DEFAULT_LOCKED_THROUGH,
    now: datetime | None = None,
) -> dict:
    """Build (but do not save) the lock for the test split of ``dataset_dir``.

    Refuses to freeze when the split is not independent: a test group that also
    appears in train or valid, a test file whose bytes also appear in train or
    valid, test images on disk that the manifest does not list, or an audit
    report whose status is BLOCKED.
    """
    version = version.strip()
    if not _VERSION_PATTERN.match(version):
        raise FrozenSetError(
            "Version must be a single file-safe label such as 'test-v1' "
            "(letters, digits, '.', '_' or '-')."
        )
    root = Path(dataset_dir).resolve()
    manifest = _read_split_manifest(root)
    rows = manifest["images"]

    problems: list[str] = []

    audit_status, coverage, test_warnings = _audit_summary(root)
    if audit_status == "BLOCKED":
        problems.append("audit_report.json status is BLOCKED; resolve the audit errors first.")

    leaking_groups = sorted(g for g, s in _group_splits(rows).items() if len(s) > 1 and "test" in s)
    for group in leaking_groups:
        problems.append(f"Group {group} appears in test and another partition.")

    test_rows = [row for row in rows if row["split"] == "test"]
    if not test_rows:
        problems.append("The test partition has no images.")

    other_hashes: dict[str, str] = {}
    for row in rows:
        if row["split"] == "test":
            continue
        path = _inside(root, row["split"], row["output"])
        if path.is_file():
            other_hashes.setdefault(sha256_of(path), row["output"])

    files: list[dict] = []
    listed: set[Path] = set()
    for row in sorted(test_rows, key=lambda r: r["output"]):
        path = _inside(root, "test", row["output"])
        if not path.is_file():
            problems.append(f"Test image is missing: {row['output']}")
            continue
        listed.add(path)
        digest = sha256_of(path)
        if digest in other_hashes:
            problems.append(
                f"Test image {row['output']} is byte-identical to {other_hashes[digest]}."
            )
        files.append(
            {
                "group_id": row["group_id"],
                "output": row["output"],
                "sha256": digest,
                "source": row.get("source"),
            }
        )

    for extra in sorted(_images_on_disk(root, "test") - listed):
        problems.append(f"Image in test/ is absent from the manifest: {extra.relative_to(root).as_posix()}")

    if problems:
        raise FrozenSetError("Cannot freeze the test set:\n  " + "\n  ".join(problems))

    groups = sorted({file["group_id"] for file in files})
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "version": version,
        "frozen_in": frozen_in,
        "locked_through": locked_through,
        "locked_at": moment.isoformat(timespec="seconds"),
        "seed": manifest.get("seed"),
        "counts": {
            "images": len(files),
            "groups": len(groups),
            "all_partitions": dict(sorted(_count_splits(rows).items())),
        },
        "audit_status": audit_status,
        "test_class_coverage": coverage,
        "test_audit_warnings": test_warnings,
        "groups": groups,
        "files": files,
        "lock_hash": _lock_hash(files, groups),
    }


def _count_splits(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["split"]] += 1
    return counts


def save_test_lock(lock: dict, output_directory) -> Path:
    """Write ``lock`` to ``{output_directory}/{version}.json``, write-once and read-only."""
    version = str(lock.get("version", ""))
    if not _VERSION_PATTERN.match(version):
        raise FrozenSetError("The lock has no file-safe version label.")
    folder = Path(output_directory).resolve()
    destination = folder / f"{version}.json"
    folder.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FrozenSetExistsError(f"A test lock for version {version} already exists.")

    temporary = folder / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write((json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)  # fails rather than overwrites
        except FileExistsError as error:
            raise FrozenSetExistsError(f"A test lock already exists at {destination}.") from error
    finally:
        temporary.unlink(missing_ok=True)
    destination.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    return destination


def load_test_lock(path) -> dict:
    """Read a lock and confirm it has not been edited since it was frozen."""
    try:
        lock = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        files, groups, recorded = lock["files"], lock["groups"], lock["lock_hash"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise FrozenSetError(f"Test lock is not usable: {error}") from error
    if _lock_hash(files, groups) != recorded:
        raise FrozenSetError("Test lock has been modified: its lock_hash no longer matches its contents.")
    return lock


def verify_test_lock(lock_path, dataset_dir) -> list[str]:
    """Return a list of problems; an empty list means the test set is exactly as frozen."""
    lock = load_test_lock(lock_path)  # raises if the lock itself was edited
    root = Path(dataset_dir).resolve()
    problems: list[str] = []

    listed: set[Path] = set()
    for entry in lock["files"]:
        path = _inside(root, "test", entry["output"])
        listed.add(path)
        if not path.is_file():
            problems.append(f"Missing: {entry['output']}")
        elif sha256_of(path) != entry["sha256"]:
            problems.append(f"Changed: {entry['output']}")

    for extra in sorted(_images_on_disk(root, "test") - listed):
        problems.append(f"Added after freeze: {extra.relative_to(root).as_posix()}")

    manifest_path = root / "split_manifest.json"
    if manifest_path.is_file():
        rows = _read_split_manifest(root)["images"]
        locked_groups = set(lock["groups"])
        for group in sorted(locked_groups):
            splits = _group_splits(rows).get(group, set())
            if splits - {"test"}:
                problems.append(f"Group {group} now also appears in {', '.join(sorted(splits - {'test'}))}.")
    return problems


def locked_hashes(lock: dict) -> frozenset[str]:
    return frozenset(entry["sha256"] for entry in lock["files"])


def leaked_paths(lock: dict, paths: Iterable) -> list[str]:
    """Paths (by file content) that match a frozen test image."""
    frozen = locked_hashes(lock)
    return [str(p) for p in paths if Path(p).is_file() and sha256_of(Path(p)) in frozen]


def leaked_groups(lock: dict, group_ids: Iterable[str]) -> list[str]:
    frozen = set(lock["groups"])
    return sorted({g for g in group_ids if g in frozen})


def assert_excluded(lock: dict, paths: Iterable = (), group_ids: Iterable[str] = ()) -> None:
    """Raise if any file or group belongs to the frozen test set.

    Call this on the file list used for training, threshold tuning and
    checkpoint selection before they run.
    """
    bad_paths = leaked_paths(lock, paths)
    bad_groups = leaked_groups(lock, group_ids)
    if bad_paths or bad_groups:
        parts = []
        if bad_paths:
            parts.append(f"{len(bad_paths)} file(s) match the frozen test set, e.g. {bad_paths[0]}")
        if bad_groups:
            parts.append(f"{len(bad_groups)} group(s) are in the frozen test set, e.g. {bad_groups[0]}")
        raise FrozenSetLeakError("; ".join(parts))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze", help="hash the test split and write a read-only lock")
    freeze.add_argument("dataset", type=Path)
    freeze.add_argument("output", type=Path, help="folder that receives {version}.json")
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--frozen-in", default=DEFAULT_FROZEN_IN)
    freeze.add_argument("--locked-through", default=DEFAULT_LOCKED_THROUGH)

    verify = commands.add_parser("verify", help="re-hash the test files and compare with the lock")
    verify.add_argument("lock", type=Path)
    verify.add_argument("dataset", type=Path)

    check = commands.add_parser("check", help="fail if any file is a frozen test image")
    check.add_argument("lock", type=Path)
    check.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            lock = freeze_test_set(args.dataset, args.version, args.frozen_in, args.locked_through)
            target = save_test_lock(lock, args.output)
            print(f"SCRUM-53: froze {lock['counts']['images']} test images in {lock['counts']['groups']} groups")
            print(f"Audit status: {lock['audit_status'] or 'no audit_report.json found'}")
            print(f"lock_hash: {lock['lock_hash']}")
            print(f"Saved (read-only): {target}")
            return 0

        if args.command == "verify":
            problems = verify_test_lock(args.lock, args.dataset)
            for problem in problems:
                print(f"  {problem}")
            print("Test set matches the lock." if not problems else f"{len(problems)} problem(s) found.")
            return 1 if problems else 0

        lock = load_test_lock(args.lock)
        bad = leaked_paths(lock, args.paths)
        for path in bad:
            print(f"  FROZEN TEST IMAGE: {path}")
        print("None of the files are in the frozen test set." if not bad else f"{len(bad)} file(s) are frozen test images.")
        return 1 if bad else 0
    except (OSError, FrozenSetError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())