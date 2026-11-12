import copy
import json
import shutil
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.dataset import frozen_test_set as frozen
from app.split_dataset import split_dataset

FIXED_TIME = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def make_source(root: Path) -> Path:
    """Five recordings with two distinct images each (enough for train/valid/test)."""
    source = root / "raw"
    for group in ("lotA", "lotB", "lotC", "lotD", "lotE"):
        (source / group).mkdir(parents=True)
        for index in range(2):
            (source / group / f"{group}_{index}.jpg").write_bytes(f"{group}-{index}".encode())
    return source


class FrozenTestSetTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        self.dataset = self.tmp / "dataset"
        split_dataset(make_source(self.tmp), self.dataset, seed=1)
        self.locks = self.tmp / "locks"

    def freeze(self, version="test-v1", dataset=None):
        return frozen.freeze_test_set(dataset or self.dataset, version, now=FIXED_TIME)

    def manifest(self):
        return json.loads((self.dataset / "split_manifest.json").read_text())

    def write_manifest(self, manifest):
        (self.dataset / "split_manifest.json").write_text(json.dumps(manifest))

    def test_freeze_records_every_test_image_with_its_hash_and_group(self):
        lock = self.freeze()
        test_rows = [r for r in self.manifest()["images"] if r["split"] == "test"]
        self.assertEqual(lock["counts"]["images"], len(test_rows))
        self.assertEqual({f["output"] for f in lock["files"]}, {r["output"] for r in test_rows})
        for entry in lock["files"]:
            self.assertEqual(entry["sha256"], frozen.sha256_of(self.dataset / entry["output"]))
        self.assertEqual(set(lock["groups"]), {r["group_id"] for r in test_rows})
        self.assertRegex(lock["lock_hash"], r"^[0-9a-f]{64}$")
        self.assertEqual(lock["frozen_in"], "Sprint 2")
        self.assertEqual(lock["locked_through"], "Sprint 7")
        self.assertEqual(sum(lock["counts"]["all_partitions"].values()), 10)

    def test_lock_hash_ignores_where_the_dataset_lives_and_the_time(self):
        other_home = self.tmp / "other_machine"
        other_home.mkdir()
        other = other_home / "dataset"
        split_dataset(make_source(other_home), other, seed=1)
        first = self.freeze()
        second = frozen.freeze_test_set(other, "test-v1", now=datetime(2030, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(first["lock_hash"], second["lock_hash"])

    def test_save_is_write_once_and_read_only(self):
        lock = self.freeze()
        path = frozen.save_test_lock(lock, self.locks)
        self.assertEqual(path, self.locks / "test-v1.json")
        self.assertFalse(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        original = path.read_bytes()

        with self.assertRaises(frozen.FrozenSetExistsError):
            frozen.save_test_lock(self.freeze(), self.locks)

        self.assertEqual(path.read_bytes(), original)
        self.assertEqual([p.name for p in self.locks.iterdir()], ["test-v1.json"])

    def test_edited_lock_is_detected_on_load(self):
        path = frozen.save_test_lock(self.freeze(), self.locks)
        path.chmod(0o644)
        data = json.loads(path.read_text())
        data["files"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(data))
        with self.assertRaises(frozen.FrozenSetError):
            frozen.load_test_lock(path)

    def test_verify_is_clean_when_nothing_changed(self):
        path = frozen.save_test_lock(self.freeze(), self.locks)
        self.assertEqual(frozen.verify_test_lock(path, self.dataset), [])

    def test_verify_reports_changed_missing_and_added_files(self):
        lock = self.freeze()
        path = frozen.save_test_lock(lock, self.locks)
        changed, missing = (self.dataset / f["output"] for f in lock["files"][:2])
        changed.write_bytes(b"tampered")
        missing.unlink()
        (self.dataset / "test" / "sneaky.png").write_bytes(b"new")

        problems = frozen.verify_test_lock(path, self.dataset)

        self.assertTrue(any(p.startswith("Changed:") for p in problems))
        self.assertTrue(any(p.startswith("Missing:") for p in problems))
        self.assertTrue(any(p.startswith("Added after freeze:") for p in problems))

    def test_verify_reports_a_locked_group_that_reappears_in_train(self):
        path = frozen.save_test_lock(self.freeze(), self.locks)
        manifest = self.manifest()
        test_group = next(r["group_id"] for r in manifest["images"] if r["split"] == "test")
        next(r for r in manifest["images"] if r["split"] == "train")["group_id"] = test_group
        self.write_manifest(manifest)
        self.assertTrue(any("now also appears in train" in p for p in frozen.verify_test_lock(path, self.dataset)))

    def test_freeze_refuses_a_group_shared_with_another_split(self):
        manifest = self.manifest()
        train_group = next(r["group_id"] for r in manifest["images"] if r["split"] == "train")
        next(r for r in manifest["images"] if r["split"] == "test")["group_id"] = train_group
        self.write_manifest(manifest)
        with self.assertRaisesRegex(frozen.FrozenSetError, "appears in test and another partition"):
            self.freeze()

    def test_freeze_refuses_a_test_image_identical_to_a_train_image(self):
        manifest = self.manifest()
        train = next(r for r in manifest["images"] if r["split"] == "train")
        test = next(r for r in manifest["images"] if r["split"] == "test")
        shutil.copyfile(self.dataset / train["output"], self.dataset / test["output"])
        with self.assertRaisesRegex(frozen.FrozenSetError, "byte-identical"):
            self.freeze()

    def test_freeze_refuses_unlisted_test_images(self):
        (self.dataset / "test" / "stray.jpg").write_bytes(b"stray")
        with self.assertRaisesRegex(frozen.FrozenSetError, "absent from the manifest"):
            self.freeze()

    def test_freeze_refuses_unlisted_non_test_images_including_test_copies(self):
        test = next(row for row in self.manifest()["images"] if row["split"] == "test")
        for split in ("train", "valid"):
            extra = self.dataset / split / "unlisted-copy.jpg"
            shutil.copyfile(self.dataset / test["output"], extra)
            with self.subTest(split=split), self.assertRaisesRegex(frozen.FrozenSetError, f"Image in {split}/ is absent"):
                self.freeze()
            extra.unlink()

    def test_freeze_refuses_a_blocked_audit_and_records_a_passing_one(self):
        report = self.dataset / "audit_report.json"
        report.write_text(json.dumps({"status": "BLOCKED"}))
        with self.assertRaisesRegex(frozen.FrozenSetError, "BLOCKED"):
            self.freeze()

        report.write_text(
            json.dumps(
                {
                    "status": "NEEDS_REVIEW",
                    "labels": {"counts": {"test": {"status": "COUNTED", "classes": {"good": {"objects": 3}}}}},
                    "warnings": ["test: class 'rare' occurs in 2 images (review threshold: 10).", "train: other"],
                }
            )
        )
        lock = self.freeze()
        self.assertEqual(lock["audit_status"], "NEEDS_REVIEW")
        self.assertEqual(lock["test_class_coverage"]["classes"]["good"]["objects"], 3)
        self.assertEqual(len(lock["test_audit_warnings"]), 1)

    def test_version_label_must_be_file_safe(self):
        for bad in ("", "../x", "a/b", "has space"):
            with self.assertRaises(frozen.FrozenSetError):
                self.freeze(bad)

    def test_guard_blocks_test_content_even_if_renamed_and_allows_other_data(self):
        lock = self.freeze()
        manifest = self.manifest()
        test = next(r for r in manifest["images"] if r["split"] == "test")
        train = next(r for r in manifest["images"] if r["split"] == "train")

        renamed_copy = self.tmp / "innocent_name.jpg"
        shutil.copyfile(self.dataset / test["output"], renamed_copy)

        frozen.assert_excluded(lock, paths=[self.dataset / train["output"]], group_ids=[train["group_id"]])
        with self.assertRaises(frozen.FrozenSetLeakError):
            frozen.assert_excluded(lock, paths=[renamed_copy])
        with self.assertRaises(frozen.FrozenSetLeakError):
            frozen.assert_excluded(lock, group_ids=[test["group_id"]])

    def test_self_consistent_but_malformed_lock_is_rejected(self):
        original = self.freeze()
        variations = []
        for key, value in (
            ("schema_version", 2), ("schema_version", True),
            ("files", []), ("files", [None]), ("groups", [None]),
            ("groups", [original["groups"][0], original["groups"][0]]),
            ("counts", {"images": 999, "groups": 1}),
        ):
            malformed = copy.deepcopy(original)
            malformed[key] = value
            variations.append(malformed)
        for field, value in (("sha256", "not-a-hash"), ("output", "../outside.jpg"), ("group_id", "")):
            malformed = copy.deepcopy(original)
            malformed["files"][0][field] = value
            variations.append(malformed)
        malformed = copy.deepcopy(original)
        malformed["files"].append(copy.deepcopy(malformed["files"][0]))
        variations.append(malformed)
        path = self.tmp / "malformed-lock.json"
        for malformed in variations:
            malformed["lock_hash"] = frozen._lock_hash(malformed["files"], malformed["groups"])
            path.write_text(json.dumps(malformed))
            with self.subTest(lock=malformed), self.assertRaises(frozen.FrozenSetError):
                frozen.load_test_lock(path)

    def test_verify_detects_manifest_membership_and_group_drift(self):
        path = frozen.save_test_lock(self.freeze(), self.locks)
        manifest = self.manifest()
        test_rows = [row for row in manifest["images"] if row["split"] == "test"]
        test_rows[0]["group_id"] = "changed-group"
        manifest["images"].remove(test_rows[1])
        self.write_manifest(manifest)
        problems = frozen.verify_test_lock(path, self.dataset)
        self.assertTrue(any("Changed test group_id" in problem for problem in problems))
        self.assertTrue(any("Removed from test manifest" in problem for problem in problems))

    def test_verify_requires_the_split_manifest(self):
        path = frozen.save_test_lock(self.freeze(), self.locks)
        (self.dataset / "split_manifest.json").unlink()
        with self.assertRaisesRegex(frozen.FrozenSetError, "No split_manifest"):
            frozen.verify_test_lock(path, self.dataset)

    def test_verify_detects_test_content_copied_to_an_unlisted_training_file(self):
        lock = self.freeze()
        path = frozen.save_test_lock(lock, self.locks)
        shutil.copyfile(self.dataset / lock["files"][0]["output"], self.dataset / "train/renamed-copy.jpg")
        problems = frozen.verify_test_lock(path, self.dataset)
        self.assertTrue(any("Frozen test content appears outside test/" in problem for problem in problems))

    def test_freeze_rejects_duplicate_outputs_and_missing_training_files(self):
        original = self.manifest()
        repeated = copy.deepcopy(original)
        repeated["images"].append(copy.deepcopy(repeated["images"][0]))
        self.write_manifest(repeated)
        with self.assertRaisesRegex(frozen.FrozenSetError, "Repeated output"):
            self.freeze()
        self.write_manifest(original)
        train = next(row for row in original["images"] if row["split"] == "train")
        (self.dataset / train["output"]).unlink()
        with self.assertRaisesRegex(frozen.FrozenSetError, "Manifest image is missing"):
            self.freeze()

    def test_optional_stable_image_ids_are_locked_and_verified(self):
        manifest = self.manifest()
        for number, row in enumerate(manifest["images"]):
            row["image_id"] = f"image-{number}"
        self.write_manifest(manifest)
        lock = self.freeze()
        path = frozen.save_test_lock(lock, self.locks)
        test = next(row for row in manifest["images"] if row["split"] == "test")
        self.assertIn(test["image_id"], {entry["image_id"] for entry in lock["files"]})
        test["image_id"] = "replaced-image"
        self.write_manifest(manifest)
        self.assertTrue(any("Changed test image_id" in problem for problem in frozen.verify_test_lock(path, self.dataset)))

    def test_training_guard_reads_only_the_supplied_non_test_files(self):
        lock = self.freeze()
        train = next(row for row in self.manifest()["images"] if row["split"] == "train")
        train_path = self.dataset / train["output"]
        with patch.object(frozen, "sha256_of", wraps=frozen.sha256_of) as hashed:
            frozen.assert_excluded(lock, paths=[train_path], group_ids=[train["group_id"]])
        self.assertEqual([call.args[0] for call in hashed.call_args_list], [train_path])

    def test_training_guard_rejects_missing_files_and_malformed_locks(self):
        with self.assertRaisesRegex(frozen.FrozenSetError, "missing or non-file"):
            frozen.assert_excluded(self.freeze(), paths=[self.tmp / "missing.jpg"])
        with self.assertRaises(frozen.FrozenSetError):
            frozen.assert_excluded({"files": [], "groups": []})

    def test_command_line_freeze_verify_and_check(self):
        self.assertEqual(frozen.main(["freeze", str(self.dataset), str(self.locks), "--version", "test-v1"]), 0)
        lock_path = self.locks / "test-v1.json"
        self.assertEqual(frozen.main(["verify", str(lock_path), str(self.dataset)]), 0)

        test = next(r for r in self.manifest()["images"] if r["split"] == "test")
        train = next(r for r in self.manifest()["images"] if r["split"] == "train")
        self.assertEqual(frozen.main(["check", str(lock_path), str(self.dataset / train["output"])]), 0)
        self.assertEqual(frozen.main(["check", str(lock_path), str(self.dataset / test["output"])]), 1)

        (self.dataset / test["output"]).write_bytes(b"changed")
        self.assertEqual(frozen.main(["verify", str(lock_path), str(self.dataset)]), 1)

        with self.assertRaises(SystemExit) as raised:  # re-freezing the same version is bad input
            frozen.main(["freeze", str(self.dataset), str(self.locks), "--version", "test-v1"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
