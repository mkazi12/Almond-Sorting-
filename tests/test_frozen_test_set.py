import json
import shutil
import stat
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

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