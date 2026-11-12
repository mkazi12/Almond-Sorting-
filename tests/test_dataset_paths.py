"""Portable paths and real filesystem containment use the same shared rules."""

import tempfile
import unittest
from pathlib import Path

from app.dataset.paths import normalize_relative_path
from app.dataset.records import DatasetPathError, relative_path_identity, resolve_dataset_path
from app.dataset.validation import DatasetValidationError, require_inside_root, safe_relative_path


class DatasetPathTests(unittest.TestCase):
    def test_both_slash_conventions_have_one_identity(self):
        for value in ("images/a.png", "images\\a.png", "./images//a.png"):
            self.assertEqual(relative_path_identity(value), "images/a.png")
            self.assertEqual(safe_relative_path(value), "images/a.png")
        self.assertEqual(normalize_relative_path(".", allow_root=True), ".")

    def test_windows_and_posix_escape_paths_are_rejected_on_every_host(self):
        bad = (
            "", ".", "../a.png", "images/../../a.png", "/tmp/a.png",
            "C:\\a.png", "C:a.png", "\\\\server\\share\\a.png",
            "images\\..\\a.png", "images/a.png:stream", "a\x00.png",
        )
        for value in bad:
            with self.subTest(value=value):
                with self.assertRaises(DatasetPathError):
                    relative_path_identity(value)
                with self.assertRaises(DatasetValidationError) as caught:
                    safe_relative_path(value)
                self.assertEqual(caught.exception.category, "unsafe_path")

    def test_outward_symlinks_are_rejected_by_records_and_import_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "dataset"
            root.mkdir()
            outside = base / "outside.png"
            outside.write_bytes(b"outside")
            try:
                (root / "linked.png").symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable")
            with self.assertRaises(DatasetPathError):
                resolve_dataset_path(root, "linked.png")
            with self.assertRaises(DatasetValidationError) as caught:
                require_inside_root(root, "linked.png")
            self.assertEqual(caught.exception.category, "unsafe_path")

    def test_inward_symlinks_and_missing_relative_files_keep_their_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_dataset_path(root, "images\\future.png"), root / "images/future.png")
            (root / "actual.png").write_bytes(b"inside")
            try:
                (root / "alias.png").symlink_to(root / "actual.png")
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable")
            self.assertEqual(resolve_dataset_path(root, "alias.png"), root / "actual.png")

    def test_symlink_cycles_use_the_shared_path_error_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                (root / "one.png").symlink_to(root / "two.png")
                (root / "two.png").symlink_to(root / "one.png")
            except (OSError, NotImplementedError):
                self.skipTest("Symlink creation unavailable")
            with self.assertRaises(DatasetPathError):
                resolve_dataset_path(root, "one.png")
            with self.assertRaises(DatasetValidationError) as caught:
                require_inside_root(root, "two.png")
            self.assertEqual(caught.exception.category, "unsafe_path")


if __name__ == "__main__":
    unittest.main()
