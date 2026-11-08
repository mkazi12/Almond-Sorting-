"""Regression checks for the raw-image preparation utility."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from app.dataset.splitting import main, split_dataset
from app.split_dataset import split_dataset as legacy_split_dataset


def make_source(root, count=10):
    source = root / "source"
    for number in range(count):
        group = source / f"recording-{number:02d}"
        (group / "nested").mkdir(parents=True)
        (group / "frame.JPG").write_bytes(f"recording-{number}-first".encode())
        (group / "nested" / "frame.png").write_bytes(f"recording-{number}-second".encode())
        (group / "labels.json").write_text('{"not": "a training export"}')
    return source


class RawSplitTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = make_source(self.root)

    def test_deterministic_groups_and_every_image_copied_once_with_unchanged_bytes(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()}
        first = split_dataset(self.source, self.root / "first", seed=1)
        second = split_dataset(self.source, self.root / "second", seed=1)
        self.assertIs(legacy_split_dataset, split_dataset)
        self.assertEqual(first, second)
        self.assertEqual({key: len(groups) for key, groups in first["groups"].items()},
                         {"train": 8, "valid": 1, "test": 1})
        self.assertEqual(len(first["images"]), 20)
        self.assertEqual(len({row["source"] for row in first["images"]}), 20)
        self.assertEqual(len({row["output"] for row in first["images"]}), 20)
        ownership = {}
        for row in first["images"]:
            self.assertEqual(ownership.setdefault(row["group_id"], row["split"]), row["split"])
            self.assertEqual((self.root / "first" / row["output"]).read_bytes(),
                             (self.source / row["source"]).read_bytes())
        self.assertEqual(before, {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*") if p.is_file()})
        self.assertFalse(list((self.root / "first").rglob("labels.json")))
        self.assertEqual(json.loads((self.root / "first" / "split_manifest.json").read_text()), first)

    def test_three_independent_groups_leave_each_split_nonempty(self):
        source = make_source(self.root / "small", count=3)
        manifest = split_dataset(source, self.root / "small-output")
        self.assertEqual([len(groups) for groups in manifest["groups"].values()], [1, 1, 1])

    def test_invalid_layout_or_destination_does_not_create_output(self):
        output = self.root / "output"
        (self.source / "loose.jpg").write_bytes(b"outside-a-group")
        with self.assertRaisesRegex(ValueError, "inside its batch"):
            split_dataset(self.source, output)
        self.assertFalse(output.exists())
        (self.source / "loose.jpg").unlink()
        with self.assertRaisesRegex(ValueError, "outside the source"):
            split_dataset(self.source, self.source / "nested-output")
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("keep")
        with self.assertRaises(ValueError):
            split_dataset(self.source, output)
        self.assertEqual(sentinel.read_text(), "keep")

    def test_empty_group_is_identified(self):
        (self.source / "empty-recording").mkdir()
        with self.assertRaisesRegex(ValueError, "empty-recording"):
            split_dataset(self.source, self.root / "output")

    def test_copy_failure_leaves_no_published_or_temporary_dataset(self):
        output = self.root / "output"
        with patch("app.dataset.splitting.shutil.copy2", side_effect=OSError("copy failed")):
            with self.assertRaisesRegex(OSError, "copy failed"):
                split_dataset(self.source, output)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob("image_split_*")))

    def test_external_symlink_is_rejected_before_copying(self):
        outside = self.root / "outside.jpg"
        outside.write_bytes(b"outside")
        link = self.source / "recording-00" / "escape.jpg"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("Symbolic links are unavailable")
        with self.assertRaisesRegex(ValueError, "external path"):
            split_dataset(self.source, self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_invalid_seed_fails_before_writing(self):
        for seed in (True, -1, 2**32, "1"):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "Seed"):
                split_dataset(self.source, self.root / "output", seed=seed)
        self.assertFalse((self.root / "output").exists())

    def test_command_retains_arguments_and_identifies_raw_output(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main([str(self.source), str(self.root / "output"), "--seed", "2"]), 0)
        self.assertIn("Raw image split only", output.getvalue())
        self.assertTrue((self.root / "output" / "split_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
