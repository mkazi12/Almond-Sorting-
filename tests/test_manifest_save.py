import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.dataset.manifest import build_manifest
from app.dataset.manifest_save import (
    ManifestDestinationError,
    ManifestSaveError,
    save_manifest,
)
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class ManifestSaveTests(unittest.TestCase):
    def setUp(self):
        dataset_directory = tempfile.TemporaryDirectory()
        output_directory = tempfile.TemporaryDirectory()
        self.addCleanup(dataset_directory.cleanup)
        self.addCleanup(output_directory.cleanup)
        self.dataset = load_dataset_fixture(Path(dataset_directory.name))
        # save_manifest returns resolved paths. Resolve here too, because temp
        # folders are reached through links on macOS (/var -> /private/var)
        # and can use 8.3 short names on Windows (C:\Users\RUNNER~1\...).
        self.output = Path(output_directory.name).resolve()

    def test_v1_and_v2_are_separate_readable_manifests(self):
        first = save_manifest(build_manifest(self.dataset, "v1"), self.output)
        second = save_manifest(build_manifest(self.dataset, "v2"), self.output)

        self.assertEqual(first, self.output / "v1.json")
        self.assertEqual(second, self.output / "v2.json")
        self.assertNotEqual(first, second)
        self.assertEqual(json.loads(first.read_text(encoding="utf-8")), build_manifest(self.dataset, "v1").to_dict())
        self.assertEqual(json.loads(second.read_text(encoding="utf-8")), build_manifest(self.dataset, "v2").to_dict())
        self.assertEqual(json.loads(first.read_text(encoding="utf-8"))["dataset_version"], "v1")
        self.assertEqual(json.loads(second.read_text(encoding="utf-8"))["dataset_version"], "v2")

    def test_existing_version_is_rejected_without_changing_contents(self):
        saved = save_manifest(build_manifest(self.dataset, "v1"), self.output)
        original = saved.read_bytes()
        changed_images = tuple(
            replace(image, source_lot="lot-changed") if image.image_id == "fix-001" else image
            for image in self.dataset.accepted
        )
        changed = build_manifest(replace(self.dataset, accepted=changed_images), "v1")

        with self.assertRaises(ManifestDestinationError):
            save_manifest(changed, self.output)

        self.assertEqual(saved.read_bytes(), original)
        self.assertEqual([path.name for path in self.output.iterdir()], ["v1.json"])
        self.assertEqual(json.loads(original)["images"][0]["source_lot"], "lot-1")

    def test_failed_publication_leaves_previous_manifests_and_no_partial_file(self):
        saved = save_manifest(build_manifest(self.dataset, "v1"), self.output)
        original = saved.read_bytes()

        with patch("app.dataset.manifest_save._publish", side_effect=OSError("publication failed")):
            with self.assertRaises(OSError):
                save_manifest(build_manifest(self.dataset, "v2"), self.output)

        def fail_during_write(path, text):
            path.write_bytes(b"{partial")
            raise OSError("write failed")

        with patch("app.dataset.manifest_save._write_complete", side_effect=fail_during_write):
            with self.assertRaises(OSError):
                save_manifest(build_manifest(self.dataset, "v2"), self.output)

        self.assertEqual(saved.read_bytes(), original)
        self.assertFalse((self.output / "v2.json").exists())
        self.assertEqual([path.name for path in self.output.iterdir()], ["v1.json"])

    def test_unsafe_version_labels_stay_inside_the_output_directory(self):
        manifest = build_manifest(self.dataset, "v1")
        labels = ["../escape", "..\\escape", "nested/escape", "nested\\escape", "..", ".", "/escape.json", "C:/escape", "C:escape"]
        escape = self.output.parent / "escape.json"
        existed = escape.exists()

        for label in labels:
            with self.subTest(label=label):
                with self.assertRaises(ManifestSaveError):
                    save_manifest(replace(manifest, dataset_version=label), self.output)

        self.assertEqual(escape.exists(), existed)
        self.assertFalse((self.output / "nested").exists())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_source_images_and_annotations_remain_unchanged(self):
        before = self.dataset_files()
        save_manifest(build_manifest(self.dataset, "v1"), self.output)
        save_manifest(build_manifest(self.dataset, "v2"), self.output)
        self.assertEqual(self.dataset_files(), before)
        self.assertIn("annotations/instances.json", before)
        self.assertIn("images/two_a.png", before)
        self.assertIn("images/a_and_b.png", before)
        self.assertIn("images/negative.png", before)

    def dataset_files(self):
        return {
            path.relative_to(self.dataset.root).as_posix(): path.read_bytes()
            for path in self.dataset.root.rglob("*")
            if path.is_file()
        }


if __name__ == "__main__":
    unittest.main()
