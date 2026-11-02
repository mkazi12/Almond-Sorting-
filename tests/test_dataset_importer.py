"""SCRUM-42 integration tests with small local COCO fixtures, no trained model."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.dataset.importer import import_coco_dataset
from app.dataset.manifest import build_manifest
from app.dataset.records import SourcePolicy, canonical_content, eligible_for_split, objects_for_image
from app.dataset.summary import summarize
from app.dataset.validation import DatasetValidationError
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class DatasetImporterTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "dataset"
        (self.root / "images").mkdir(parents=True)
        (self.root / "annotations").mkdir()
        for name in ("valid.png", "negative.jpg", "badbox.png", "unlisted.png"):
            Image.new("RGB", (80, 60), (150, 100, 70)).save(self.root / "images" / name)
        self.document = {
            "images": [
                {"id": 1, "file_name": "images/valid.png", "width": 80, "height": 60},
                {"id": 2, "file_name": "images/negative.jpg", "width": 80, "height": 60},
                {"id": 3, "file_name": "images/badbox.png", "width": 80, "height": 60},
            ],
            "annotations": [
                {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 10]},
                {"id": 11, "image_id": 1, "category_id": 2, "bbox": [30, 10, 20.5, 12]},
                {"id": 12, "image_id": 3, "category_id": 2, "bbox": [79, 10, 5, 20]},
            ],
            "categories": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        }
        self.write_json()

    def write_json(self):
        (self.root / "annotations" / "instances.json").write_text(json.dumps(self.document), encoding="utf-8")

    def importer(self, **overrides):
        args = dict(
            annotation_file="annotations/instances.json",
            class_mapping={1: "A", 2: "B"},
            dataset_namespace="test-fixture-A",
            image_directory="images",
            allow_negative_images=True,
        )
        args.update(overrides)
        return import_coco_dataset(self.root, **args)

    def test_accepted_rejected_and_summary_handoff(self):
        dataset = self.importer()
        self.assertEqual(len(dataset.accepted), 2)
        self.assertEqual(len(dataset.rejected), 2)
        self.assertEqual({p.image_path.as_posix() for p in dataset.accepted}, {"images/valid.png", "images/negative.jpg"})
        reasons = {r.image_path.name: [reason.category for reason in r.reasons] for r in dataset.rejected}
        self.assertEqual(reasons["badbox.png"], ["invalid_bbox"])
        self.assertEqual(reasons["unlisted.png"], ["missing_annotation"])
        summary = summarize(dataset)
        self.assertEqual((summary.accepted_image_count, summary.rejected_image_count, summary.negative_image_count), (2, 2, 1))
        self.assertEqual({c.name: (c.annotation_count, c.image_count) for c in summary.classes}, {"A": (1, 1), "B": (1, 1)})
        manifest = build_manifest(dataset, "v1")
        self.assertEqual(len(manifest.images), 2)
        self.assertTrue(all(not item.split_eligible for item in manifest.images))
        valid = next(img for img in dataset.accepted if img.image_path.name == "valid.png")
        self.assertEqual(len(objects_for_image(dataset, valid)), 2)

    def test_repeat_import_root_relocation_and_unmodified_source(self):
        before = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.root.rglob("*") if p.is_file()}
        first = self.importer()
        second = self.importer()
        self.assertEqual(canonical_content(first), canonical_content(second))
        self.assertEqual(len({item.image_id for item in second.accepted}), len(second.accepted))
        after = {p.relative_to(self.root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        copied = self.root.parent / "another_location"
        shutil.copytree(self.root, copied)
        moved = import_coco_dataset(copied, annotation_file="annotations/instances.json",
                                   class_mapping={1: "A", 2: "B"}, dataset_namespace="test-fixture-A",
                                   image_directory="images", allow_negative_images=True)
        self.assertEqual(canonical_content(first), canonical_content(moved))

    def test_provenance_is_preserved_and_missing_data_blocks_splits(self):
        dataset = self.importer(
            provenance_by_image={"images/valid.png": {"source_lot": "lot-1", "source_session": "run-1", "near_duplicate_group_id": "dupe-2"}},
            source_policy=SourcePolicy(require_lot=True, require_session=True),
        )
        valid = next(img for img in dataset.accepted if img.image_path.name == "valid.png")
        negative = next(img for img in dataset.accepted if img.image_path.name == "negative.jpg")
        self.assertEqual((valid.source_lot, valid.source_session, valid.near_duplicate_group_id), ("lot-1", "run-1", "dupe-2"))
        self.assertTrue(eligible_for_split(valid, dataset.source_policy))
        self.assertFalse(eligible_for_split(negative, dataset.source_policy))

    def test_actual_zakari_fixture_can_be_imported_and_summarized(self):
        fixture_root = self.root.parent / "zakari_fixture"
        load_dataset_fixture(fixture_root)  # generates authored placeholders, not actual almonds
        result = import_coco_dataset(
            fixture_root,
            annotation_file="annotations/instances.json",
            class_mapping={1: "A", 2: "B"},
            dataset_namespace="zakari-test-fixture",
            image_directory="images",
            allow_negative_images=True,
        )
        summary = summarize(result)
        self.assertEqual((summary.accepted_image_count, summary.rejected_image_count, summary.negative_image_count), (3, 0, 1))
        self.assertEqual({item.name: (item.annotation_count, item.image_count) for item in summary.classes}, {"A": (3, 2), "B": (1, 1)})
        self.assertEqual(len(build_manifest(result, "fixture-v1").images), 3)

    def test_negative_image_policy(self):
        dataset = self.importer(allow_negative_images=False)
        self.assertEqual(len(dataset.accepted), 1)
        self.assertEqual(len(dataset.rejected), 3)
        self.assertEqual({reason.category for row in dataset.rejected for reason in row.reasons if row.image_path.name == "negative.jpg"}, {"negative_image_not_allowed"})

    def test_unknown_class_rejects_one_image_not_others(self):
        self.document["annotations"][0]["category_id"] = 999
        self.write_json()
        dataset = self.importer()
        self.assertEqual({r.image_path.name for r in dataset.rejected}, {"valid.png", "badbox.png", "unlisted.png"})
        target = next(r for r in dataset.rejected if r.image_path.name == "valid.png")
        self.assertEqual(target.reasons[0].category, "unknown_class_id")
        self.assertEqual(summarize(dataset).negative_image_count, 1)

    def test_unreadable_unsupported_and_missing_sources(self):
        (self.root / "images" / "valid.png").write_bytes(b"corrupted")
        Image.new("RGB", (80, 60)).save(self.root / "images" / "extra.bmp")
        (self.root / "images" / "badbox.png").unlink()
        dataset = self.importer()
        codes = {r.image_path.name: {reason.category for reason in r.reasons} for r in dataset.rejected}
        self.assertIn("unreadable_image", codes["valid.png"])
        self.assertIn("unsupported_image", codes["extra.bmp"])
        self.assertIn("unreadable_image", codes["badbox.png"])

    def test_document_level_errors_are_not_silent_negatives(self):
        ann = self.root / "annotations" / "instances.json"
        ann.unlink()
        with self.assertRaises(DatasetValidationError) as missing:
            self.importer()
        self.assertEqual(missing.exception.category, "missing_annotation_file")
        ann.write_text("{bad json", encoding="utf-8")
        with self.assertRaises(DatasetValidationError) as broken:
            self.importer()
        self.assertEqual(broken.exception.category, "malformed_annotation_file")

    def test_dataset_root_and_image_path_safety(self):
        with self.assertRaises(DatasetValidationError) as caught:
            self.importer(annotation_file="../outside.json")
        self.assertEqual(caught.exception.category, "unsafe_path")
        self.document["images"][0]["file_name"] = "C:\\secret.jpg"
        self.write_json()
        with self.assertRaises(DatasetValidationError) as caught:
            self.importer()
        self.assertEqual(caught.exception.category, "unsafe_path")

    def test_image_base_for_flat_coco_files(self):
        for entry in self.document["images"]:
            entry["file_name"] = entry["file_name"].replace("images/", "")
        self.write_json()
        dataset = self.importer(coco_filename_base="images")
        self.assertEqual(len(dataset.accepted), 2)
        self.assertEqual(summarize(dataset).negative_image_count, 1)

    def test_namespace_changes_identity_while_version_label_does_not(self):
        a = self.importer()
        b = self.importer(dataset_namespace="another-dataset")
        self.assertNotEqual({r.image_id for r in a.accepted}, {r.image_id for r in b.accepted})
        self.assertEqual(build_manifest(a, "v1").content_hash, build_manifest(a, "v2").content_hash)


if __name__ == "__main__":
    unittest.main()
