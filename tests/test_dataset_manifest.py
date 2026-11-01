import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.dataset.manifest import (
    COCO_ASSOCIATION_RULE,
    SCHEMA_VERSION,
    build_manifest,
    manifest_json,
)
from app.dataset.records import resolve_dataset_path
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class DatasetManifestTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset = load_dataset_fixture(Path(temporary_directory.name))

    def test_accepted_images_are_included_and_rejected_records_are_excluded(self):
        before = self.file_bytes()
        manifest = build_manifest(self.dataset, "v1")
        encoded = manifest_json(manifest)
        document = json.loads(encoded)

        self.assertEqual(self.file_bytes(), before)
        self.assertEqual(manifest.schema_version, SCHEMA_VERSION)
        self.assertEqual(document["schema_version"], SCHEMA_VERSION)
        self.assertEqual(document["dataset_version"], "v1")
        self.assertEqual(document["content_hash"], manifest.content_hash)
        self.assertEqual(document["coco_association"]["rule"], COCO_ASSOCIATION_RULE)
        self.assertIn("coco_image_id", document["coco_association"]["note"])
        self.assertEqual(
            [image.image_id for image in manifest.images],
            ["fix-001", "fix-002", "fix-003"],
        )
        self.assertNotIn("not-a-real-capture", encoded)
        self.assertNotIn("unreadable_image", encoded)
        self.assertNotIn("missing_annotation", encoded)

        rejected = self.dataset.rejected[0]
        renamed = replace(self.dataset, rejected=(replace(rejected, image_id="rej-1"),))
        self.assertEqual(
            build_manifest(renamed, "v1").content_hash,
            manifest.content_hash,
        )

    def test_negative_image_keeps_missing_provenance_and_unknown_values(self):
        manifest = build_manifest(self.dataset, "v1")
        negative = self.image(manifest, "fix-003")
        document = json.loads(manifest_json(manifest))
        encoded_negative = next(
            item for item in document["images"] if item["image_id"] == "fix-003"
        )

        self.assertEqual(negative.coco_image_id, 3)
        self.assertIsNone(negative.source_lot)
        self.assertEqual(negative.source_session, "session-2")
        self.assertIsNone(negative.near_duplicate_group_id)
        self.assertFalse(negative.split_eligible)
        self.assertEqual(negative.split_blockers, ("source_lot",))
        self.assertIsNone(encoded_negative["source_lot"])
        self.assertIsNone(encoded_negative["near_duplicate_group_id"])
        self.assertEqual(encoded_negative["split_blockers"], ["source_lot"])
        self.assertNotIn("unknown", manifest_json(manifest))

    def test_supplied_groups_and_relative_paths_resolve_from_the_root(self):
        manifest = build_manifest(self.dataset, "v1")
        first = self.image(manifest, "fix-001")
        second = self.image(manifest, "fix-002")
        record = next(item for item in self.dataset.accepted if item.image_id == "fix-001")

        self.assertEqual(first.near_duplicate_group_id, "dup-1")
        self.assertEqual(second.near_duplicate_group_id, "dup-1")
        self.assertEqual(first.source_lot, "lot-1")
        self.assertEqual(first.source_session, "session-1")
        self.assertTrue(first.split_eligible)
        self.assertEqual(first.split_blockers, ())
        self.assertEqual(first.image_path, "images/two_a.png")
        self.assertEqual(first.annotation_path, "annotations/instances.json")
        resolved = resolve_dataset_path(self.dataset.root, first.image_path)
        self.assertEqual(resolved, self.dataset.resolve(record.image_path))
        self.assertTrue(resolved.is_file())
        encoded = manifest_json(manifest)
        self.assertNotIn(self.dataset.root.as_posix(), encoded)
        self.assertNotIn("\\", encoded)

    def test_repeated_input_has_identical_serialization_and_hash(self):
        first = build_manifest(self.dataset, "v1")
        second = build_manifest(self.dataset, "v1")
        self.assertEqual(manifest_json(first), manifest_json(second))
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertRegex(first.content_hash, r"^[0-9a-f]{64}$")

        parsed = self.dataset.annotations[0]
        reordered = replace(
            self.dataset,
            accepted=tuple(reversed(self.dataset.accepted)),
            annotations=(replace(parsed, objects=tuple(reversed(parsed.objects))),),
        )
        self.assertEqual(
            manifest_json(build_manifest(reordered, "v1")),
            manifest_json(first),
        )

    def test_relocation_preserves_the_hash(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        moved = load_dataset_fixture(Path(temporary_directory.name))
        self.assertNotEqual(self.dataset.root, moved.root)

        first = build_manifest(self.dataset, "v1")
        second = build_manifest(moved, "v1")
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(manifest_json(first), manifest_json(second))
        self.assertNotIn(moved.root.as_posix(), manifest_json(second))

    def test_version_label_does_not_change_the_content_hash(self):
        first = build_manifest(self.dataset, "2026.1")
        second = build_manifest(self.dataset, "2026.2")
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertNotEqual(manifest_json(first), manifest_json(second))
        self.assertIn("2026.1", manifest_json(first))
        self.assertNotIn("2026.2", manifest_json(first))
        with self.assertRaises(ValueError):
            build_manifest(self.dataset, "   ")

    def test_provenance_or_annotation_changes_alter_the_hash(self):
        original = build_manifest(self.dataset, "v1")

        changed_lot = tuple(
            replace(image, source_lot="lot-9") if image.image_id == "fix-001" else image
            for image in self.dataset.accepted
        )
        provenance = replace(self.dataset, accepted=changed_lot)
        self.assertNotEqual(build_manifest(provenance, "v1").content_hash, original.content_hash)

        parsed = self.dataset.annotations[0]
        changed_objects = tuple(
            replace(obj, bbox_xywh=(1.0, 2.0, 3.0, 4.0)) if obj.annotation_id == 11 else obj
            for obj in parsed.objects
        )
        annotated = replace(self.dataset, annotations=(replace(parsed, objects=changed_objects),))
        annotated_manifest = build_manifest(annotated, "v1")
        self.assertNotEqual(annotated_manifest.content_hash, original.content_hash)
        self.assertEqual(
            [image.image_id for image in annotated_manifest.images],
            [image.image_id for image in original.images],
        )

    def test_image_file_bytes_are_not_part_of_the_hash(self):
        original = build_manifest(self.dataset, "v1")
        image_path = self.dataset.resolve("images/two_a.png")
        image_path.write_bytes(image_path.read_bytes() + b"tamper")
        self.assertEqual(build_manifest(self.dataset, "v1").content_hash, original.content_hash)

    def image(self, manifest, image_id):
        return next(item for item in manifest.images if item.image_id == image_id)

    def file_bytes(self):
        return {
            path.relative_to(self.dataset.root).as_posix(): path.read_bytes()
            for path in self.dataset.root.rglob("*")
            if path.is_file()
        }


if __name__ == "__main__":
    unittest.main()
