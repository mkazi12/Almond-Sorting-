import json
import struct
import tempfile
import unittest
from pathlib import Path

from app.dataset.records import (
    UNKNOWN_SOURCE,
    SourcePolicy,
    canonical_content,
    coco_image_for,
    eligible_for_split,
    missing_provenance_fields,
    objects_for_image,
    relative_path_identity,
    source_label,
)
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class DatasetFixtureTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.dataset = load_dataset_fixture(Path(temporary_directory.name))

    def image(self, image_id):
        return next(item for item in self.dataset.accepted if item.image_id == image_id)

    def test_shared_coco_file_joins_each_accepted_image(self):
        parsed = self.dataset.annotations
        self.assertEqual(len(parsed), 1)
        self.assertEqual(relative_path_identity(parsed[0].path), "annotations/instances.json")
        classes = {item.category_id: item.name for item in parsed[0].classes}
        self.assertEqual(classes, {1: "A", 2: "B"})

        joined = {}
        for image_id in ("fix-001", "fix-002", "fix-003"):
            image = self.image(image_id)
            self.assertEqual(
                relative_path_identity(image.annotation_path),
                "annotations/instances.json",
            )
            joined[image_id] = tuple(
                (obj.annotation_id, classes[obj.category_id])
                for obj in objects_for_image(self.dataset, image)
            )

        self.assertEqual(joined["fix-001"], ((11, "A"), (12, "A")))
        self.assertEqual(joined["fix-002"], ((21, "A"), (22, "B")))
        self.assertEqual(joined["fix-003"], ())

    def test_negative_image_has_a_coco_entry_and_no_objects(self):
        negative = self.image("fix-003")
        parsed = self.dataset.annotations[0]
        self.assertTrue(any(item.coco_image_id == 3 for item in parsed.images))
        self.assertFalse(any(obj.coco_image_id == 3 for obj in parsed.objects))

        entry = coco_image_for(self.dataset, negative)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.coco_image_id, negative.coco_image_id)
        self.assertEqual(entry.file_name, "images/negative.png")
        self.assertEqual((entry.width, entry.height), (negative.width, negative.height))
        self.assertEqual(objects_for_image(self.dataset, negative), ())
        self.assertIn(negative.image_id, [item.image_id for item in self.dataset.accepted])

    def test_missing_lot_stays_accepted_and_ineligible_for_split(self):
        negative = self.image("fix-003")
        policy = self.dataset.source_policy
        self.assertEqual(policy, SourcePolicy(require_lot=True, require_session=True))
        self.assertIsNone(negative.source_lot)
        self.assertEqual(negative.source_session, "session-2")
        self.assertIsNone(negative.near_duplicate_group_id)
        self.assertEqual(missing_provenance_fields(negative, policy), ("source_lot",))
        self.assertEqual(source_label(negative.source_lot), UNKNOWN_SOURCE)
        self.assertEqual(source_label(negative.source_session), "session-2")
        self.assertFalse(eligible_for_split(negative, policy))
        self.assertTrue(eligible_for_split(self.image("fix-001"), policy))
        self.assertTrue(eligible_for_split(self.image("fix-002"), policy))
        self.assertEqual(
            [item.image_id for item in self.dataset.accepted],
            ["fix-001", "fix-002", "fix-003"],
        )

    def test_one_rejected_image_keeps_every_reason(self):
        self.assertEqual(len(self.dataset.rejected), 1)
        rejected = self.dataset.rejected[0]
        self.assertIsNone(rejected.image_id)
        self.assertEqual(
            relative_path_identity(rejected.image_path),
            "images/not-a-real-capture.jpg",
        )
        self.assertEqual(
            [reason.category for reason in rejected.reasons],
            ["unreadable_image", "missing_annotation"],
        )
        reason_count = sum(len(item.reasons) for item in self.dataset.rejected)
        self.assertEqual(reason_count, 2)
        self.assertEqual(len(self.dataset.rejected), 1)
        self.assertFalse(self.dataset.resolve(rejected.image_path).is_file())

    def test_placeholder_images_match_the_accepted_records(self):
        for image in self.dataset.accepted:
            path = self.dataset.resolve(image.image_path)
            self.assert_png(path, image.width, image.height)

    def test_moving_the_root_keeps_ids_and_canonical_content(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        moved = load_dataset_fixture(Path(temporary_directory.name))

        self.assertNotEqual(self.dataset.root, moved.root)
        self.assertEqual(
            [image.image_id for image in self.dataset.accepted],
            [image.image_id for image in moved.accepted],
        )
        self.assertEqual(canonical_content(self.dataset), canonical_content(moved))
        moved_image = next(item for item in moved.accepted if item.image_id == "fix-001")
        self.assertNotEqual(
            self.dataset.resolve(self.image("fix-001").image_path),
            moved.resolve(moved_image.image_path),
        )
        encoded = json.dumps(canonical_content(self.dataset))
        self.assertNotIn(self.dataset.root.as_posix(), encoded)
        self.assertNotIn(str(self.dataset.root), encoded)

    def assert_png(self, path: Path, width: int, height: int) -> None:
        data = path.read_bytes()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(data[12:16], b"IHDR")
        self.assertEqual(struct.unpack(">II", data[16:24]), (width, height))


if __name__ == "__main__":
    unittest.main()
