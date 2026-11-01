import json
import os
import tempfile
import unittest
from pathlib import Path

from app.dataset.records import (
    AcceptedImage,
    ClassSpec,
    CocoImage,
    DatasetInput,
    DatasetPathError,
    GroundTruthObject,
    ParsedAnnotations,
    SourcePolicy,
    UNKNOWN_SOURCE,
    canonical_content,
    coco_image_for,
    eligible_for_split,
    missing_provenance_fields,
    objects_for_image,
    resolve_dataset_path,
    source_label,
)


def accepted_image(**overrides) -> AcceptedImage:
    values = {
        "image_id": "img-1",
        "coco_image_id": 1,
        "image_path": Path("images/a.png"),
        "annotation_path": Path("annotations/one.json"),
        "width": 8,
        "height": 8,
        "source_lot": "lot-1",
        "source_session": "session-1",
        "near_duplicate_group_id": "dup-1",
    }
    values.update(overrides)
    return AcceptedImage(**values)


class DatasetRecordTests(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)

    def test_paths_resolve_from_the_dataset_root(self):
        root = self.directory
        resolved = resolve_dataset_path(root, "images/two_a.png")
        self.assertEqual(resolved, (root / "images" / "two_a.png").resolve())

        bad_paths = [
            "../outside.png",
            "/outside.png",
            str((root / "outside.png").resolve()),
            ".",
            "",
        ]
        if os.name == "nt":
            bad_paths.append("C:/outside.png")
        for bad_path in bad_paths:
            with self.subTest(path=bad_path):
                with self.assertRaises(DatasetPathError):
                    resolve_dataset_path(root, bad_path)

    def test_missing_provenance_follows_the_source_policy(self):
        image = accepted_image(source_lot="  ", source_session=None)
        self.assertIsNone(image.source_lot)
        self.assertIsNone(image.source_session)
        self.assertEqual(source_label(image.source_lot), UNKNOWN_SOURCE)
        self.assertEqual(source_label(" lot-1 "), "lot-1")

        required = SourcePolicy(require_lot=True, require_session=True)
        lot_only = SourcePolicy(require_lot=True, require_session=False)
        optional = SourcePolicy(require_lot=False, require_session=False)
        self.assertEqual(
            missing_provenance_fields(image, required),
            ("source_lot", "source_session"),
        )
        self.assertEqual(missing_provenance_fields(image, lot_only), ("source_lot",))
        self.assertEqual(missing_provenance_fields(image, optional), ())
        self.assertFalse(eligible_for_split(image, required))
        self.assertTrue(eligible_for_split(image, optional))

    def test_join_uses_annotation_file_and_coco_image_id(self):
        image = accepted_image(coco_image_id=7, annotation_path=Path("annotations/one.json"))
        matching = GroundTruthObject(1, 7, 1, (0, 0, 2, 2))
        other_file = GroundTruthObject(2, 7, 2, (3, 3, 2, 2))
        dataset = DatasetInput(
            root=self.directory,
            source_policy=SourcePolicy(require_lot=True, require_session=True),
            accepted=(image,),
            rejected=(),
            annotations=(
                ParsedAnnotations(
                    path=Path("annotations/one.json"),
                    classes=(ClassSpec(1, "A"),),
                    images=(CocoImage(7, "images/a.png", 8, 8),),
                    objects=(matching,),
                ),
                ParsedAnnotations(
                    path=Path("annotations/two.json"),
                    classes=(ClassSpec(2, "B"),),
                    images=(CocoImage(7, "images/other.png", 8, 8),),
                    objects=(other_file,),
                ),
            ),
        )

        joined = objects_for_image(dataset, dataset.accepted[0])
        self.assertEqual(joined, (matching,))
        entry = coco_image_for(dataset, dataset.accepted[0])
        self.assertIsNotNone(entry)
        self.assertEqual(entry.file_name, "images/a.png")

    def test_canonical_content_omits_the_dataset_root(self):
        first = self.sample_dataset(self.directory / "one")
        second = self.sample_dataset(self.directory / "two")
        self.assertNotEqual(first.root, second.root)
        self.assertEqual(canonical_content(first), canonical_content(second))
        self.assertEqual(
            [image.image_id for image in first.accepted],
            [image.image_id for image in second.accepted],
        )
        self.assertNotEqual(first.resolve("images/a.png"), second.resolve("images/a.png"))
        encoded = json.dumps(canonical_content(first))
        self.assertNotIn(first.root.as_posix(), encoded)
        self.assertNotIn("dataset_version", encoded)
        self.assertNotIn("content_hash", encoded)

    def test_accepted_image_ids_stay_unique(self):
        image = accepted_image()
        with self.assertRaises(ValueError):
            DatasetInput(
                root=self.directory,
                source_policy=SourcePolicy(require_lot=True, require_session=True),
                accepted=(image, accepted_image()),
                rejected=(),
                annotations=(),
            )

    def sample_dataset(self, root: Path) -> DatasetInput:
        root.mkdir()
        image = accepted_image()
        return DatasetInput(
            root=root,
            source_policy=SourcePolicy(require_lot=True, require_session=True),
            accepted=(image,),
            rejected=(),
            annotations=(
                ParsedAnnotations(
                    path=image.annotation_path,
                    classes=(ClassSpec(1, "A"),),
                    images=(CocoImage(1, image.image_path, image.width, image.height),),
                    objects=(),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
