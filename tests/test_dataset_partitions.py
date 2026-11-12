"""Metadata split handoff needs neither image decoding nor a database."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.dataset.manifest import build_manifest
from app.dataset.partitions import (
    PartitionError,
    build_partition_manifest,
    partition_manifest_json,
    select_development_images,
)
from app.dataset.records import (
    AcceptedImage,
    ClassSpec,
    CocoImage,
    DatasetInput,
    GroundTruthObject,
    ParsedAnnotations,
    SourcePolicy,
)


def dataset_fixture() -> DatasetInput:
    images = tuple(
        AcceptedImage(f"img-{index}", index, Path(f"images/{index}.png"), Path("labels.json"),
                      40, 30, f"lot-{group}", f"session-{group}", None)
        for index, group in ((1, 1), (2, 1), (3, 2), (4, 3))
    )
    parsed = ParsedAnnotations(
        Path("labels.json"), (ClassSpec(7, "fixture-class"),),
        tuple(CocoImage(i.coco_image_id, i.image_path.as_posix(), 40, 30) for i in images),
        (GroundTruthObject(10, 1, 7, (2, 3, 5, 6)),),
    )
    return DatasetInput(Path("not-read"), SourcePolicy(True, True), images, (), (parsed,))


class DatasetPartitionTests(unittest.TestCase):
    def setUp(self):
        self.dataset = dataset_fixture()
        self.assignments = {"img-1": "train", "img-2": "train", "img-3": "valid", "img-4": "test"}

    def build(self, dataset=None, assignments=None):
        return build_partition_manifest(self.dataset if dataset is None else dataset, "v1",
                                        self.assignments if assignments is None else assignments)

    def test_preserves_ids_labels_negatives_and_source_hash(self):
        manifest = self.build()
        document = json.loads(partition_manifest_json(manifest))
        self.assertEqual(document["kind"], "labeled_dataset_partitions")
        self.assertEqual(manifest.source_content_hash, build_manifest(self.dataset, "v1").content_hash)
        self.assertEqual(document["classes"], [{"category_id": 7, "name": "fixture-class"}])
        first, negative = manifest.images[:2]
        self.assertEqual(first.image.image_id, "img-1")
        self.assertEqual(first.image.coco_image_id, 1)
        self.assertEqual(first.objects[0].category_id, 7)
        self.assertEqual(first.objects[0].bbox_xywh, (2., 3., 5., 6.))
        self.assertEqual(first.group_id, negative.group_id)
        self.assertEqual(negative.objects, ())
        self.assertEqual(document["images"][1]["objects"], [])
        self.assertEqual(document["images"][1]["image_path"], "images/2.png")

    def test_order_and_root_relocation_do_not_change_serialization(self):
        first = self.build()
        moved = replace(self.dataset, root=Path("somewhere-else"),
                        accepted=tuple(reversed(self.dataset.accepted)))
        reversed_map = dict(reversed(list(self.assignments.items())))
        self.assertEqual(partition_manifest_json(first),
                         partition_manifest_json(self.build(moved, reversed_map)))

    def test_unknown_or_missing_assignments_are_rejected(self):
        for assignments in ({**self.assignments, "unknown": "train"},
                            {k: v for k, v in self.assignments.items() if k != "img-3"}):
            with self.subTest(assignments=assignments), self.assertRaisesRegex(PartitionError, "exactly"):
                self.build(assignments=assignments)

    def test_unsupported_split_is_rejected(self):
        with self.assertRaisesRegex(PartitionError, "unsupported split"):
            self.build(assignments={**self.assignments, "img-3": "validation"})

    def test_related_sources_cannot_cross_splits(self):
        with self.assertRaisesRegex(PartitionError, "crosses splits"):
            self.build(assignments={**self.assignments, "img-2": "valid"})

    def test_transitive_duplicate_link_cannot_cross_splits(self):
        images = list(self.dataset.accepted)
        images[1] = replace(images[1], near_duplicate_group_id="linked")
        images[2] = replace(images[2], near_duplicate_group_id="linked")
        with self.assertRaisesRegex(PartitionError, "crosses splits"):
            self.build(replace(self.dataset, accepted=tuple(images)))

    def test_missing_required_provenance_blocks_partitioning(self):
        images = self.dataset.accepted
        changed = replace(self.dataset, accepted=(replace(images[0], source_lot=None), *images[1:]))
        with self.assertRaisesRegex(PartitionError, "img-1: source_lot"):
            self.build(changed)

    def test_frozen_image_id_or_group_is_rejected_without_loading_pixels(self):
        manifest = self.build()
        with patch.object(Path, "open", side_effect=AssertionError("must not read pixels")):
            for ids, groups in ((["img-1"], []), ([], [manifest.images[0].group_id])):
                with self.subTest(ids=ids, groups=groups), self.assertRaisesRegex(PartitionError, "Frozen test"):
                    select_development_images(manifest, "train", frozen_image_ids=ids,
                                              frozen_group_ids=groups)

    def test_development_selection_rejects_test_and_preserves_valid_negatives(self):
        manifest = self.build()
        selected = select_development_images(manifest, "valid", frozen_image_ids=["img-4"],
                                             frozen_group_ids=[])
        self.assertEqual([item.image.image_id for item in selected], ["img-3"])
        self.assertEqual(selected[0].objects, ())
        with self.assertRaisesRegex(PartitionError, "never test"):
            select_development_images(manifest, "test", frozen_image_ids=[], frozen_group_ids=[])

    def test_hash_changes_when_assignment_or_labels_change(self):
        original = self.build()
        reassigned = {**self.assignments, "img-3": "test", "img-4": "valid"}
        self.assertNotEqual(original.content_hash, self.build(assignments=reassigned).content_hash)
        annotation = self.dataset.annotations[0]
        changed = replace(self.dataset, annotations=(replace(annotation, objects=()),))
        self.assertNotEqual(original.content_hash, self.build(changed).content_hash)

    def test_conflicting_class_names_are_rejected(self):
        extra = ParsedAnnotations(Path("other.json"), (ClassSpec(7, "different"),), (), ())
        with self.assertRaisesRegex(PartitionError, "Conflicting names"):
            self.build(replace(self.dataset, annotations=(*self.dataset.annotations, extra)))
