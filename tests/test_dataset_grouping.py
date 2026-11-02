import tempfile
import unittest
from pathlib import Path

from app.dataset.grouping import (
    GROUP_ID_PREFIX,
    GroupingError,
    assign_groups,
    make_group_id,
)
from app.dataset.records import AcceptedImage, SourcePolicy
from tests.fixtures.load_dataset_fixture import load_dataset_fixture

NO_REQUIREMENTS = SourcePolicy(require_lot=False, require_session=False)
REQUIRE_BOTH = SourcePolicy(require_lot=True, require_session=True)


def image(image_id, lot=None, session=None, dup=None):
    return AcceptedImage(
        image_id=image_id,
        coco_image_id=1,
        image_path=f"images/{image_id}.png",
        annotation_path="annotations/instances.json",
        width=80,
        height=60,
        source_lot=lot,
        source_session=session,
        near_duplicate_group_id=dup,
    )


class AssignGroupsTests(unittest.TestCase):
    def test_images_sharing_a_lot_are_one_group(self):
        result = assign_groups(
            [image("a", lot="L1"), image("b", lot="L1"), image("c", lot="L2")], NO_REQUIREMENTS
        )
        mapping = result.group_of()
        self.assertEqual(mapping["a"], mapping["b"])
        self.assertNotEqual(mapping["a"], mapping["c"])
        self.assertEqual(len(result.groups), 2)

    def test_links_are_transitive_across_lot_and_session(self):
        # a-b share a lot, b-c share a session: all three must stay together.
        result = assign_groups(
            [
                image("a", lot="L1", session="S1"),
                image("b", lot="L1", session="S2"),
                image("c", lot="L9", session="S2"),
            ],
            REQUIRE_BOTH,
        )
        self.assertEqual(len(result.groups), 1)
        self.assertEqual(result.groups[0].image_ids, ("a", "b", "c"))

    def test_near_duplicate_id_merges_otherwise_separate_lots(self):
        result = assign_groups(
            [image("a", lot="L1", dup="d1"), image("b", lot="L2", dup="d1")], NO_REQUIREMENTS
        )
        self.assertEqual(len(result.groups), 1)

    def test_no_lot_or_session_value_is_shared_between_groups(self):
        images = [
            image("a", lot="L1", session="S1"),
            image("b", lot="L2", session="S1"),
            image("c", lot="L3", session="S3"),
            image("d", lot="L3", session="S4"),
            image("e", lot="L5", session="S5"),
        ]
        result = assign_groups(images, REQUIRE_BOTH)
        by_id = {img.image_id: img for img in images}
        owner = {}
        for group in result.groups:
            for image_id in group.image_ids:
                for value in (by_id[image_id].source_lot, by_id[image_id].source_session):
                    self.assertEqual(owner.setdefault(value, group.group_id), group.group_id)

    def test_lot_and_session_with_same_name_are_not_merged(self):
        result = assign_groups(
            [image("a", lot="A"), image("b", session="A")], NO_REQUIREMENTS
        )
        self.assertEqual(len(result.groups), 2)

    def test_group_id_does_not_depend_on_input_order(self):
        images = [image("a", lot="L1"), image("b", lot="L1", session="S1"), image("c", lot="L2")]
        forward = assign_groups(images, NO_REQUIREMENTS).group_of()
        backward = assign_groups(list(reversed(images)), NO_REQUIREMENTS).group_of()
        self.assertEqual(forward, backward)
        self.assertTrue(all(g.startswith(GROUP_ID_PREFIX) for g in forward.values()))

    def test_adding_a_frame_to_an_existing_lot_keeps_the_group_id(self):
        before = assign_groups([image("a", lot="L1")], NO_REQUIREMENTS).group_of()["a"]
        after = assign_groups([image("a", lot="L1"), image("b", lot="L1")], NO_REQUIREMENTS).group_of()
        self.assertEqual(after["a"], before)

    def test_missing_required_provenance_is_held_out_not_guessed(self):
        result = assign_groups(
            [image("a", lot="L1", session="S1"), image("b", lot="L1", session=None)], REQUIRE_BOTH
        )
        self.assertEqual([u.image_id for u in result.ungrouped], ["b"])
        self.assertEqual(result.ungrouped[0].missing, ("source_session",))
        self.assertNotIn("b", result.group_of())

    def test_image_with_no_keys_at_all_is_ungrouped(self):
        result = assign_groups([image("a")], NO_REQUIREMENTS)
        self.assertEqual(result.groups, ())
        self.assertEqual(result.ungrouped[0].image_id, "a")

    def test_repeated_image_id_is_an_error(self):
        with self.assertRaises(GroupingError):
            assign_groups([image("a", lot="L1"), image("a", lot="L2")], NO_REQUIREMENTS)

    def test_make_group_id_needs_a_key(self):
        with self.assertRaises(GroupingError):
            make_group_id([])

    def test_works_on_the_development_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = load_dataset_fixture(Path(directory))
        result = assign_groups(dataset.accepted, dataset.source_policy)
        # fix-001 and fix-002 share lot-1; fix-003 has no lot, which the policy requires.
        self.assertEqual([g.image_ids for g in result.groups], [("fix-001", "fix-002")])
        self.assertEqual([u.image_id for u in result.ungrouped], ["fix-003"])


if __name__ == "__main__":
    unittest.main()