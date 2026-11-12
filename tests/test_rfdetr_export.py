import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.dataset.frozen_test_set import freeze_test_set
from app.dataset.partitions import build_partition_manifest
from app.dataset.manifest import build_manifest
from app.dataset.rfdetr_export import (
    COCO_FILE_NAME,
    PROVENANCE_FILE_NAME,
    RfdetrExportError,
    SplitAssignment,
    export_rfdetr_dataset,
    load_file_inputs,
)
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


CLASS_MAP = {1: "A", 2: "B"}


def assignments():
    return (
        SplitAssignment("fix-001", "train", "group-lot-1"),
        SplitAssignment("fix-002", "train", "group-lot-1"),
        SplitAssignment("fix-003", "valid", "group-session-2"),
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_preserves_ids_paths_classes_boxes_and_dimensions(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "sprint-2-v1")
    output = tmp_path / "rfdetr"

    result = export_rfdetr_dataset(dataset, manifest, assignments(), CLASS_MAP, output)

    assert result.split_counts == {"train": 2, "valid": 1}
    assert result.annotation_counts == {"train": 4, "valid": 0}
    train = read_json(output / "train" / COCO_FILE_NAME)
    assert train["info"]["dataset_version"] == "sprint-2-v1"
    assert train["info"]["dataset_content_hash"] == manifest.content_hash
    assert train["categories"] == [
        {"id": 1, "name": "A", "supercategory": "object"},
        {"id": 2, "name": "B", "supercategory": "object"},
    ]
    assert [(item["id"], item["file_name"], item["width"], item["height"]) for item in train["images"]] == [
        (1, "images/two_a.png", 80, 60),
        (2, "images/a_and_b.png", 80, 60),
    ]
    assert [(item["id"], item["image_id"], item["category_id"], item["bbox"]) for item in train["annotations"]] == [
        (11, 1, 1, [10.0, 10.0, 20.0, 15.0]),
        (12, 1, 1, [40.0, 30.0, 20.0, 15.0]),
        (21, 2, 1, [5.0, 5.0, 18.0, 12.0]),
        (22, 2, 2, [40.0, 28.0, 22.0, 16.0]),
    ]
    assert train["annotations"][0]["area"] == 300.0
    assert (output / "train" / "images" / "two_a.png").read_bytes() == dataset.resolve(
        "images/two_a.png"
    ).read_bytes()


def test_negative_image_and_source_provenance_are_preserved(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "sprint-2-v1")
    output = tmp_path / "rfdetr"

    export_rfdetr_dataset(dataset, manifest, assignments(), CLASS_MAP, output)

    valid = read_json(output / "valid" / COCO_FILE_NAME)
    assert valid["annotations"] == []
    assert valid["images"] == [
        {
            "file_name": "images/negative.png",
            "height": 60,
            "id": 3,
            "near_duplicate_group_id": None,
            "source_group_id": "group-session-2",
            "source_lot": None,
            "source_session": "session-2",
            "stable_image_id": "fix-003",
            "width": 80,
        }
    ]
    provenance = read_json(output / PROVENANCE_FILE_NAME)
    assert provenance["dataset_version"] == "sprint-2-v1"
    negative = next(row for row in provenance["images"] if row["image_id"] == "fix-003")
    assert negative["annotation_count"] == 0
    assert negative["source_group_id"] == "group-session-2"
    assert negative["split"] == "valid"


def test_test_assignment_is_not_exported(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "sprint-2-v1")
    split_rows = (
        SplitAssignment("fix-001", "train", "group-lot-1"),
        SplitAssignment("fix-002", "train", "group-lot-1"),
        SplitAssignment("fix-003", "test", "frozen-group"),
    )
    output = tmp_path / "rfdetr"

    export_rfdetr_dataset(dataset, manifest, split_rows, CLASS_MAP, output)

    all_images = read_json(output / "train" / COCO_FILE_NAME)["images"]
    all_images += read_json(output / "valid" / COCO_FILE_NAME)["images"]
    assert {item["id"] for item in all_images} == {1, 2}
    assert not (output / "valid" / "images" / "negative.png").exists()


def test_frozen_file_or_group_leak_is_an_identifiable_error(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "sprint-2-v1")
    frozen_root = tmp_path / "frozen"
    (frozen_root / "test").mkdir(parents=True)
    (frozen_root / "test" / "copied.png").write_bytes(dataset.resolve("images/two_a.png").read_bytes())
    (frozen_root / "split_manifest.json").write_text(json.dumps({"images": [
        {"output": "test/copied.png", "source": "original.png", "group_id": "frozen-group", "split": "test"}
    ]}))
    lock = freeze_test_set(frozen_root, "test-v1")

    with pytest.raises(RfdetrExportError, match=r"\[FROZEN_TEST_LEAK\].*match the frozen test set"):
        export_rfdetr_dataset(
            dataset,
            manifest,
            assignments(),
            CLASS_MAP,
            tmp_path / "rfdetr",
            frozen_test_lock=lock,
        )
    assert not (tmp_path / "rfdetr").exists()


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("missing_image", "MISSING_IMAGE"),
        ("unknown_class", "UNKNOWN_CLASS"),
        ("invalid_box", "INVALID_BOX"),
    ],
)
def test_bad_inputs_fail_before_any_output_is_published(tmp_path, failure, expected_code):
    dataset = load_dataset_fixture(tmp_path / "source")
    class_map = CLASS_MAP
    if failure == "missing_image":
        dataset.resolve("images/two_a.png").unlink()
    elif failure == "unknown_class":
        class_map = {1: "A"}
    else:
        parsed = dataset.annotations[0]
        objects = tuple(
            replace(obj, bbox_xywh=(70, 50, 20, 20)) if obj.annotation_id == 11 else obj
            for obj in parsed.objects
        )
        dataset = replace(dataset, annotations=(replace(parsed, objects=objects),))
    manifest = build_manifest(dataset, "sprint-2-v1")

    with pytest.raises(RfdetrExportError) as raised:
        export_rfdetr_dataset(dataset, manifest, assignments(), class_map, tmp_path / "rfdetr")

    assert raised.value.code == expected_code
    assert f"[{expected_code}]" in str(raised.value)
    assert not (tmp_path / "rfdetr").exists()


def test_file_loader_uses_versioned_manifest_and_split_manifest(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "sprint-2-v1")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    split_path = tmp_path / "split_manifest.json"
    split_path.write_text(
        json.dumps(
            {
                "images": [
                    {"image_id": row.image_id, "split": row.split, "group_id": row.source_group_id}
                    for row in assignments()
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded_dataset, loaded_manifest, loaded_assignments = load_file_inputs(
        dataset.root, manifest_path, split_path
    )

    assert loaded_manifest.to_dict() == manifest.to_dict()
    assert loaded_assignments == assignments()
    assert [image.image_id for image in loaded_dataset.accepted] == [
        "fix-001",
        "fix-002",
        "fix-003",
    ]


def test_source_group_may_not_cross_splits(tmp_path):
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "sprint-2-v1")
    leaking = (
        SplitAssignment("fix-001", "train", "same-capture"),
        SplitAssignment("fix-002", "valid", "same-capture"),
        SplitAssignment("fix-003", "valid", "other-capture"),
    )

    with pytest.raises(RfdetrExportError, match=r"\[SOURCE_GROUP_LEAK\]"):
        export_rfdetr_dataset(dataset, manifest, leaking, CLASS_MAP, tmp_path / "rfdetr")


def test_checked_partition_handoff_exports_with_canonical_groups(tmp_path):
    from dataclasses import replace
    dataset = load_dataset_fixture(tmp_path / "source")
    # Complete provenance for the checked handoff's negative validation image.
    dataset = replace(dataset, accepted=tuple(
        replace(image, source_lot="lot-2") if image.image_id == "fix-003" else image
        for image in dataset.accepted
    ))
    manifest = build_manifest(dataset, "v2")
    partition = build_partition_manifest(dataset, "v2", {
        "fix-001": "train", "fix-002": "train", "fix-003": "valid"
    })
    manifest_path = tmp_path / "manifest.json"
    partition_path = tmp_path / "partitions.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    partition_path.write_text(json.dumps(partition.to_dict()))
    loaded, versioned, selected = load_file_inputs(dataset.root, manifest_path, partition_path)
    result = export_rfdetr_dataset(loaded, versioned, selected, CLASS_MAP, tmp_path / "export")
    valid = read_json(result.output_directory / "valid" / COCO_FILE_NAME)
    assert valid["images"][0]["source_group_id"] == partition.images[2].group_id
    assert valid["annotations"] == []
    assert selected[0].source_group_id == selected[1].source_group_id

    tampered = partition.to_dict()
    tampered["images"][0]["group_id"] = "invented-group"
    partition_path.write_text(json.dumps(tampered))
    with pytest.raises(RfdetrExportError, match="PARTITION_MISMATCH"):
        load_file_inputs(dataset.root, manifest_path, partition_path)


def test_malformed_frozen_lock_fails_before_loading_pixels(tmp_path):
    from unittest.mock import patch
    dataset = load_dataset_fixture(tmp_path / "source")
    manifest = build_manifest(dataset, "v1")
    with patch("app.dataset.rfdetr_export.Image.open", side_effect=AssertionError("no pixel reads")):
        with pytest.raises(RfdetrExportError, match="INVALID_TEST_LOCK"):
            export_rfdetr_dataset(dataset, manifest, assignments(), CLASS_MAP, tmp_path / "export",
                                  frozen_test_lock={"files": [], "groups": []})
