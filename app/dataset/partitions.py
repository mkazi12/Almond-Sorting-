"""Checked split assignments for accepted, labeled dataset records.

This metadata handoff preserves source IDs, labels and grouping; it does not
choose partitions, read images, export RF-DETR COCO files or launch training.
Both hashes cover metadata and labels, not source-file bytes. Consumers must
also verify image/checkpoint bytes and call the frozen-content guard.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.dataset.grouping import assign_groups
from app.dataset.manifest import build_manifest
from app.dataset.records import (
    AcceptedImage,
    ClassSpec,
    DatasetInput,
    GroundTruthObject,
    objects_for_image,
    relative_path_identity,
)

SPLITS = ("train", "valid", "test")


class PartitionError(ValueError):
    """A split assignment or requested development selection is unsafe."""


@dataclass(frozen=True)
class PartitionImage:
    image: AcceptedImage
    group_id: str
    split: str
    objects: tuple[GroundTruthObject, ...]

    def to_dict(self) -> dict:
        image = self.image
        return {
            "image_id": image.image_id,
            "coco_image_id": image.coco_image_id,
            "image_path": relative_path_identity(image.image_path),
            "annotation_path": relative_path_identity(image.annotation_path),
            "width": image.width,
            "height": image.height,
            "source_lot": image.source_lot,
            "source_session": image.source_session,
            "near_duplicate_group_id": image.near_duplicate_group_id,
            "group_id": self.group_id,
            "split": self.split,
            "objects": [
                {
                    "annotation_id": obj.annotation_id,
                    "category_id": obj.category_id,
                    "bbox_xywh": list(obj.bbox_xywh),
                }
                for obj in self.objects
            ],
        }


@dataclass(frozen=True)
class PartitionManifest:
    dataset_version: str
    source_content_hash: str
    content_hash: str
    classes: tuple[ClassSpec, ...]
    images: tuple[PartitionImage, ...]

    def to_dict(self) -> dict:
        return {
            **_payload(self.dataset_version, self.source_content_hash,
                       self.classes, self.images),
            "content_hash": self.content_hash,
        }


def _json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload(version, source_hash, classes, images) -> dict:
    return {
        "schema_version": 1,
        "kind": "labeled_dataset_partitions",
        "dataset_version": version,
        "source_content_hash": source_hash,
        "classes": [{"category_id": item.category_id, "name": item.name} for item in classes],
        "images": [item.to_dict() for item in images],
    }


def build_partition_manifest(
    dataset: DatasetInput,
    dataset_version: str,
    assignments: Mapping[str, str],
) -> PartitionManifest:
    """Validate a supplied image-ID-to-split map without changing its choices.

    Every accepted ID must occur exactly once. Related lot/session/duplicate
    groups stay whole. Missing required provenance blocks the entire handoff.
    Rejected records are never assigned; reviewed negatives retain zero objects.
    """
    if not isinstance(assignments, Mapping):
        raise PartitionError("assignments must map accepted image IDs to train, valid or test.")
    if any(not isinstance(key, str) or not key.strip() for key in assignments):
        raise PartitionError("Assignment image IDs must be nonempty strings.")
    expected = {image.image_id for image in dataset.accepted}
    supplied = set(assignments)
    if expected != supplied:
        raise PartitionError(
            f"Assignments must cover accepted IDs exactly; missing={sorted(expected - supplied)}, "
            f"unknown={sorted(supplied - expected)}."
        )
    for image_id, split in assignments.items():
        if split not in SPLITS:
            raise PartitionError(f"Image {image_id}: unsupported split {split!r}; use {SPLITS}.")
    grouping = assign_groups(dataset.accepted, dataset.source_policy)
    if grouping.ungrouped:
        details = ", ".join(f"{item.image_id}: {','.join(item.missing)}" for item in grouping.ungrouped)
        raise PartitionError(f"Cannot partition images without required provenance: {details}.")
    for group in grouping.groups:
        selected = {assignments[image_id] for image_id in group.image_ids}
        if len(selected) > 1:
            raise PartitionError(
                f"Source group {group.group_id} crosses splits {sorted(selected)}: {group.image_ids}."
            )
    groups = grouping.group_of()
    classes_by_id: dict[int, str] = {}
    for parsed in dataset.annotations:
        for item in parsed.classes:
            if item.category_id in classes_by_id and classes_by_id[item.category_id] != item.name:
                raise PartitionError(f"Conflicting names for category ID {item.category_id}.")
            classes_by_id[item.category_id] = item.name
    classes = tuple(ClassSpec(key, value) for key, value in sorted(classes_by_id.items()))
    source = build_manifest(dataset, dataset_version)
    images = tuple(
        PartitionImage(
            image=image,
            group_id=groups[image.image_id],
            split=assignments[image.image_id],
            objects=tuple(sorted(objects_for_image(dataset, image), key=lambda obj: obj.annotation_id)),
        )
        for image in sorted(dataset.accepted, key=lambda item: item.image_id)
    )
    digest = hashlib.sha256(_json(_payload(source.dataset_version, source.content_hash,
                                          classes, images)).encode("utf-8")).hexdigest()
    return PartitionManifest(source.dataset_version, source.content_hash, digest, classes, images)


def partition_manifest_json(manifest: PartitionManifest) -> str:
    """Return deterministic JSON; the absolute dataset root is never included."""
    return _json(manifest.to_dict())


def _membership(values: Iterable[str], label: str) -> set[str]:
    if isinstance(values, (str, bytes)):
        raise PartitionError(f"{label} must be an iterable of IDs, not a single string.")
    result = set(values)
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise PartitionError(f"{label} must contain nonempty string IDs.")
    return result


def select_development_images(
    manifest: PartitionManifest,
    split: str,
    *,
    frozen_image_ids: Iterable[str],
    frozen_group_ids: Iterable[str],
) -> tuple[PartitionImage, ...]:
    """Select train/valid metadata after explicit frozen ID/group exclusions.

    Supply membership from the independently approved frozen test inventory.
    This function never reads pixels; call it before any image-loading loop.
    It also excludes this manifest's test partition. Content-hash exclusion is
    a separate required guard, because copied images can have different IDs.
    """
    if split not in ("train", "valid"):
        raise PartitionError("Development consumers may select only train or valid, never test.")
    frozen_ids = _membership(frozen_image_ids, "frozen_image_ids")
    frozen_groups = _membership(frozen_group_ids, "frozen_group_ids")
    for item in manifest.images:
        if item.split == "test":
            frozen_ids.add(item.image.image_id)
            frozen_groups.add(item.group_id)
    selected = tuple(item for item in manifest.images if item.split == split)
    for item in selected:
        if item.image.image_id in frozen_ids or item.group_id in frozen_groups:
            raise PartitionError(
                f"Frozen test membership overlaps {split}: image {item.image.image_id}, "
                f"group {item.group_id}."
            )
    return selected
