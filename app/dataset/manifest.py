"""Build a versioned manifest from accepted dataset records.

The manifest lists accepted images only. Rejected records stay out.
Negative images stay in. Paths are relative to ``DatasetInput.root`` and
use the existing path identity helper. This module does not write files,
summarize the dataset, or decide which images were accepted.

Provisional COCO association, not confirmed by an importer: an object
matches an accepted image when the annotation file's relative path and
``coco_image_id`` both match. A ``coco_image_id`` is unique only inside
one annotation file, and several images may share one file.

``content_hash`` is the SHA-256 hex digest of the UTF-8 canonical JSON
for the source policy, each accepted image's manifest fields, and the
objects joined by that COCO rule. Joined objects contribute their
annotation id, category id, category name, and bbox.

The hash does not cover the absolute dataset root, the ``dataset_version``
label, ``schema_version``, rejected records, or the bytes of image and
annotation files. It does not verify those files.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from app.dataset.records import (
    AcceptedImage,
    DatasetInput,
    SourcePolicy,
    eligible_for_split,
    missing_provenance_fields,
    objects_for_image,
    relative_path_identity,
)


SCHEMA_VERSION = 1
COCO_ASSOCIATION_RULE = "annotation_file_and_coco_image_id"
COCO_ASSOCIATION_NOTE = (
    "Provisional assumption, not confirmed by an importer: an object matches "
    "an accepted image when the annotation file's relative path and "
    "coco_image_id both match. A coco_image_id is unique only inside one "
    "annotation file, and several images may share one file. An accepted "
    "negative image can have a COCO images entry and no matching objects."
)


@dataclass(frozen=True)
class ManifestImage:
    image_id: str
    coco_image_id: int
    image_path: str
    annotation_path: str
    source_lot: str | None
    source_session: str | None
    near_duplicate_group_id: str | None
    split_eligible: bool
    split_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "annotation_path": self.annotation_path,
            "coco_image_id": self.coco_image_id,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "near_duplicate_group_id": self.near_duplicate_group_id,
            "source_lot": self.source_lot,
            "source_session": self.source_session,
            "split_blockers": list(self.split_blockers),
            "split_eligible": self.split_eligible,
        }


@dataclass(frozen=True)
class DatasetManifest:
    schema_version: int
    dataset_version: str
    content_hash: str
    source_policy: SourcePolicy
    images: tuple[ManifestImage, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coco_association": {
                "note": COCO_ASSOCIATION_NOTE,
                "rule": COCO_ASSOCIATION_RULE,
            },
            "content_hash": self.content_hash,
            "dataset_version": self.dataset_version,
            "images": [image.to_dict() for image in self.images],
            "schema_version": self.schema_version,
            "source_policy": {
                "require_lot": self.source_policy.require_lot,
                "require_session": self.source_policy.require_session,
            },
        }


def build_manifest(dataset: DatasetInput, dataset_version: str) -> DatasetManifest:
    """Return a manifest for ``dataset`` labeled with ``dataset_version``.

    Unknown lot, session, and near-duplicate group values are ``None``
    and become JSON null. Split blockers are the missing required source
    fields under ``dataset.source_policy``.
    """
    label = dataset_version.strip()
    if not label:
        raise ValueError("dataset_version must be an explicit non-empty label.")

    images = tuple(
        _manifest_image(dataset, image)
        for image in sorted(dataset.accepted, key=lambda item: item.image_id)
    )
    return DatasetManifest(
        schema_version=SCHEMA_VERSION,
        dataset_version=label,
        content_hash=_content_hash(dataset, images),
        source_policy=dataset.source_policy,
        images=images,
    )


def manifest_json(manifest: DatasetManifest) -> str:
    """Serialize ``manifest`` with sorted keys and compact separators."""
    return _canonical_json(manifest.to_dict())


def _manifest_image(dataset: DatasetInput, image: AcceptedImage) -> ManifestImage:
    policy = dataset.source_policy
    return ManifestImage(
        image_id=image.image_id,
        coco_image_id=image.coco_image_id,
        image_path=relative_path_identity(image.image_path),
        annotation_path=relative_path_identity(image.annotation_path),
        source_lot=image.source_lot,
        source_session=image.source_session,
        near_duplicate_group_id=image.near_duplicate_group_id,
        split_eligible=eligible_for_split(image, policy),
        split_blockers=missing_provenance_fields(image, policy),
    )


def _content_hash(dataset: DatasetInput, images: tuple[ManifestImage, ...]) -> str:
    by_id = {image.image_id: image for image in dataset.accepted}
    payload = {
        "images": [
            {
                **manifest_image.to_dict(),
                "joined_objects": _joined_objects(dataset, by_id[manifest_image.image_id]),
            }
            for manifest_image in images
        ],
        "source_policy": {
            "require_lot": dataset.source_policy.require_lot,
            "require_session": dataset.source_policy.require_session,
        },
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return digest


def _joined_objects(dataset: DatasetInput, image: AcceptedImage) -> list[dict[str, Any]]:
    """Objects for ``image`` using annotation-file identity plus coco_image_id."""
    names = _class_names(dataset, image)
    objects = [
        {
            "annotation_id": obj.annotation_id,
            "bbox_xywh": list(obj.bbox_xywh),
            "category_id": obj.category_id,
            "category_name": names.get(obj.category_id),
        }
        for obj in objects_for_image(dataset, image)
    ]
    objects.sort(key=lambda item: item["annotation_id"])
    return objects


def _class_names(dataset: DatasetInput, image: AcceptedImage) -> dict[int, str]:
    identity = relative_path_identity(image.annotation_path)
    names: dict[int, str] = {}
    for parsed in dataset.annotations:
        if relative_path_identity(parsed.path) != identity:
            continue
        for spec in parsed.classes:
            names[spec.category_id] = spec.name
    return names


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
