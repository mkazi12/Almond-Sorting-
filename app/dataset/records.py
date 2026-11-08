"""Dataset records shared by import, manifest, summary, and preview code.

Paths stored on these records are relative to ``DatasetInput.root``.
Absolute locations come only from ``resolve_dataset_path``. Copying an
identical dataset to another root keeps stable ids and
``canonical_content`` unchanged. A future manifest can hash that content
and store a version label separately. This module does not build a
manifest, summarize a dataset, or decide which files are valid.

COCO assumption: annotations live in COCO instances files. Several images
may share one file. An object belongs to an accepted image when the
annotation-file path and ``coco_image_id`` both match. A negative image
still has a COCO ``images`` entry and zero objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.dataset.paths import (
    DatasetPathError,
    relative_dataset_path,
    relative_path_identity,
    resolve_dataset_path,
)

UNKNOWN_SOURCE = "unknown"


def _normalize_source_value(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def source_label(value: str | None) -> str:
    """Lot or session group used by later image counts. Blank means unknown."""
    normalized = _normalize_source_value(value)
    if normalized is None:
        return UNKNOWN_SOURCE
    return normalized


def _require_text(value: str, message: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(message)
    return stripped


def _require_int(value: int, message: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(message)
    return value


@dataclass(frozen=True)
class ClassSpec:
    category_id: int
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "category_id", _require_int(self.category_id, "category_id must be an integer.")
        )
        object.__setattr__(self, "name", _require_text(self.name, "A class needs a name."))


@dataclass(frozen=True)
class GroundTruthObject:
    annotation_id: int
    coco_image_id: int
    category_id: int
    bbox_xywh: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "annotation_id",
            _require_int(self.annotation_id, "annotation_id must be an integer."),
        )
        object.__setattr__(
            self,
            "coco_image_id",
            _require_int(self.coco_image_id, "coco_image_id must be an integer."),
        )
        object.__setattr__(
            self,
            "category_id",
            _require_int(self.category_id, "category_id must be an integer."),
        )
        object.__setattr__(self, "bbox_xywh", _bbox_xywh(self.bbox_xywh))


@dataclass(frozen=True)
class CocoImage:
    """One COCO ``images`` entry, including negatives with no annotations."""

    coco_image_id: int
    file_name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coco_image_id",
            _require_int(self.coco_image_id, "coco_image_id must be an integer."),
        )
        object.__setattr__(self, "file_name", relative_path_identity(self.file_name))
        width = _require_int(self.width, "width must be an integer.")
        height = _require_int(self.height, "height must be an integer.")
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)


@dataclass(frozen=True)
class ParsedAnnotations:
    """Annotations from one COCO file.

    Join key: this file's relative path plus ``GroundTruthObject.coco_image_id``.
    """

    path: Path
    classes: tuple[ClassSpec, ...]
    images: tuple[CocoImage, ...]
    objects: tuple[GroundTruthObject, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", relative_dataset_path(self.path))
        object.__setattr__(self, "classes", tuple(self.classes))
        object.__setattr__(self, "images", tuple(self.images))
        object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True)
class SourcePolicy:
    """Which source fields count as required provenance.

    Missing required provenance does not remove an accepted image. It makes
    that image ineligible for splitting.
    """

    require_lot: bool
    require_session: bool


@dataclass(frozen=True)
class AcceptedImage:
    image_id: str
    coco_image_id: int
    image_path: Path
    annotation_path: Path
    width: int
    height: int
    source_lot: str | None
    source_session: str | None
    near_duplicate_group_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "image_id", _require_text(self.image_id, "Accepted images need a stable image_id.")
        )
        object.__setattr__(
            self,
            "coco_image_id",
            _require_int(self.coco_image_id, "coco_image_id must be an integer."),
        )
        object.__setattr__(self, "image_path", relative_dataset_path(self.image_path))
        object.__setattr__(self, "annotation_path", relative_dataset_path(self.annotation_path))
        width = _require_int(self.width, "width must be an integer.")
        height = _require_int(self.height, "height must be an integer.")
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "source_lot", _normalize_source_value(self.source_lot))
        object.__setattr__(self, "source_session", _normalize_source_value(self.source_session))
        object.__setattr__(
            self,
            "near_duplicate_group_id",
            _normalize_source_value(self.near_duplicate_group_id),
        )


@dataclass(frozen=True)
class RejectionReason:
    category: str
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "category", _require_text(self.category, "A rejection reason needs a category.")
        )
        object.__setattr__(
            self, "detail", _require_text(self.detail, "A rejection reason needs a detail.")
        )


@dataclass(frozen=True)
class RejectedImage:
    """One rejected image. ``reasons`` may contain several categories.

    Later rejected totals count these records, not the reasons inside them.
    ``image_path`` is kept when no stable ``image_id`` is available.
    """

    image_path: Path
    image_id: str | None
    reasons: tuple[RejectionReason, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_path", relative_dataset_path(self.image_path))
        if self.image_id is not None:
            object.__setattr__(
                self,
                "image_id",
                _require_text(self.image_id, "Rejected image_id, when present, must not be blank."),
            )
        reasons = tuple(self.reasons)
        if not reasons:
            raise ValueError("A rejected image needs at least one categorized reason.")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True)
class DatasetInput:
    """Accepted and rejected records for one dataset root.

    Later summaries can count accepted and rejected images, group accepted
    images by lot and session (blank values use ``UNKNOWN_SOURCE``), count
    annotations and distinct images per class through the COCO join, count
    accepted images with no joined objects, and list each rejected path
    with every reason. Those summaries are not computed here.
    """

    root: Path
    source_policy: SourcePolicy
    accepted: tuple[AcceptedImage, ...]
    rejected: tuple[RejectedImage, ...]
    annotations: tuple[ParsedAnnotations, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        accepted = tuple(self.accepted)
        image_ids = [image.image_id for image in accepted]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("Accepted image_id values must be unique.")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "rejected", tuple(self.rejected))
        object.__setattr__(self, "annotations", tuple(self.annotations))

    def resolve(self, relative_path: str | Path) -> Path:
        return resolve_dataset_path(self.root, relative_path)


def missing_provenance_fields(image: AcceptedImage, policy: SourcePolicy) -> tuple[str, ...]:
    """Required source fields that are blank under ``policy``."""
    missing: list[str] = []
    if policy.require_lot and image.source_lot is None:
        missing.append("source_lot")
    if policy.require_session and image.source_session is None:
        missing.append("source_session")
    return tuple(missing)


def eligible_for_split(image: AcceptedImage, policy: SourcePolicy) -> bool:
    """Return whether an accepted image may be used in a split.

    Missing required provenance leaves the image accepted and blocks splitting.
    """
    return not missing_provenance_fields(image, policy)


def _annotation_files(dataset: DatasetInput, image: AcceptedImage) -> tuple[ParsedAnnotations, ...]:
    identity = relative_path_identity(image.annotation_path)
    return tuple(
        parsed
        for parsed in dataset.annotations
        if relative_path_identity(parsed.path) == identity
    )


def objects_for_image(
    dataset: DatasetInput, image: AcceptedImage
) -> tuple[GroundTruthObject, ...]:
    """Objects joined on annotation-file identity and ``coco_image_id``."""
    return tuple(
        obj
        for parsed in _annotation_files(dataset, image)
        for obj in parsed.objects
        if obj.coco_image_id == image.coco_image_id
    )


def coco_image_for(dataset: DatasetInput, image: AcceptedImage) -> CocoImage | None:
    """COCO ``images`` entry joined on annotation-file identity and id."""
    for parsed in _annotation_files(dataset, image):
        for coco_image in parsed.images:
            if coco_image.coco_image_id == image.coco_image_id:
                return coco_image
    return None


def canonical_content(dataset: DatasetInput) -> dict[str, Any]:
    """Root-independent content a future manifest can hash.

    The dataset root is omitted. A future manifest stores its version label
    separately from a hash of this content.
    """
    return {
        "source_policy": {
            "require_lot": dataset.source_policy.require_lot,
            "require_session": dataset.source_policy.require_session,
        },
        "accepted": [
            {
                "image_id": image.image_id,
                "coco_image_id": image.coco_image_id,
                "image_path": relative_path_identity(image.image_path),
                "annotation_path": relative_path_identity(image.annotation_path),
                "width": image.width,
                "height": image.height,
                "source_lot": image.source_lot,
                "source_session": image.source_session,
                "near_duplicate_group_id": image.near_duplicate_group_id,
            }
            for image in sorted(dataset.accepted, key=lambda item: item.image_id)
        ],
        "rejected": [
            {
                "image_id": rejected.image_id,
                "image_path": relative_path_identity(rejected.image_path),
                "reasons": [
                    {"category": reason.category, "detail": reason.detail}
                    for reason in rejected.reasons
                ],
            }
            for rejected in sorted(dataset.rejected, key=_rejected_sort_key)
        ],
        "annotations": [
            {
                "path": relative_path_identity(parsed.path),
                "classes": [
                    {"category_id": item.category_id, "name": item.name}
                    for item in sorted(parsed.classes, key=lambda item: (item.category_id, item.name))
                ],
                "images": [
                    {
                        "coco_image_id": item.coco_image_id,
                        "file_name": item.file_name,
                        "width": item.width,
                        "height": item.height,
                    }
                    for item in sorted(parsed.images, key=lambda item: item.coco_image_id)
                ],
                "objects": [
                    {
                        "annotation_id": item.annotation_id,
                        "coco_image_id": item.coco_image_id,
                        "category_id": item.category_id,
                        "bbox_xywh": list(item.bbox_xywh),
                    }
                    for item in sorted(parsed.objects, key=lambda item: item.annotation_id)
                ],
            }
            for parsed in sorted(
                dataset.annotations, key=lambda parsed: relative_path_identity(parsed.path)
            )
        ],
    }


def _rejected_sort_key(rejected: RejectedImage) -> tuple[str, str, tuple[str, ...]]:
    return (
        relative_path_identity(rejected.image_path),
        rejected.image_id or "",
        tuple(reason.category for reason in rejected.reasons),
    )


def _bbox_xywh(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError("bbox_xywh must contain x, y, width, and height.")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("bbox_xywh values must be numbers.")
        numbers.append(float(item))
    return (numbers[0], numbers[1], numbers[2], numbers[3])
