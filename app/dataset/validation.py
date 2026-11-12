"""
Validation is deliberately separate from folder discovery and manifest saving.
Only valid pairs become CocoImage/GroundTruthObject records. Error categories
are stable machine-readable values; details contain filenames and explanations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from app.dataset.records import ClassSpec, CocoImage, GroundTruthObject, RejectionReason
from app.dataset.paths import DatasetPathError, normalize_relative_path, resolve_dataset_path
from app.dataset.image_io import DatasetImageError, read_image_dimensions


class DatasetValidationError(ValueError):
    """A COCO document or its configuration is too ambiguous to import safely."""

    def __init__(self, category: str, detail: str):
        self.category = category
        self.detail = detail
        super().__init__(f"{category}: {detail}")


def safe_relative_path(value: str | Path, *, allow_root: bool = False) -> str:
    """Normalize portable COCO paths; reject absolute, traversal and drive paths.

    Does not follow symlinks. The importer also checks real filesystem containment
    before reading any path.
    """
    try:
        return normalize_relative_path(value, allow_root=allow_root)
    except DatasetPathError as error:
        raise DatasetValidationError("unsafe_path", str(error)) from error


def require_inside_root(root: Path, relative: str) -> Path:
    """Safely resolve a source file, including rejecting outward-pointing symlinks."""
    try:
        return resolve_dataset_path(root, relative)
    except DatasetPathError as error:
        raise DatasetValidationError("unsafe_path", str(error)) from error


def _int_id(value: object) -> bool:
    return type(value) is int


def _positive_int(value: object) -> bool:
    return _int_id(value) and value > 0


def _finite_numeric(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


@dataclass(frozen=True)
class CocoIndex:
    """Checked COCO associations; individual boxes/images still need validation."""

    images_by_path: dict[str, dict]
    annotations_by_id: dict[int, tuple[dict, ...]]
    declared_category_ids: frozenset[int]
    classes: tuple[ClassSpec, ...]


@dataclass(frozen=True)
class ImageCheck:
    """One image-check result used internally by the SCRUM-42 importer."""

    image: CocoImage | None
    objects: tuple[GroundTruthObject, ...]
    reasons: tuple[RejectionReason, ...]

    @property
    def accepted(self) -> bool:
        return self.image is not None and not self.reasons


def index_coco_document(
    document: object,
    class_mapping: Mapping[int, str],
    *,
    filename_base: str = ".",
) -> CocoIndex:
    """Check COCO document relationships and index records by image path.

    Missing/duplicate image IDs, duplicate annotation IDs and orphan references
    make the shared document ambiguous and raise DatasetValidationError.
    Individual malformed annotations with a valid image_id are handled by
    check_image_pair so other good images may still be imported.
    """
    if not isinstance(document, dict) or any(
        not isinstance(document.get(key), list) for key in ("images", "annotations", "categories")
    ):
        raise DatasetValidationError(
            "malformed_annotation_file", "COCO JSON must contain images, annotations and categories arrays."
        )
    if not isinstance(class_mapping, Mapping) or not class_mapping:
        raise DatasetValidationError("invalid_class_mapping", "Supply a nonempty class-ID-to-name mapping.")
    for key, name in class_mapping.items():
        if not _int_id(key) or not isinstance(name, str) or not name.strip():
            raise DatasetValidationError("invalid_class_mapping", "Class IDs must be integers and names nonblank strings.")

    base = safe_relative_path(filename_base, allow_root=True)
    declared: dict[int, str] = {}
    for position, item in enumerate(document["categories"]):
        if not isinstance(item, dict) or not _int_id(item.get("id")) or not isinstance(item.get("name"), str) or not item["name"].strip():
            raise DatasetValidationError("malformed_annotation_file", f"COCO category at index {position} needs integer id and name.")
        category_id = item["id"]
        if category_id in declared:
            raise DatasetValidationError("duplicate_category_id", f"COCO category ID {category_id} appears more than once.")
        declared[category_id] = item["name"]
        if category_id in class_mapping and class_mapping[category_id] != item["name"]:
            raise DatasetValidationError(
                "class_mapping_mismatch",
                f"COCO category {category_id} is named {item['name']!r}, but the supplied mapping says {class_mapping[category_id]!r}.",
            )

    images: dict[str, dict] = {}
    image_ids: set[int] = set()
    for position, item in enumerate(document["images"]):
        if not isinstance(item, dict) or not _int_id(item.get("id")) or not isinstance(item.get("file_name"), str):
            raise DatasetValidationError("malformed_annotation_file", f"COCO image at index {position} needs integer id and file_name.")
        coco_id = item["id"]
        if coco_id in image_ids:
            raise DatasetValidationError("duplicate_image_id", f"COCO image ID {coco_id} occurs more than once.")
        image_ids.add(coco_id)
        rel = safe_relative_path(item["file_name"])
        if base != ".":
            rel = safe_relative_path(f"{base}/{rel}")
        if rel in images:
            raise DatasetValidationError("duplicate_image_path", f"Multiple COCO image entries reference {rel!r}.")
        images[rel] = item

    annotations: dict[int, list[dict]] = {image_id: [] for image_id in image_ids}
    annotation_ids: set[int] = set()
    for position, item in enumerate(document["annotations"]):
        if not isinstance(item, dict) or not _int_id(item.get("image_id")):
            raise DatasetValidationError("malformed_annotation_file", f"Annotation at index {position} lacks a valid image_id; cannot associate an error with a filename.")
        image_id = item["image_id"]
        if image_id not in image_ids:
            raise DatasetValidationError("orphan_annotation", f"Annotation at index {position} references nonexistent COCO image ID {image_id}.")
        annotation_id = item.get("id")
        if _int_id(annotation_id):
            if annotation_id in annotation_ids:
                raise DatasetValidationError("duplicate_annotation_id", f"COCO annotation ID {annotation_id} is duplicated.")
            annotation_ids.add(annotation_id)
        annotations[image_id].append(item)

    classes = tuple(ClassSpec(category_id, name) for category_id, name in sorted(declared.items()) if category_id in class_mapping)
    return CocoIndex(images, {key: tuple(value) for key, value in annotations.items()}, frozenset(declared), classes)


def check_image_pair(
    image_path: Path,
    *,
    image_name: str,
    coco_entry: dict | None,
    annotations: Sequence[dict],
    declared_category_ids: frozenset[int],
    class_mapping: Mapping[int, str],
    allow_negative_images: bool,
) -> ImageCheck:
    """Validate one JPG/PNG and its COCO annotations; never modify source files.

    Any bad object rejects the whole image/annotation pair, not just the object.
    Missing COCO image entries cannot be considered intentional negatives.
    """
    reasons: list[RejectionReason] = []
    image = None
    image_width = image_height = None
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        reasons.append(RejectionReason("unsupported_image", f"{image_name}: only JPG and PNG images are supported."))
    else:
        try:
            image_width, image_height = read_image_dimensions(image_path)
        except (DatasetImageError, OSError) as exc:
            reasons.append(RejectionReason("unreadable_image", f"{image_name}: {exc}"))

    if coco_entry is None:
        reasons.append(RejectionReason("missing_annotation", f"{image_name}: no matching COCO images entry."))
        return ImageCheck(None, (), tuple(reasons))

    coco_id = coco_entry["id"]  # index_coco_document already validated this relationship
    if not _positive_int(coco_entry.get("width")) or not _positive_int(coco_entry.get("height")):
        reasons.append(RejectionReason("malformed_annotation", f"{image_name}: COCO image width and height must be positive integers."))
    elif image_width is not None and (coco_entry["width"], coco_entry["height"]) != (image_width, image_height):
        reasons.append(RejectionReason(
            "dimension_mismatch", f"{image_name}: COCO declares {coco_entry['width']}x{coco_entry['height']}, actual image is {image_width}x{image_height}."
        ))

    valid_objects: list[GroundTruthObject] = []
    for position, item in enumerate(annotations):
        tag = f"{image_name}, annotation index {position}"
        if not _int_id(item.get("id")) or not _int_id(item.get("category_id")) or "bbox" not in item:
            reasons.append(RejectionReason("malformed_annotation", f"{tag}: missing or invalid id, category_id, or bbox."))
            continue
        category_id = item["category_id"]
        if category_id not in declared_category_ids or category_id not in class_mapping:
            reasons.append(RejectionReason("unknown_class_id", f"{tag}: unknown or unapproved category ID {category_id}."))
            continue
        bbox = item["bbox"]
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(not _finite_numeric(value) for value in bbox):
            reasons.append(RejectionReason("invalid_bbox", f"{tag}: bbox must have four finite numeric values [x, y, width, height]."))
            continue
        x, y, width, height = (float(value) for value in bbox)
        if x < 0 or y < 0 or width <= 0 or height <= 0 or (image_width is not None and (x + width > image_width or y + height > image_height)):
            reasons.append(RejectionReason("invalid_bbox", f"{tag}: bbox {bbox!r} has invalid dimensions or extends beyond the image."))
            continue
        # We do not create an accepted record unless image decoding has succeeded.
        valid_objects.append(GroundTruthObject(item["id"], coco_id, category_id, (x, y, width, height)))

    if not annotations and not allow_negative_images:
        reasons.append(RejectionReason("negative_image_not_allowed", f"{image_name}: zero annotations; negative images are disabled by policy."))

    if reasons or image_width is None or image_height is None:
        return ImageCheck(None, (), tuple(reasons))
    image = CocoImage(coco_id, image_name, image_width, image_height)
    return ImageCheck(image, tuple(valid_objects), ())
