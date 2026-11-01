"""Summarize accepted and rejected records from a DatasetInput.

Counts use the existing COCO join: an object belongs to an accepted image
when the annotation file's relative path and ``coco_image_id`` both match.
A negative image is an accepted image with exactly one COCO image entry and
zero joined objects. A missing or repeated match is an error, not a negative.
This module does not draw previews, write reports, or validate image files
or bounding boxes.
"""

from dataclasses import dataclass

from app.dataset.records import (
    AcceptedImage,
    DatasetInput,
    ParsedAnnotations,
    RejectionReason,
    UNKNOWN_SOURCE,
    coco_image_for,
    objects_for_image,
    relative_path_identity,
    source_label,
)


class SummaryError(ValueError):
    """The dataset cannot be summarized because an association or class mapping is unclear."""


@dataclass(frozen=True)
class GroupCount:
    """Accepted-image count for one lot or session label.

    Missing source values use ``UNKNOWN_SOURCE``.
    """

    label: str
    image_count: int


@dataclass(frozen=True)
class ClassSummary:
    """Annotation and image counts for one configured category id."""

    category_id: int
    name: str
    annotation_count: int
    image_count: int


@dataclass(frozen=True)
class RejectedImageSummary:
    """One rejected image. ``reasons`` may contain several categories."""

    filename: str
    image_path: str
    image_id: str | None
    reasons: tuple[RejectionReason, ...]


@dataclass(frozen=True)
class DatasetSummary:
    accepted_image_count: int
    rejected_image_count: int
    images_by_lot: tuple[GroupCount, ...]
    images_by_session: tuple[GroupCount, ...]
    classes: tuple[ClassSummary, ...]
    negative_image_count: int
    rejected_images: tuple[RejectedImageSummary, ...]


def summarize(dataset: DatasetInput) -> DatasetSummary:
    """Return counts for ``dataset``, including accepted images with missing provenance."""
    class_names = _configured_classes(dataset)
    annotation_counts = {category_id: 0 for category_id in class_names}
    images_by_class: dict[int, set[str]] = {category_id: set() for category_id in class_names}
    negative_image_count = 0

    for image in dataset.accepted:
        annotation_file = _annotation_file(dataset, image)
        _require_one_coco_image(dataset, annotation_file, image)
        objects = objects_for_image(dataset, image)
        if not objects:
            negative_image_count += 1
            continue
        present: set[int] = set()
        for obj in objects:
            if obj.category_id not in class_names:
                raise SummaryError(
                    f"Accepted image {image.image_id} annotation {obj.annotation_id} "
                    f"uses category {obj.category_id}, which has no configured class name."
                )
            annotation_counts[obj.category_id] += 1
            present.add(obj.category_id)
        for category_id in present:
            images_by_class[category_id].add(image.image_id)

    classes = tuple(
        ClassSummary(
            category_id=category_id,
            name=class_names[category_id],
            annotation_count=annotation_counts[category_id],
            image_count=len(images_by_class[category_id]),
        )
        for category_id in sorted(class_names)
    )
    rejected_images = tuple(_rejected_image(item) for item in dataset.rejected)
    return DatasetSummary(
        accepted_image_count=len(dataset.accepted),
        rejected_image_count=len(rejected_images),
        images_by_lot=_group_counts(dataset.accepted, lambda image: source_label(image.source_lot)),
        images_by_session=_group_counts(
            dataset.accepted, lambda image: source_label(image.source_session)
        ),
        classes=classes,
        negative_image_count=negative_image_count,
        rejected_images=rejected_images,
    )


def _configured_classes(dataset: DatasetInput) -> dict[int, str]:
    names: dict[int, str] = {}
    for parsed in dataset.annotations:
        for spec in parsed.classes:
            current = names.get(spec.category_id)
            if current is None:
                names[spec.category_id] = spec.name
                continue
            if current != spec.name:
                raise SummaryError(
                    f"Category {spec.category_id} is mapped to both {current!r} and {spec.name!r}."
                )
    return names


def _annotation_file(dataset: DatasetInput, image: AcceptedImage) -> ParsedAnnotations:
    identity = relative_path_identity(image.annotation_path)
    matches = tuple(
        parsed
        for parsed in dataset.annotations
        if relative_path_identity(parsed.path) == identity
    )
    if not matches:
        raise SummaryError(
            f"Accepted image {image.image_id} has no annotation file {identity}."
        )
    if len(matches) > 1:
        raise SummaryError(
            f"Accepted image {image.image_id} matches {len(matches)} annotation files at {identity}."
        )
    return matches[0]


def _require_one_coco_image(
    dataset: DatasetInput, annotation_file: ParsedAnnotations, image: AcceptedImage
) -> None:
    identity = relative_path_identity(annotation_file.path)
    matches = tuple(
        entry
        for entry in annotation_file.images
        if entry.coco_image_id == image.coco_image_id
    )
    if not matches or coco_image_for(dataset, image) is None:
        raise SummaryError(
            f"Accepted image {image.image_id} has no COCO image {image.coco_image_id} in {identity}."
        )
    if len(matches) > 1:
        raise SummaryError(
            f"Accepted image {image.image_id} has {len(matches)} COCO image entries "
            f"for id {image.coco_image_id} in {identity}."
        )


def _group_counts(images: tuple[AcceptedImage, ...], label_of) -> tuple[GroupCount, ...]:
    counts: dict[str, int] = {}
    for image in images:
        label = label_of(image)
        counts[label] = counts.get(label, 0) + 1
    return tuple(GroupCount(label, counts[label]) for label in sorted(counts))


def _rejected_image(rejected) -> RejectedImageSummary:
    image_path = relative_path_identity(rejected.image_path)
    return RejectedImageSummary(
        filename=image_path.rsplit("/", 1)[-1],
        image_path=image_path,
        image_id=rejected.image_id,
        reasons=rejected.reasons,
    )
