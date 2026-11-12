"""Export approved dataset records as RF-DETR-compatible COCO splits.

The exporter keeps the approved COCO image and category IDs, includes
accepted negative images, and records source provenance in both the COCO
image entries and a companion ``dataset_provenance.json`` file.  It validates
the complete input before publishing anything.

Command-line use::

    python -m app.dataset.rfdetr_export DATASET_ROOT MANIFEST SPLIT_MANIFEST \
        CLASS_MAP OUTPUT [--frozen-test-lock LOCK]

``CLASS_MAP`` may be ``{"1": "almond"}``, a ``categories`` object, or a
plain COCO category list.  Split-manifest rows are matched by ``image_id``
when present, otherwise by their ``source`` or ``output`` path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from PIL import Image, UnidentifiedImageError

from app.dataset.frozen_test_set import FrozenSetLeakError, assert_excluded, load_test_lock
from app.dataset.manifest import DatasetManifest, build_manifest
from app.dataset.partitions import build_partition_manifest
from app.dataset.paths import relative_dataset_path, relative_path_identity, resolve_dataset_path
from app.dataset.records import (
    AcceptedImage,
    ClassSpec,
    CocoImage,
    DatasetInput,
    GroundTruthObject,
    ParsedAnnotations,
    SourcePolicy,
    coco_image_for,
    objects_for_image,
)


COCO_FILE_NAME = "_annotations.coco.json"
PROVENANCE_FILE_NAME = "dataset_provenance.json"
EXPORT_SCHEMA_VERSION = 1
FITTING_SPLITS = ("train", "valid")
KNOWN_SPLITS = frozenset((*FITTING_SPLITS, "test"))


class RfdetrExportError(ValueError):
    """An identifiable source-data error prevented the export."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        image_id: str | None = None,
        annotation_id: int | None = None,
    ) -> None:
        self.code = code
        self.image_id = image_id
        self.annotation_id = annotation_id
        context = []
        if image_id is not None:
            context.append(f"image_id={image_id}")
        if annotation_id is not None:
            context.append(f"annotation_id={annotation_id}")
        suffix = f" ({', '.join(context)})" if context else ""
        super().__init__(f"[{code}] {message}{suffix}")


@dataclass(frozen=True)
class SplitAssignment:
    image_id: str
    split: str
    source_group_id: str

    def __post_init__(self) -> None:
        image_id = self.image_id.strip()
        split = self.split.strip().lower()
        group = self.source_group_id.strip()
        if not image_id:
            raise RfdetrExportError("INVALID_ASSIGNMENT", "An assignment needs an image_id.")
        if split not in KNOWN_SPLITS:
            raise RfdetrExportError(
                "INVALID_ASSIGNMENT",
                f"Split must be one of {', '.join(sorted(KNOWN_SPLITS))}: {self.split!r}.",
                image_id=image_id,
            )
        if not group:
            raise RfdetrExportError(
                "INVALID_ASSIGNMENT", "An assignment needs a source_group_id.", image_id=image_id
            )
        object.__setattr__(self, "image_id", image_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "source_group_id", group)


@dataclass(frozen=True)
class ExportResult:
    output_directory: Path
    split_counts: Mapping[str, int]
    annotation_counts: Mapping[str, int]


def export_rfdetr_dataset(
    dataset: DatasetInput,
    manifest: DatasetManifest,
    assignments: Iterable[SplitAssignment],
    class_map: Mapping[int, str],
    output_directory: str | Path,
    *,
    frozen_test_lock: Mapping | None = None,
) -> ExportResult:
    """Validate and publish ``train`` and ``valid`` COCO inputs.

    The destination must not already exist.  Validation and staging finish
    before the destination is published, so a failed conversion does not
    leave a partial RF-DETR dataset behind.
    """
    output = Path(output_directory).resolve()
    if output.exists():
        raise RfdetrExportError("OUTPUT_EXISTS", f"Output already exists: {output}")

    approved_classes = _validated_class_map(class_map)
    assignment_by_id = _validated_assignments(manifest, assignments)
    _validate_manifest(dataset, manifest)
    _validate_source_classes(dataset, approved_classes)
    _validate_group_isolation(assignment_by_id.values())

    if frozen_test_lock is not None:
        try:
            fitting = [item for item in assignment_by_id.values() if item.split != "test"]
            assert_excluded(dict(frozen_test_lock), group_ids=[item.source_group_id for item in fitting])
            frozen_ids = {item.get("image_id") for item in frozen_test_lock["files"]}
            for item in fitting:
                if item.image_id in frozen_ids:
                    raise FrozenSetLeakError(f"Frozen test image ID {item.image_id} is assigned to {item.split}.")
        except FrozenSetLeakError as error:
            raise RfdetrExportError("FROZEN_TEST_LEAK", str(error)) from error
        except ValueError as error:
            raise RfdetrExportError("INVALID_TEST_LOCK", str(error)) from error

    prepared: dict[str, list[tuple[AcceptedImage, SplitAssignment, list[dict]]]] = {
        split: [] for split in FITTING_SPLITS
    }
    used_image_ids: dict[int, str] = {}
    used_annotation_ids: dict[str, dict[int, str]] = {split: {} for split in FITTING_SPLITS}

    for image in sorted(dataset.accepted, key=lambda item: item.image_id):
        assignment = assignment_by_id[image.image_id]
        if assignment.split == "test":
            continue
        _validate_unique_image_id(image, used_image_ids)
        source_path = dataset.resolve(image.image_path)
        _validate_image_file(image, source_path)
        _validate_coco_image_entry(dataset, image)
        annotations = _annotations_for_image(
            dataset,
            image,
            approved_classes,
            used_annotation_ids[assignment.split],
        )
        prepared[assignment.split].append((image, assignment, annotations))

    fitting_rows = [row for rows in prepared.values() for row in rows]
    if frozen_test_lock is not None:
        try:
            assert_excluded(
                dict(frozen_test_lock),
                paths=[dataset.resolve(image.image_path) for image, _, _ in fitting_rows],
                group_ids=[assignment.source_group_id for _, assignment, _ in fitting_rows],
            )
        except FrozenSetLeakError as error:
            raise RfdetrExportError("FROZEN_TEST_LEAK", str(error)) from error
        except ValueError as error:
            raise RfdetrExportError("INVALID_TEST_LOCK", str(error)) from error

    documents = {
        split: _coco_document(split, rows, manifest, approved_classes)
        for split, rows in prepared.items()
    }
    provenance = _provenance_document(prepared, manifest, approved_classes)
    _publish(dataset, output, prepared, documents, provenance)
    return ExportResult(
        output_directory=output,
        split_counts={split: len(rows) for split, rows in prepared.items()},
        annotation_counts={
            split: sum(len(annotations) for _, _, annotations in rows)
            for split, rows in prepared.items()
        },
    )


def _validated_class_map(class_map: Mapping[int, str]) -> dict[int, str]:
    classes: dict[int, str] = {}
    for raw_id, raw_name in class_map.items():
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise RfdetrExportError("INVALID_CLASS_MAP", f"Class ID must be an integer: {raw_id!r}.")
        name = str(raw_name).strip()
        if not name:
            raise RfdetrExportError("INVALID_CLASS_MAP", f"Class {raw_id} has no name.")
        classes[raw_id] = name
    if not classes:
        raise RfdetrExportError("INVALID_CLASS_MAP", "The approved class map is empty.")
    if len(set(classes.values())) != len(classes):
        raise RfdetrExportError("INVALID_CLASS_MAP", "Approved class names must be unique.")
    return dict(sorted(classes.items()))


def _validated_assignments(
    manifest: DatasetManifest, assignments: Iterable[SplitAssignment]
) -> dict[str, SplitAssignment]:
    result: dict[str, SplitAssignment] = {}
    for assignment in assignments:
        if assignment.image_id in result:
            raise RfdetrExportError(
                "DUPLICATE_ASSIGNMENT",
                "An image has more than one split assignment.",
                image_id=assignment.image_id,
            )
        result[assignment.image_id] = assignment
    expected = {image.image_id for image in manifest.images}
    unknown = sorted(set(result) - expected)
    missing = sorted(expected - set(result))
    if unknown:
        raise RfdetrExportError(
            "UNKNOWN_IMAGE", "Split assignment is not present in the source manifest.", image_id=unknown[0]
        )
    if missing:
        raise RfdetrExportError(
            "MISSING_ASSIGNMENT", "Source-manifest image has no split assignment.", image_id=missing[0]
        )
    return result


def _validate_manifest(dataset: DatasetInput, manifest: DatasetManifest) -> None:
    rebuilt = build_manifest(dataset, manifest.dataset_version)
    if rebuilt.to_dict() != manifest.to_dict():
        raise RfdetrExportError(
            "MANIFEST_MISMATCH",
            "The versioned manifest does not match the supplied dataset records and labels.",
        )


def _validate_source_classes(dataset: DatasetInput, approved: Mapping[int, str]) -> None:
    for parsed in dataset.annotations:
        for source_class in parsed.classes:
            expected = approved.get(source_class.category_id)
            if expected is None:
                raise RfdetrExportError(
                    "UNKNOWN_CLASS",
                    f"Class ID {source_class.category_id} in {relative_path_identity(parsed.path)} "
                    "is absent from the approved class map.",
                )
            if source_class.name != expected:
                raise RfdetrExportError(
                    "CLASS_NAME_MISMATCH",
                    f"Class ID {source_class.category_id} is {source_class.name!r} in labels "
                    f"but {expected!r} in the approved class map.",
                )


def _validate_group_isolation(assignments: Iterable[SplitAssignment]) -> None:
    splits_by_group: dict[str, set[str]] = {}
    for assignment in assignments:
        splits_by_group.setdefault(assignment.source_group_id, set()).add(assignment.split)
    for group, splits in sorted(splits_by_group.items()):
        if len(splits) > 1:
            raise RfdetrExportError(
                "SOURCE_GROUP_LEAK",
                f"Source group {group!r} appears in multiple splits: {', '.join(sorted(splits))}.",
            )


def _validate_unique_image_id(image: AcceptedImage, used: dict[int, str]) -> None:
    previous = used.get(image.coco_image_id)
    if previous is not None:
        raise RfdetrExportError(
            "DUPLICATE_COCO_IMAGE_ID",
            f"COCO image ID {image.coco_image_id} is also used by {previous}; preserving both "
            "would produce an ambiguous RF-DETR dataset.",
            image_id=image.image_id,
        )
    used[image.coco_image_id] = image.image_id


def _validate_image_file(image: AcceptedImage, path: Path) -> None:
    if not path.is_file():
        raise RfdetrExportError("MISSING_IMAGE", f"Image file does not exist: {path}", image_id=image.image_id)
    try:
        with Image.open(path) as opened:
            opened.load()
            actual = opened.size
    except (OSError, UnidentifiedImageError) as error:
        raise RfdetrExportError(
            "UNREADABLE_IMAGE", f"Image cannot be decoded: {path}: {error}", image_id=image.image_id
        ) from error
    expected = (image.width, image.height)
    if actual != expected:
        raise RfdetrExportError(
            "IMAGE_DIMENSION_MISMATCH",
            f"Manifest dimensions {expected[0]}x{expected[1]} do not match file dimensions "
            f"{actual[0]}x{actual[1]}.",
            image_id=image.image_id,
        )


def _validate_coco_image_entry(dataset: DatasetInput, image: AcceptedImage) -> None:
    source = coco_image_for(dataset, image)
    if source is None:
        raise RfdetrExportError(
            "MISSING_LABEL_IMAGE", "Annotation file has no matching images entry.", image_id=image.image_id
        )
    if (source.width, source.height) != (image.width, image.height):
        raise RfdetrExportError(
            "LABEL_DIMENSION_MISMATCH",
            f"Label dimensions {source.width}x{source.height} do not match manifest dimensions "
            f"{image.width}x{image.height}.",
            image_id=image.image_id,
        )
    if relative_path_identity(source.file_name) != relative_path_identity(image.image_path):
        raise RfdetrExportError(
            "IMAGE_PATH_MISMATCH",
            f"Label image path {source.file_name!r} does not match manifest path "
            f"{relative_path_identity(image.image_path)!r}.",
            image_id=image.image_id,
        )


def _annotations_for_image(
    dataset: DatasetInput,
    image: AcceptedImage,
    approved: Mapping[int, str],
    used_ids: dict[int, str],
) -> list[dict]:
    exported: list[dict] = []
    for obj in sorted(objects_for_image(dataset, image), key=lambda item: item.annotation_id):
        if obj.category_id not in approved:
            raise RfdetrExportError(
                "UNKNOWN_CLASS",
                f"Annotation uses unapproved class ID {obj.category_id}.",
                image_id=image.image_id,
                annotation_id=obj.annotation_id,
            )
        previous = used_ids.get(obj.annotation_id)
        if previous is not None:
            raise RfdetrExportError(
                "DUPLICATE_ANNOTATION_ID",
                f"Annotation ID is also used by image {previous} in this split.",
                image_id=image.image_id,
                annotation_id=obj.annotation_id,
            )
        used_ids[obj.annotation_id] = image.image_id
        x, y, width, height = obj.bbox_xywh
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            _invalid_box(image, obj, "coordinates must be finite")
        if x < 0 or y < 0:
            _invalid_box(image, obj, "x and y must be non-negative")
        if width <= 0 or height <= 0:
            _invalid_box(image, obj, "width and height must be positive")
        if x + width > image.width or y + height > image.height:
            _invalid_box(image, obj, "box extends outside the source image")
        exported.append(
            {
                "id": obj.annotation_id,
                "image_id": image.coco_image_id,
                "category_id": obj.category_id,
                "bbox": [x, y, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
    return exported


def _invalid_box(image: AcceptedImage, obj: GroundTruthObject, reason: str) -> None:
    raise RfdetrExportError(
        "INVALID_BOX",
        f"Bounding box {list(obj.bbox_xywh)} is invalid: {reason}.",
        image_id=image.image_id,
        annotation_id=obj.annotation_id,
    )


def _coco_document(
    split: str,
    rows: list[tuple[AcceptedImage, SplitAssignment, list[dict]]],
    manifest: DatasetManifest,
    classes: Mapping[int, str],
) -> dict:
    images = []
    annotations = []
    for image, assignment, image_annotations in rows:
        images.append(
            {
                "id": image.coco_image_id,
                "file_name": relative_path_identity(image.image_path),
                "width": image.width,
                "height": image.height,
                "stable_image_id": image.image_id,
                "source_group_id": assignment.source_group_id,
                "source_lot": image.source_lot,
                "source_session": image.source_session,
                "near_duplicate_group_id": image.near_duplicate_group_id,
            }
        )
        annotations.extend(image_annotations)
    return {
        "info": {
            "description": f"RF-DETR {split} input",
            "dataset_version": manifest.dataset_version,
            "dataset_content_hash": manifest.content_hash,
            "export_schema_version": EXPORT_SCHEMA_VERSION,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_id, "name": name, "supercategory": "object"} for category_id, name in classes.items()],
    }


def _provenance_document(
    prepared: Mapping[str, list[tuple[AcceptedImage, SplitAssignment, list[dict]]]],
    manifest: DatasetManifest,
    classes: Mapping[int, str],
) -> dict:
    rows = []
    for split in FITTING_SPLITS:
        for image, assignment, annotations in prepared[split]:
            rows.append(
                {
                    "image_id": image.image_id,
                    "coco_image_id": image.coco_image_id,
                    "image_path": relative_path_identity(image.image_path),
                    "annotation_path": relative_path_identity(image.annotation_path),
                    "split": split,
                    "source_group_id": assignment.source_group_id,
                    "source_lot": image.source_lot,
                    "source_session": image.source_session,
                    "near_duplicate_group_id": image.near_duplicate_group_id,
                    "annotation_count": len(annotations),
                }
            )
    return {
        "export_schema_version": EXPORT_SCHEMA_VERSION,
        "dataset_version": manifest.dataset_version,
        "dataset_content_hash": manifest.content_hash,
        "class_map": {str(category_id): name for category_id, name in classes.items()},
        "images": rows,
    }


def _publish(
    dataset: DatasetInput,
    output: Path,
    prepared: Mapping[str, list[tuple[AcceptedImage, SplitAssignment, list[dict]]]],
    documents: Mapping[str, dict],
    provenance: dict,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        for split in FITTING_SPLITS:
            split_root = staging / split
            split_root.mkdir()
            for image, _, _ in prepared[split]:
                relative = relative_dataset_path(image.image_path)
                destination = split_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dataset.resolve(relative), destination)
            _write_json(split_root / COCO_FILE_NAME, documents[split])
        _write_json(staging / PROVENANCE_FILE_NAME, provenance)
        try:
            os.rename(staging, output)
        except FileExistsError as error:
            raise RfdetrExportError("OUTPUT_EXISTS", f"Output already exists: {output}") from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_json(path: Path, document: dict) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_file_inputs(
    dataset_root: str | Path,
    manifest_path: str | Path,
    split_manifest_path: str | Path,
) -> tuple[DatasetInput, DatasetManifest, tuple[SplitAssignment, ...]]:
    """Load repository manifest files into the validated export records."""
    root = Path(dataset_root).resolve()
    manifest_document = _read_json(manifest_path, "versioned manifest")
    source_policy = manifest_document.get("source_policy") or {}
    manifest_images = manifest_document.get("images")
    if not isinstance(manifest_images, list):
        raise RfdetrExportError("INVALID_MANIFEST", "Versioned manifest has no images list.")

    parsed_by_path: dict[str, ParsedAnnotations] = {}
    accepted: list[AcceptedImage] = []
    for row in manifest_images:
        annotation_path = relative_path_identity(row["annotation_path"])
        if annotation_path not in parsed_by_path:
            parsed_by_path[annotation_path] = _load_annotations(root, annotation_path)
        parsed = parsed_by_path[annotation_path]
        coco_id = row["coco_image_id"]
        source_image = next((item for item in parsed.images if item.coco_image_id == coco_id), None)
        if source_image is None:
            raise RfdetrExportError(
                "MISSING_LABEL_IMAGE",
                "Annotation file has no matching images entry.",
                image_id=str(row.get("image_id")),
            )
        accepted.append(
            AcceptedImage(
                image_id=row["image_id"],
                coco_image_id=coco_id,
                image_path=row["image_path"],
                annotation_path=annotation_path,
                width=source_image.width,
                height=source_image.height,
                source_lot=row.get("source_lot"),
                source_session=row.get("source_session"),
                near_duplicate_group_id=row.get("near_duplicate_group_id"),
            )
        )
    dataset = DatasetInput(
        root=root,
        source_policy=SourcePolicy(
            require_lot=bool(source_policy.get("require_lot")),
            require_session=bool(source_policy.get("require_session")),
        ),
        accepted=tuple(accepted),
        rejected=(),
        annotations=tuple(parsed_by_path.values()),
    )
    rebuilt = build_manifest(dataset, str(manifest_document.get("dataset_version", "")))
    if rebuilt.to_dict() != manifest_document:
        raise RfdetrExportError(
            "MANIFEST_MISMATCH", "Versioned manifest content or hash does not match its labels."
        )
    assignments = _assignments_from_file(split_manifest_path, rebuilt, dataset)
    return dataset, rebuilt, assignments


def _load_annotations(root: Path, relative_path: str) -> ParsedAnnotations:
    path = resolve_dataset_path(root, relative_path)
    document = _read_json(path, "annotation file")
    try:
        return ParsedAnnotations(
            path=relative_path,
            classes=tuple(ClassSpec(item["id"], item["name"]) for item in document["categories"]),
            images=tuple(
                CocoImage(item["id"], item["file_name"], item["width"], item["height"])
                for item in document["images"]
            ),
            objects=tuple(
                GroundTruthObject(item["id"], item["image_id"], item["category_id"], tuple(item["bbox"]))
                for item in document["annotations"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RfdetrExportError("INVALID_LABEL_FILE", f"{relative_path}: {error}") from error


def _assignments_from_file(
    path: str | Path, manifest: DatasetManifest, dataset: DatasetInput
) -> tuple[SplitAssignment, ...]:
    document = _read_json(path, "split manifest")
    rows = document.get("images")
    if not isinstance(rows, list):
        raise RfdetrExportError("INVALID_SPLIT_MANIFEST", "Split manifest has no images list.")
    if document.get("kind") == "labeled_dataset_partitions":
        # Rebuild the refactor's checked handoff so hashes, objects, class map,
        # dimensions and transitive source groups cannot silently go stale.
        try:
            choices = {row["image_id"]: row["split"] for row in rows}
            if len(choices) != len(rows):
                raise ValueError("Duplicate image IDs in the partition manifest.")
            checked = build_partition_manifest(dataset, manifest.dataset_version, choices)
            if checked.to_dict() != document:
                raise ValueError("Partition metadata, labels, version or hashes do not match the source dataset.")
        except (KeyError, TypeError, ValueError) as error:
            raise RfdetrExportError("PARTITION_MISMATCH", str(error)) from error
        return tuple(SplitAssignment(item.image.image_id, item.split, item.group_id)
                     for item in checked.images)
    by_id = {image.image_id: image for image in manifest.images}
    by_path = {image.image_path: image for image in manifest.images}
    assignments: list[SplitAssignment] = []
    for row in rows:
        source = row.get("source")
        output = row.get("output")
        match = by_id.get(str(row.get("image_id", "")))
        if match is None and source is not None:
            match = by_path.get(relative_path_identity(source))
        if match is None and output is not None:
            match = by_path.get(relative_path_identity(output))
        if match is None:
            continue
        assignments.append(SplitAssignment(match.image_id, row.get("split", ""), row.get("group_id", "")))
    return tuple(assignments)


def load_class_map(path: str | Path) -> dict[int, str]:
    document = _read_json(path, "class map")
    if isinstance(document, dict) and "categories" in document:
        document = document["categories"]
    if isinstance(document, list):
        try:
            return _validated_class_map({item["id"]: item["name"] for item in document})
        except (KeyError, TypeError) as error:
            raise RfdetrExportError("INVALID_CLASS_MAP", str(error)) from error
    if isinstance(document, dict):
        try:
            return _validated_class_map({int(key): value for key, value in document.items()})
        except (TypeError, ValueError) as error:
            raise RfdetrExportError("INVALID_CLASS_MAP", str(error)) from error
    raise RfdetrExportError("INVALID_CLASS_MAP", "Class map must be an object or COCO category list.")


def _read_json(path: str | Path, label: str) -> dict | list:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise RfdetrExportError("INVALID_JSON", f"Cannot read {label} {path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("class_map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen-test-lock", type=Path)
    args = parser.parse_args(argv)
    try:
        dataset, manifest, assignments = load_file_inputs(
            args.dataset_root, args.manifest, args.split_manifest
        )
        lock = load_test_lock(args.frozen_test_lock) if args.frozen_test_lock else None
        result = export_rfdetr_dataset(
            dataset,
            manifest,
            assignments,
            load_class_map(args.class_map),
            args.output,
            frozen_test_lock=lock,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Exported RF-DETR dataset to {result.output_directory}")
    for split in FITTING_SPLITS:
        print(
            f"{split}: {result.split_counts[split]} images, "
            f"{result.annotation_counts[split]} annotations"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
