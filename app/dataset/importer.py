"""SCRUM-42: read a COCO image collection and construct Zakari's DatasetInput.

The importer is stateless and read-only. Each call returns a fresh inventory
with deterministic, dataset-scoped image IDs. It never persists records or
modifies the original dataset. One call imports one COCO annotation JSON.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Mapping

from app.dataset.records import (
    AcceptedImage,
    DatasetInput,
    ParsedAnnotations,
    RejectedImage,
    RejectionReason,
    SourcePolicy,
)
from app.dataset.validation import (
    DatasetValidationError,
    check_image_pair,
    index_coco_document,
    require_inside_root,
    safe_relative_path,
)

# We report supported/commonly used unsupported image formats, not arbitrary
# text/JSON files. Any format explicitly listed in COCO is always checked.
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def _stable_image_id(namespace: str, relative_path: str) -> str:
    """Stable across root relocation and dataset versions; changes if renamed."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"dinos-dataset/{namespace}/{relative_path}"))


def import_coco_dataset(
    dataset_root: str | Path,
    *,
    annotation_file: str | Path,
    class_mapping: Mapping[int, str],
    dataset_namespace: str,
    image_directory: str | Path = ".",
    coco_filename_base: str | Path = ".",
    allow_negative_images: bool = False,
    source_policy: SourcePolicy | None = None,
    provenance_by_image: Mapping[str, Mapping[str, str | None]] | None = None,
) -> DatasetInput:
    """Return a validated DatasetInput from one COCO images collection.

    All path arguments are relative to dataset_root. COCO file_name paths
    are resolved relative to coco_filename_base. For many RF-DETR exports,
    invoke this on the train/valid/test folder with annotation_file set to
    ``_annotations.coco.json``, image_directory='.', filename_base='.'.

    Missing lot/session values remain None. Accepted images without required
    provenance are *not* excluded, but Zakari's split-eligibility functions
    will flag them. dataset_namespace must identify the source dataset
    independently of its version or absolute filesystem location.

    Raises DatasetValidationError for document-wide problems (missing/broken
    COCO JSON, ambiguous IDs, unsafe paths). Individual image failures become
    RejectedImage entries with structured reasons.
    """
    root = Path(dataset_root).resolve()
    if not root.is_dir():
        raise DatasetValidationError("missing_dataset", f"Dataset directory does not exist: {root}.")
    if not isinstance(dataset_namespace, str) or not dataset_namespace.strip():
        raise DatasetValidationError("missing_dataset_namespace", "Provide a stable, nonblank dataset_namespace.")
    namespace = dataset_namespace.strip()
    if type(allow_negative_images) is not bool:
        raise DatasetValidationError("invalid_negative_policy", "allow_negative_images must be an explicit boolean.")
    policy = source_policy if source_policy is not None else SourcePolicy(True, True)
    if not isinstance(policy, SourcePolicy):
        raise DatasetValidationError("invalid_source_policy", "source_policy must be a SourcePolicy instance.")

    annotation_rel = safe_relative_path(annotation_file)
    image_dir_rel = safe_relative_path(image_directory, allow_root=True)
    base_rel = safe_relative_path(coco_filename_base, allow_root=True)
    annotation_path = require_inside_root(root, annotation_rel)
    image_dir = root if image_dir_rel == "." else require_inside_root(root, image_dir_rel)
    if not image_dir.is_dir():
        raise DatasetValidationError("missing_image_directory", f"Image directory does not exist: {image_dir_rel}.")
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise DatasetValidationError("missing_annotation_file", f"COCO annotation file not found: {annotation_rel}.") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("malformed_annotation_file", f"Cannot read COCO annotation file {annotation_rel}: {exc}.") from exc

    index = index_coco_document(document, class_mapping, filename_base=base_rel)
    if image_dir_rel != ".":
        for image_rel in index.images_by_path:
            if not image_rel.startswith(image_dir_rel + "/"):
                raise DatasetValidationError(
                    "image_outside_collection",
                    f"COCO image {image_rel!r} is not inside image_directory {image_dir_rel!r}.",
                )

    metadata: dict[str, Mapping[str, str | None]] = {}
    if provenance_by_image is not None:
        for key, value in provenance_by_image.items():
            rel = safe_relative_path(key)
            if not isinstance(value, Mapping):
                raise DatasetValidationError("invalid_provenance", f"Provenance for {rel} must be a mapping.")
            for field in ("source_lot", "source_session", "near_duplicate_group_id"):
                if field in value and value[field] is not None and not isinstance(value[field], str):
                    raise DatasetValidationError("invalid_provenance", f"{rel}: {field} must be a string or None.")
            metadata[rel] = value

    candidate_paths: set[str] = set(index.images_by_path)
    for path in image_dir.rglob("*"):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        # Preserve symlink paths instead of path.resolve() here. The explicit
        # containment check below catches symlinks pointing outside the root.
        candidate_paths.add(path.relative_to(root).as_posix())

    accepted: list[AcceptedImage] = []
    rejected: list[RejectedImage] = []
    parsed_images = []
    parsed_objects = []
    for image_rel in sorted(candidate_paths):
        image_id = _stable_image_id(namespace, image_rel)
        try:
            path = require_inside_root(root, image_rel)
        except DatasetValidationError as exc:
            rejected.append(RejectedImage(Path(image_rel), image_id, (RejectionReason(exc.category, exc.detail),)))
            continue
        coco_entry = index.images_by_path.get(image_rel)
        objects = index.annotations_by_id.get(coco_entry["id"], ()) if coco_entry else ()
        checked = check_image_pair(
            path,
            image_name=image_rel,
            coco_entry=coco_entry,
            annotations=objects,
            declared_category_ids=index.declared_category_ids,
            class_mapping=class_mapping,
            allow_negative_images=allow_negative_images,
        )
        if not checked.accepted:
            rejected.append(RejectedImage(Path(image_rel), image_id, checked.reasons))
            continue
        extra = metadata.get(image_rel, {})
        accepted.append(AcceptedImage(
            image_id=image_id,
            coco_image_id=checked.image.coco_image_id,
            image_path=Path(image_rel),
            annotation_path=Path(annotation_rel),
            width=checked.image.width,
            height=checked.image.height,
            source_lot=extra.get("source_lot"),
            source_session=extra.get("source_session"),
            near_duplicate_group_id=extra.get("near_duplicate_group_id"),
        ))
        parsed_images.append(checked.image)
        parsed_objects.extend(checked.objects)

    parsed = ParsedAnnotations(
        path=Path(annotation_rel),
        classes=index.classes,
        images=tuple(parsed_images),
        objects=tuple(parsed_objects),
    )
    return DatasetInput(
        root=root,
        source_policy=policy,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        annotations=(parsed,),
    )
