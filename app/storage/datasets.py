"""Persist existing dataset records without guessing categories or splits."""

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from app.dataset.manifest import build_manifest, manifest_json
from app.dataset.records import objects_for_image, relative_path_identity
from app.storage.errors import ValidationError
from app.storage.media import add_media_bytes, add_media_file
from app.storage.models import (Category, Dataset, DatasetVersion, DatasetVersionStatus,
    DatasetImage, DatasetImageStatus, DatasetAnnotation, DatasetImageRejection, MediaKind)
from app.storage.operations import nonblank, normalized_box, require, writing
from app.storage.recordings import probe_media


def import_dataset_version(session, store, dataset_input, manifest, *, dataset_name, category_map):
    """category_map keys are (annotation relative path, local COCO category ID).

    Values are registered category keys. Split assignments remain unset; source
    provenance and split eligibility are not assignments to train/valid/test.
    """
    name = nonblank(dataset_name, "Dataset name")
    if build_manifest(dataset_input, manifest.dataset_version) != manifest:
        raise ValidationError("Manifest does not match dataset records. Rebuild the manifest before importing.")
    categories = {c.key: c.id for c in session.scalars(select(Category))}
    images = []
    paths = set()
    for image in dataset_input.accepted:
        path = relative_path_identity(image.image_path)
        if path in paths:
            raise ValidationError(f"Duplicate source image {path}. Remove the duplicate record.")
        paths.add(path)
        source = dataset_input.resolve(image.image_path)
        if not source.is_relative_to(dataset_input.root):
            raise ValidationError("Dataset image link escapes the source root. Remove the external link.")
        kind, details = probe_media(source)
        if kind != MediaKind.IMAGE or (details['width'], details['height']) != (image.width, image.height):
            raise ValidationError(f"Image dimensions changed for {path}. Rebuild dataset records.")
        boxes = []
        for obj in objects_for_image(dataset_input, image):
            key = (relative_path_identity(image.annotation_path), obj.category_id)
            category = categories.get(category_map.get(key))
            if category is None:
                raise ValidationError(f"Map COCO category {key} to a registered category key.")
            x, y, width, height = obj.bbox_xywh
            boxes.append(dict(category_id=category, source_annotation_id=obj.annotation_id,
                **normalized_box((x, y, x + width, y + height), image.width, image.height)))
        images.append((image, source, details, boxes))
    for rejected in dataset_input.rejected:
        path = relative_path_identity(rejected.image_path)
        if path in paths:
            raise ValidationError(f"Duplicate source image {path}. Accepted and rejected paths must be distinct.")
        paths.add(path)
    writing(session)
    with session.begin_nested():
        session.execute(insert(Dataset).values(name=name).on_conflict_do_nothing(index_elements=[Dataset.name]))
        dataset = session.scalar(select(Dataset).where(Dataset.name == name).with_for_update())
        if session.scalar(select(DatasetVersion.id).where(DatasetVersion.dataset_id == dataset.id,
                          DatasetVersion.label == manifest.dataset_version)) is not None:
            raise ValidationError("This dataset version label already exists. Choose a new version label.")
        manifest_file = add_media_bytes(session, store, manifest_json(manifest).encode('utf-8'),
                                        MediaKind.DATASET_MANIFEST, '.json')
        version = DatasetVersion(dataset=dataset, label=manifest.dataset_version,
            content_hash=manifest.content_hash, manifest_schema_version=manifest.schema_version,
            manifest_file=manifest_file, require_lot=manifest.source_policy.require_lot,
            require_session=manifest.source_policy.require_session, source_root=str(dataset_input.root))
        session.add(version)
        session.flush()
        for image, source, details, boxes in images:
            media = add_media_file(session, store, source, MediaKind.IMAGE, **details)
            session.add(DatasetImage(dataset_version_id=version.id, status=DatasetImageStatus.ACCEPTED,
                source_path=relative_path_identity(image.image_path), image_key=image.image_id,
                media_file_id=media.id, width=image.width, height=image.height,
                source_lot=image.source_lot, source_session=image.source_session,
                near_duplicate_group_id=image.near_duplicate_group_id,
                source_coco_image_id=image.coco_image_id,
                source_annotation_path=relative_path_identity(image.annotation_path),
                annotations=[DatasetAnnotation(**box) for box in boxes]))
        for image in dataset_input.rejected:
            session.add(DatasetImage(dataset_version_id=version.id, status=DatasetImageStatus.REJECTED,
                source_path=relative_path_identity(image.image_path), image_key=image.image_id,
                rejections=[DatasetImageRejection(category=r.category, detail=r.detail) for r in image.reasons]))
        session.flush()
    return version


def freeze_dataset_version(session, version_id):
    writing(session)
    require(session, DatasetVersion, version_id)
    row = session.scalar(select(DatasetVersion).where(DatasetVersion.id == version_id)
                         .with_for_update().execution_options(populate_existing=True))
    if row.status == DatasetVersionStatus.DRAFT:
        row.status, row.frozen_at = DatasetVersionStatus.FROZEN, func.now()
        session.flush()
    return row
