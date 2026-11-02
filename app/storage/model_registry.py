"""Versioned model registration, with serialized per-family version numbers."""

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import selectinload

from app.storage.errors import NotFoundError, ValidationError
from app.storage.media import add_media_file
from app.storage.models import (Category, DatasetVersion, MediaKind, Model, ModelVersion,
                                ModelVersionClass, ModelVersionStatus, ModelSource)
from app.storage.operations import integer, nonblank, require, writing


def register_model_version(session, store, model_name, *, onnx_path=None, checkpoint_path=None,
                           class_keys, architecture, framework, input_size,
                           training_dataset_version_id=None):
    name = nonblank(model_name, "Model name")
    architecture = nonblank(architecture, "Architecture")
    framework = nonblank(framework, "Framework")
    keys = list(class_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValidationError("Supply a non-empty class map with no repeated category keys.")
    if len(input_size) != 2:
        raise ValidationError("Input size must be (width, height).")
    width, height = (integer(v, "Input dimension", 1) for v in input_size)
    if onnx_path is None and checkpoint_path is None:
        raise ValidationError("Select an ONNX file or a model checkpoint.")
    categories = {c.key: c for c in session.scalars(select(Category).where(Category.key.in_(keys)))}
    if set(keys) != categories.keys():
        raise ValidationError(f"Unknown category keys: {sorted(set(keys) - categories.keys())}. Use registered categories.")
    if training_dataset_version_id is not None:
        require(session, DatasetVersion, training_dataset_version_id)
    writing(session)
    with session.begin_nested():
        session.execute(insert(Model).values(name=name).on_conflict_do_nothing(index_elements=[Model.name]))
        model = session.scalar(select(Model).where(Model.name == name).with_for_update())
        number = session.scalar(select(func.coalesce(func.max(ModelVersion.version_number), 0)).where(
            ModelVersion.model_id == model.id)) + 1
        onnx = add_media_file(session, store, onnx_path, MediaKind.MODEL_ONNX) if onnx_path else None
        checkpoint = (add_media_file(session, store, checkpoint_path, MediaKind.MODEL_CHECKPOINT)
                      if checkpoint_path else None)
        row = ModelVersion(model=model, version_number=number, source=ModelSource.IMPORTED,
                           status=ModelVersionStatus.READY if onnx else ModelVersionStatus.DRAFT,
                           architecture=architecture, framework=framework,
                           input_width=width, input_height=height, onnx_file=onnx,
                           checkpoint_file=checkpoint,
                           training_dataset_version_id=training_dataset_version_id,
                           classes=[ModelVersionClass(class_index=i, category_id=categories[key].id)
                                    for i, key in enumerate(keys)])
        session.add(row)
        session.flush()
    return row


def _loaded():
    return (selectinload(ModelVersion.model), selectinload(ModelVersion.onnx_file),
            selectinload(ModelVersion.checkpoint_file),
            selectinload(ModelVersion.classes).selectinload(ModelVersionClass.category))


def get_model_version(session, id):
    row = session.scalar(select(ModelVersion).where(ModelVersion.id == id).options(*_loaded()))
    if row is None:
        raise NotFoundError(f"Model version {id} was not found. Refresh the model list.")
    return row


def list_model_versions(session, status=ModelVersionStatus.READY):
    query = select(ModelVersion).options(*_loaded()).order_by(ModelVersion.created_at.desc(), ModelVersion.id.desc())
    if status is not None:
        query = query.where(ModelVersion.status == status)
    return list(session.scalars(query))
