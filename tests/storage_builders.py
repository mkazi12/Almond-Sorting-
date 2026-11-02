"""Builders for valid storage rows, so each test states only what it varies."""

import secrets
import warnings

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.storage.models import (
    Category,
    InspectionSession,
    MediaFile,
    MediaKind,
    Model,
    ModelSource,
    ModelVersion,
    ModelVersionClass,
    Recording,
    RecordingFile,
)


CATEGORY_KEYS = ("almond", "defect", "rock", "stick", "shell", "hull")


def assert_rejected(session: Session, *rows: object, constraint: str | None = None) -> IntegrityError:
    """Assert the database refuses ``rows``; the rest of the test continues normally."""
    with pytest.raises(IntegrityError) as caught, warnings.catch_warnings():
        # Rows built with relationship arguments are attached to their parents
        # before they join the session; SQLAlchemy warns when begin_nested()
        # flushes those parents first. The rows are added right after.
        warnings.filterwarnings("ignore", message=r"Object of type .* not in session")
        with session.begin_nested():
            session.add_all(rows)
    if constraint is not None:
        assert constraint in str(caught.value.orig), str(caught.value.orig)
    return caught.value


def category(session: Session, key: str) -> Category:
    # Seeded reference data; skip autoflush so half-built rows stay unflushed.
    with session.no_autoflush:
        return session.scalars(select(Category).where(Category.key == key)).one()


def media(session: Session, kind: MediaKind = MediaKind.IMAGE, **overrides) -> MediaFile:
    folder = {MediaKind.VIDEO: "videos", MediaKind.MODEL_ONNX: "models"}.get(kind, "images")
    values = dict(
        kind=kind,
        relative_path=f"{folder}/2026/11/{secrets.token_hex(8)}",
        sha256=secrets.token_hex(32),
        size_bytes=1024,
    )
    values.update(overrides)
    row = MediaFile(**values)
    session.add(row)
    session.flush()
    return row


def model_version(session: Session, **overrides) -> ModelVersion:
    """A ready ONNX model version whose class indices map to every category."""
    model = Model(name=f"almond-sort-{secrets.token_hex(4)}")
    values = dict(
        model=model,
        version_number=1,
        source=ModelSource.TRAINED,
        architecture="rf-detr-medium",
        framework="rfdetr",
        input_width=576,
        input_height=576,
        onnx_file=media(session, MediaKind.MODEL_ONNX),
    )
    values.update(overrides)
    version = ModelVersion(**values)
    version.classes = [
        ModelVersionClass(class_index=index, category_id=category(session, key).id)
        for index, key in enumerate(CATEGORY_KEYS)
    ]
    session.add(version)
    session.flush()
    return version


def recording(session: Session, video_count: int = 1) -> Recording:
    """A recording split across ``video_count`` one-hour 30 fps video files."""
    row = Recording(name="Morning inspection", line_name="Line 02", camera_name="Camera 01")
    row.files = [
        RecordingFile(
            position=position,
            start_offset_ms=position * 3_600_000,
            media_file=media(
                session, MediaKind.VIDEO, width=1280, height=720, fps=30.0,
                frame_count=108_000, duration_ms=3_600_000,
            ),
        )
        for position in range(video_count)
    ]
    session.add(row)
    session.flush()
    return row


def inspection_session(session: Session, **overrides) -> InspectionSession:
    values = dict(
        recording=recording(session),
        model_version=model_version(session),
        sample_fps=1.0,
        confidence_threshold=0.8,
    )
    values.update(overrides)
    row = InspectionSession(**values)
    session.add(row)
    session.flush()
    return row
