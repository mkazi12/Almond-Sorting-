"""Inspection lifecycle, query/replay operations, and idempotent bulk results.

All functions use caller-owned transactions. Pass bounded result batches and
commit each call to limit crash loss. Replay intervals are [start_ms, end_ms).
"""

from dataclasses import dataclass
from itertools import islice

from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import selectinload

from app.storage.errors import ValidationError
from app.storage.model_registry import get_model_version
from app.storage.models import (Category, Detection, Frame, InspectionSession, JobStatus,
                                MediaKind, ModelVersion, ModelVersionStatus, Recording)
from app.storage.operations import finite, integer, nonblank, normalized_box, require, writing
from app.storage.recordings import get_recording


@dataclass(frozen=True)
class DetectionResult:
    class_index: int
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class FrameResult:
    media_file_id: object
    frame_index: int | None
    timestamp_ms: int
    detections: tuple[DetectionResult, ...] = ()


def create_session(session, recording_id, model_version_id, *, name=None,
                   category_keys=None, interval_start_ms=0, interval_end_ms=None,
                   sample_fps=1., confidence_threshold=.5):
    recording = get_recording(session, recording_id)
    version = get_model_version(session, model_version_id)
    _ready(version)
    integer(interval_start_ms, "Interval start")
    if interval_end_ms is not None:
        integer(interval_end_ms, "Interval end", interval_start_ms + 1)
    finite(confidence_threshold, "Confidence threshold", 0, 1)
    videos = any(f.media_file.kind == MediaKind.VIDEO for f in recording.files)
    if videos:
        if sample_fps is None or finite(sample_fps, "Sampling FPS", 0) <= 0:
            raise ValidationError("Video sampling FPS must be positive.")
        duration = max(f.start_offset_ms + f.media_file.duration_ms for f in recording.files)
        if interval_start_ms >= duration or (interval_end_ms is not None and interval_end_ms > duration):
            raise ValidationError("Inspection interval exceeds recording duration.")
    else:
        sample_fps = None
    available = {item.category.key: item.category for item in version.classes}
    keys = list(available) if category_keys is None else list(category_keys)
    if not keys or len(set(keys)) != len(keys) or not set(keys) <= available.keys():
        raise ValidationError("Select at least one unique category from this model's class map.")
    writing(session)
    row = InspectionSession(recording=recording, model_version=version, name=name,
                            categories=[available[key] for key in keys],
                            interval_start_ms=interval_start_ms, interval_end_ms=interval_end_ms,
                            sample_fps=sample_fps, confidence_threshold=confidence_threshold)
    session.add(row)
    session.flush()
    return row


def _ready(version):
    if version.status != ModelVersionStatus.READY or version.onnx_media_file_id is None:
        raise ValidationError("This model is not ready for inference. Select a ready ONNX model version.")


def _locked(session, id):
    writing(session)
    require(session, InspectionSession, id)
    return session.scalar(select(InspectionSession).where(InspectionSession.id == id)
                          .with_for_update().execution_options(populate_existing=True))


def start_session(session, session_id):
    row = _locked(session, session_id)
    if row.status != JobStatus.QUEUED:
        raise ValidationError("Only a queued inspection can start. Create a new inspection to rerun it.")
    _ready(get_model_version(session, row.model_version_id))
    row.status, row.started_at = JobStatus.RUNNING, func.now()
    session.flush()
    return row


def _finish(session, id, status, error=None):
    row = _locked(session, id)
    allowed = {JobStatus.RUNNING} if status == JobStatus.COMPLETED else {JobStatus.QUEUED, JobStatus.RUNNING}
    if row.status not in allowed:
        raise ValidationError(f"Cannot mark {row.status} inspection {status}. Refresh its state.")
    row.status, row.finished_at, row.error_message = status, func.now(), error
    session.flush()
    return row


def complete_session(session, session_id):
    return _finish(session, session_id, JobStatus.COMPLETED)


def fail_session(session, session_id, error):
    return _finish(session, session_id, JobStatus.FAILED, nonblank(error, "Failure reason"))


def cancel_session(session, session_id):
    return _finish(session, session_id, JobStatus.CANCELLED)


def save_frame_results(session, session_id, results):
    """Save new frames, returning their count. Duplicate frames keep original results.

    Results may be FrameResult records or four-tuples. Detections must be
    DetectionResult records. Categories/threshold are applied before insertion.
    One savepoint makes the whole call atomic, including multiple chunks.
    """
    row = _locked(session, session_id)
    if row.status != JobStatus.RUNNING:
        raise ValidationError("Start the inspection before saving results.")
    recording = get_recording(session, row.recording_id)
    media = {f.media_file_id: f for f in recording.files}
    classes = {c.class_index: c.category_id for c in get_model_version(session, row.model_version_id).classes}
    selected = {category.id for category in row.categories}
    iterator = iter(results)
    saved = 0
    with session.begin_nested():
        while batch := list(islice(iterator, 500)):
            frames, boxes = [], {}
            for item in batch:
                item = item if isinstance(item, FrameResult) else FrameResult(*item)
                file = media.get(item.media_file_id)
                if file is None:
                    raise ValidationError("Frame media does not belong to the session recording.")
                source = file.media_file
                integer(item.timestamp_ms, "Frame timestamp", row.interval_start_ms)
                if row.interval_end_ms is not None and item.timestamp_ms >= row.interval_end_ms:
                    raise ValidationError("Frame timestamp lies outside the selected interval.")
                if source.kind == MediaKind.IMAGE:
                    if item.frame_index is not None or item.timestamp_ms != file.start_offset_ms:
                        raise ValidationError("Still images need frame_index=None and their recording offset timestamp.")
                else:
                    integer(item.frame_index, "Frame index")
                    if item.frame_index >= source.frame_count:
                        raise ValidationError("Frame index exceeds the video frame count.")
                    expected = file.start_offset_ms + item.frame_index * 1000 / source.fps
                    if abs(item.timestamp_ms - expected) > 1:
                        raise ValidationError("Timestamp must match the video frame and recording offset (within 1 ms).")
                key = (item.media_file_id, item.frame_index)
                if key in boxes:
                    continue
                detections = []
                for det in item.detections:
                    integer(det.class_index, "Class index")
                    if det.class_index not in classes:
                        raise ValidationError("Model class index is not in the registered class map.")
                    finite(det.confidence, "Confidence", 0, 1)
                    bbox = normalized_box(det.box, source.width, source.height)
                    category = classes[det.class_index]
                    if category in selected and det.confidence >= row.confidence_threshold:
                        detections.append(dict(category_id=category, confidence=det.confidence, **bbox))
                boxes[key] = detections
                frames.append(dict(session_id=row.id, media_file_id=item.media_file_id,
                                   frame_index=item.frame_index, timestamp_ms=item.timestamp_ms))
            inserted = session.execute(insert(Frame).values(frames).on_conflict_do_nothing(
                index_elements=[Frame.session_id, Frame.media_file_id, Frame.frame_index])
                .returning(Frame.id, Frame.media_file_id, Frame.frame_index)).all()
            saved += len(inserted)
            detections = [dict(frame_id=id, **box) for id, file_id, index in inserted
                          for box in boxes[(file_id, index)]]
            for offset in range(0, len(detections), 1000):
                # RETURNING enables SQLAlchemy's cached insertmanyvalues path;
                # psycopg executemany otherwise sends one INSERT per detection.
                session.connection().execute(insert(Detection.__table__).returning(Detection.id),
                    detections[offset:offset + 1000]).close()
    return saved


def get_session_summary(session, session_id):
    row = require(session, InspectionSession, session_id)
    version = get_model_version(session, row.model_version_id)
    counts = dict(session.execute(select(Detection.category_id, func.count()).join(Frame)
                                 .where(Frame.session_id == row.id).group_by(Detection.category_id)).all())
    return dict(id=row.id, status=row.status, model_label=f"{version.model.name} v{version.version_number}",
                model_version=version, recording=get_recording(session, row.recording_id),
                interval_start_ms=row.interval_start_ms, interval_end_ms=row.interval_end_ms,
                frame_count=session.scalar(select(func.count()).select_from(Frame).where(Frame.session_id == row.id)),
                category_counts={c.key: counts.get(c.id, 0) for c in row.categories},
                created_at=row.created_at, started_at=row.started_at, finished_at=row.finished_at)


def list_sessions(session, search=None, model_version_id=None, since=None, limit=50, offset=0):
    integer(limit, "Limit", 1)
    integer(offset, "Offset")
    if limit > 1000:
        raise ValidationError("Limit must be at most 1000. Page through the results.")
    query = select(InspectionSession).options(selectinload(InspectionSession.recording),
        selectinload(InspectionSession.model_version).selectinload(ModelVersion.model))
    if search:
        pattern = f"%{search.replace('!', '!!').replace('%', '!%').replace('_', '!_')}%"
        query = query.where(or_(InspectionSession.name.ilike(pattern, escape="!"),
            InspectionSession.recording.has(Recording.name.ilike(pattern, escape="!"))))
    if model_version_id is not None:
        query = query.where(InspectionSession.model_version_id == model_version_id)
    if since is not None:
        if since.utcoffset() is None:
            raise ValidationError("Since must include a timezone.")
        query = query.where(InspectionSession.created_at >= since)
    return list(session.scalars(query.order_by(InspectionSession.created_at.desc(), InspectionSession.id.desc())
                               .limit(limit).offset(offset)))


def frames_between(session, session_id, start_ms, end_ms):
    require(session, InspectionSession, session_id)
    integer(start_ms, "Replay start")
    integer(end_ms, "Replay end", start_ms + 1)
    return list(session.scalars(select(Frame).where(Frame.session_id == session_id,
        Frame.timestamp_ms >= start_ms, Frame.timestamp_ms < end_ms).options(
            selectinload(Frame.media_file), selectinload(Frame.detections).selectinload(Detection.category))
        .order_by(Frame.timestamp_ms, Frame.media_file_id, Frame.frame_index, Frame.id)))
