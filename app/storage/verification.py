"""Check stored files and report the records that depend on them."""

import hashlib
from dataclasses import dataclass

from sqlalchemy import select

from app.storage.errors import MediaStorageError
from app.storage.models import (MediaFile, ModelVersion, RecordingFile, InspectionSession,
    DatasetImage, DatasetVersion, TrainingRun, Frame)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MediaIssue:
    media_file_id: object
    relative_path: str
    reason: str
    owners: tuple[str, ...]


def media_owners(session, media_id):
    owners = []
    for version in session.scalars(select(ModelVersion).where(
            (ModelVersion.onnx_media_file_id == media_id) | (ModelVersion.checkpoint_media_file_id == media_id))):
        owners.append(f'model version {version.id}')
    recordings = list(session.scalars(select(RecordingFile.recording_id).where(RecordingFile.media_file_id == media_id)))
    owners.extend(f'recording {id}' for id in recordings)
    if recordings:
        owners.extend(f'session {id}' for id in session.scalars(select(InspectionSession.id).where(
            InspectionSession.recording_id.in_(recordings))))
    owners.extend(f'session {id}' for id in session.scalars(select(Frame.session_id)
        .where(Frame.media_file_id == media_id).distinct()) if f'session {id}' not in owners)
    owners.extend(f'dataset version {id}' for id in session.scalars(select(DatasetImage.dataset_version_id)
        .where(DatasetImage.media_file_id == media_id).distinct()))
    owners.extend(f'dataset manifest {id}' for id in session.scalars(select(DatasetVersion.id)
        .where(DatasetVersion.manifest_media_file_id == media_id)))
    owners.extend(f'training log {id}' for id in session.scalars(select(TrainingRun.id)
        .where(TrainingRun.log_media_file_id == media_id)))
    return tuple(owners)


def verify_media(session, store, *, full=False):
    issues = []
    for media in session.scalars(select(MediaFile).order_by(MediaFile.relative_path).execution_options(yield_per=500)):
        reason = None
        try:
            path = store.resolve(media.relative_path)
            if path.stat().st_size != media.size_bytes:
                reason = 'size mismatch'
            elif full:
                if sha256_file(path) != media.sha256:
                    reason = 'SHA-256 mismatch'
            else:
                with path.open('rb') as handle:
                    handle.read(1)
        except (OSError, MediaStorageError) as error:
            reason = str(error)
        if reason:
            issues.append(MediaIssue(media.id, media.relative_path, reason, media_owners(session, media.id)))
    return issues
