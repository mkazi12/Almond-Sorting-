"""Import footage and reopen its ordered media; the caller owns the transaction."""

from pathlib import Path
import math

import cv2
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.storage.errors import ValidationError, NotFoundError
from app.storage.media import add_media_file
from app.storage.models import MediaKind, Recording, RecordingFile
from app.storage.operations import nonblank, writing


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def probe_media(source):
    path = Path(source)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValidationError(f"Cannot decode image {path}. Choose a readable image.")
        height, width = image.shape[:2]
        return MediaKind.IMAGE, dict(width=width, height=height)
    capture = cv2.VideoCapture(str(path))
    try:
        values = [capture.get(prop) for prop in (
            cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
            cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT)]
        if not capture.isOpened() or any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValidationError(f"Cannot probe video {path}. Choose a video with valid duration and FPS.")
        width, height, fps, count = values
        return MediaKind.VIDEO, dict(width=int(width), height=int(height), fps=fps,
                                    frame_count=int(count), duration_ms=round(count * 1000 / fps))
    finally:
        capture.release()


def create_recording(session, store, name, files, *, line_name=None, camera_name=None, started_at=None):
    name = nonblank(name, "Recording name")
    files = list(files)
    if not files:
        raise ValidationError("Select at least one recording file.")
    if started_at is not None and started_at.utcoffset() is None:
        raise ValidationError("Recording capture time must include a timezone.")
    probes = [(path, *probe_media(path)) for path in files]
    if len({kind for _, kind, _ in probes}) > 1:
        raise ValidationError("Import images and videos as separate recordings.")
    writing(session)
    with session.begin_nested():
        row = Recording(name=name, line_name=line_name, camera_name=camera_name, started_at=started_at)
        offset = 0
        row.files = []
        for position, (path, kind, details) in enumerate(probes):
            media = add_media_file(session, store, path, kind, **details)
            row.files.append(RecordingFile(position=position, media_file=media, start_offset_ms=offset))
            offset += details.get("duration_ms", 0)
        session.add(row)
        session.flush()
    return row


def get_recording(session, id):
    row = session.scalar(select(Recording).where(Recording.id == id).options(
        selectinload(Recording.files).selectinload(RecordingFile.media_file)))
    if row is None:
        raise NotFoundError(f"Recording {id} was not found. Refresh the recording list.")
    return row
