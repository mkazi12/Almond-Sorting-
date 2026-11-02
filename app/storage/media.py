"""Save and retrieve media files beneath the configured media root.

Files are stored under ``<root>/<kind>/<YYYY>/<MM>/<random>.<ext>`` and the
database records only the forward-slash path relative to the root, so the
whole folder can move to another drive or machine (or be restored from a
backup) by changing ``DINOS_MEDIA_ROOT``.

Writes are atomic: data goes to ``<name>.partial``, is flushed to disk, then
renamed into place. A crash or full disk never leaves a truncated file under
a final name. Generated names avoid characters Windows rejects (``:``) and
case-only collisions.

``MediaStore`` touches only the filesystem. ``add_media_file`` and
``add_media_bytes`` also insert the ``media_files`` row in the caller's
session; if that insert fails, the copied file is removed again.
"""

import errno
import hashlib
import io
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from sqlalchemy.orm import Session

from app.storage.errors import MediaMissingError, MediaPathError, MediaStorageError
from app.storage.models import MediaFile, MediaKind


CHUNK_SIZE = 1024 * 1024
# Python's built-in table only. The module-level mimetypes functions also
# read the Windows registry, so the same file could get a different type on
# the client PC than on a developer Mac.
MIME_TYPES = mimetypes.MimeTypes()
# Common camera container missing from the built-in table.
MIME_TYPES.add_type("video/x-matroska", ".mkv")
PARTIAL_SUFFIX = ".partial"
KIND_DIRECTORIES = {
    MediaKind.VIDEO: "videos",
    MediaKind.IMAGE: "images",
    MediaKind.MODEL_CHECKPOINT: "models",
    MediaKind.MODEL_ONNX: "models",
    MediaKind.DATASET_MANIFEST: "manifests",
    MediaKind.TRAINING_LOG: "logs",
    MediaKind.EXPORT: "exports",
}


@dataclass(frozen=True)
class StoredFile:
    relative_path: str
    sha256: str
    size_bytes: int


class MediaStore:
    """Filesystem access confined to one media root.

    Use ``MediaStore.open`` at startup: it creates the root if needed and
    fails early with an actionable message when the folder is unusable.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    @classmethod
    def open(cls, root: str | Path) -> "MediaStore":
        store = cls(root)
        store._check_writable()
        return store

    def import_file(self, source: str | Path, kind: MediaKind) -> StoredFile:
        """Copy ``source`` beneath the root and return where it was stored."""
        source = Path(source)
        try:
            stream = source.open("rb")
        except FileNotFoundError as error:
            raise MediaStorageError(
                f"Could not find {source}. Check that the file still exists and choose it again."
            ) from error
        except IsADirectoryError as error:
            raise MediaStorageError(f"{source} is a folder. Choose a file instead.") from error
        except OSError as error:
            raise MediaStorageError(
                f"Could not read {source}: {_reason(error)}. Check that you have permission "
                f"to open it, then try again."
            ) from error
        with stream:
            return self._store_stream(stream, kind, _suffix(source.name))

    def save_bytes(self, data: bytes, kind: MediaKind, suffix: str) -> StoredFile:
        """Store generated content such as an exported image or report."""
        return self._store_stream(io.BytesIO(data), kind, _suffix(suffix))

    def path_for(self, relative_path: str) -> Path:
        """Absolute path for a stored relative path. Does not check existence.

        Raises MediaPathError for paths that are absolute, use backslashes or
        drive letters, contain ``.`` or ``..`` segments, or resolve outside
        the root through a link.
        """
        parts = relative_path.split("/")
        if (
            not relative_path
            or "\\" in relative_path
            or PurePosixPath(relative_path).is_absolute()
            or re.match(r"^[A-Za-z]:", relative_path)
            or any(part in ("", ".", "..") for part in parts)
        ):
            raise MediaPathError(
                f"Stored media path {relative_path!r} is not a safe relative path. "
                f"The database row may be corrupt; restore it from a backup."
            )
        path = self.root.joinpath(*parts)
        if not path.resolve().is_relative_to(self.root):
            raise MediaPathError(
                f"Stored media path {relative_path!r} points outside the media folder "
                f"{self.root}. Remove the link inside the media folder."
            )
        return path

    def resolve(self, relative_path: str) -> Path:
        """Absolute path of an existing stored file, or MediaMissingError."""
        path = self.path_for(relative_path)
        if not path.is_file():
            raise MediaMissingError(
                f"Media file {relative_path} is missing from {self.root}. Check that "
                f"DINOS_MEDIA_ROOT points at the right folder, or restore the file from a backup."
            )
        return path

    def remove(self, relative_path: str) -> None:
        """Delete a stored file if it exists. Used to undo a failed save."""
        self.path_for(relative_path).unlink(missing_ok=True)

    def _check_writable(self) -> None:
        if self.root.exists() and not self.root.is_dir():
            raise MediaStorageError(
                f"Media root {self.root} is a file, not a folder. Set DINOS_MEDIA_ROOT to a folder."
            )
        probe = self.root / f".write-test-{uuid.uuid4().hex}"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            probe.write_bytes(b"")
            probe.unlink()
        except OSError as error:
            raise MediaStorageError(
                f"Cannot create or write to media root {self.root}: {_reason(error)}. Grant your user "
                f"write permission to that folder, or set DINOS_MEDIA_ROOT to a folder you can "
                f"write to."
            ) from error

    def _store_stream(self, stream: BinaryIO, kind: MediaKind, suffix: str) -> StoredFile:
        now = datetime.now(UTC)
        relative_path = f"{KIND_DIRECTORIES[kind]}/{now:%Y}/{now:%m}/{uuid.uuid4().hex}{suffix}"
        final = self.path_for(relative_path)
        partial = final.with_name(final.name + PARTIAL_SUFFIX)
        digest = hashlib.sha256()
        size = 0
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            with partial.open("xb") as handle:
                while chunk := stream.read(CHUNK_SIZE):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, final)
        except OSError as error:
            partial.unlink(missing_ok=True)
            if error.errno == errno.ENOSPC:
                hint = "The disk is full. Free up space or move DINOS_MEDIA_ROOT to a larger drive."
            else:
                hint = "Check free disk space and write permission for the media folder."
            raise MediaStorageError(
                f"Could not save the file to {self.root}: {_reason(error)}. {hint}"
            ) from error
        return StoredFile(relative_path=relative_path, sha256=digest.hexdigest(), size_bytes=size)


def add_media_file(
    session: Session,
    store: MediaStore,
    source: str | Path,
    kind: MediaKind,
    **details: Any,
) -> MediaFile:
    """Copy ``source`` into the store and add its ``media_files`` row.

    ``details`` sets optional MediaFile columns (``width``, ``height``,
    ``duration_ms``, ``fps``, ``frame_count``, ``codec``, ``content_type``,
    ``created_by``). The row is flushed so its id is available; committing
    stays with the caller.
    """
    source = Path(source)
    stored = store.import_file(source, kind)
    details.setdefault("original_filename", source.name)
    details.setdefault("content_type", MIME_TYPES.guess_type(source.name)[0])
    return _register(session, store, stored, kind, details)


def add_media_bytes(
    session: Session,
    store: MediaStore,
    data: bytes,
    kind: MediaKind,
    suffix: str,
    **details: Any,
) -> MediaFile:
    """Store generated bytes and add their ``media_files`` row."""
    stored = store.save_bytes(data, kind, suffix)
    details.setdefault("content_type", MIME_TYPES.guess_type(f"file{_suffix(suffix)}")[0])
    return _register(session, store, stored, kind, details)


def media_path(store: MediaStore, media_file: MediaFile) -> Path:
    """Absolute path of a recorded media file, or MediaMissingError."""
    return store.resolve(media_file.relative_path)


def _register(
    session: Session, store: MediaStore, stored: StoredFile, kind: MediaKind, details: dict[str, Any]
) -> MediaFile:
    media_file = MediaFile(
        kind=kind,
        relative_path=stored.relative_path,
        sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        **details,
    )
    try:
        with session.begin_nested():
            session.add(media_file)
    except BaseException:
        store.remove(stored.relative_path)
        raise
    return media_file


def _suffix(name: str) -> str:
    """Lowercase extension of ``name`` when it is short and alphanumeric, else ''."""
    suffix = name if name.startswith(".") and name.count(".") == 1 else Path(name).suffix
    return suffix.lower() if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", suffix) else ""


def _reason(error: OSError) -> str:
    return error.strerror or str(error)
