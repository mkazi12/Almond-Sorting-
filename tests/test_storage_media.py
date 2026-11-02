"""SCRUM-47: media is saved beneath the configured root with relative paths in the database."""

import errno
import hashlib
import os

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.storage import media as media_module
from app.storage.errors import MediaMissingError, MediaPathError, MediaStorageError
from app.storage.media import MediaStore, add_media_bytes, add_media_file, media_path
from app.storage.models import MediaFile, MediaKind


VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + bytes(range(256)) * 64


@pytest.fixture
def source_video(tmp_path):
    path = tmp_path / "uploads" / "Morning Shift Line 02.MP4"
    path.parent.mkdir()
    path.write_bytes(VIDEO_BYTES)
    return path


def stored_files(root):
    return sorted(path for path in root.rglob("*") if path.is_file())


def test_import_copies_beneath_the_root_with_a_relative_path(media_store, source_video):
    stored = media_store.import_file(source_video, MediaKind.VIDEO)

    assert stored.relative_path.startswith("videos/")
    assert stored.relative_path.endswith(".mp4")
    assert "\\" not in stored.relative_path and ":" not in stored.relative_path
    assert stored.sha256 == hashlib.sha256(VIDEO_BYTES).hexdigest()
    assert stored.size_bytes == len(VIDEO_BYTES)

    path = media_store.resolve(stored.relative_path)
    assert path.is_relative_to(media_store.root)
    assert path.read_bytes() == VIDEO_BYTES
    assert source_video.read_bytes() == VIDEO_BYTES
    assert stored_files(media_store.root) == [path]


def test_each_save_gets_its_own_file(media_store, source_video):
    first = media_store.import_file(source_video, MediaKind.VIDEO)
    second = media_store.import_file(source_video, MediaKind.VIDEO)
    assert first.relative_path != second.relative_path


def test_generated_bytes_are_saved_by_kind(media_store):
    stored = media_store.save_bytes(b"onnx-weights", MediaKind.MODEL_ONNX, ".ONNX")
    assert stored.relative_path.startswith("models/")
    assert stored.relative_path.endswith(".onnx")
    assert media_store.resolve(stored.relative_path).read_bytes() == b"onnx-weights"


@pytest.mark.parametrize("name, expected", [("a.JPG", ".jpg"), ("archive.tar.gz", ".gz"),
                                            ("no_extension", ""), ("odd.ext with space", "")])
def test_file_extensions_are_normalized(media_store, tmp_path, name, expected):
    source = tmp_path / name
    source.write_bytes(b"x")
    stored = media_store.import_file(source, MediaKind.IMAGE)
    filename = stored.relative_path.rsplit("/", 1)[1]
    # Stored names are a 32-character random hex stem plus the normalized extension.
    assert filename[32:] == expected


@pytest.mark.parametrize(
    "relative_path",
    ["/etc/passwd", "C:/Windows/x", "videos\\a.mp4", "../outside.mp4", "videos/../../x", "videos//a", "./a", ""],
)
def test_unsafe_relative_paths_are_rejected(media_store, relative_path):
    with pytest.raises(MediaPathError):
        media_store.resolve(relative_path)


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs extra privileges on Windows")
def test_links_that_escape_the_root_are_rejected(media_store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.png").write_bytes(b"x")
    (media_store.root / "images").symlink_to(outside, target_is_directory=True)

    with pytest.raises(MediaPathError, match="outside the media folder"):
        media_store.resolve("images/secret.png")


def test_missing_files_explain_how_to_recover(media_store):
    with pytest.raises(MediaMissingError) as caught:
        media_store.resolve("videos/2026/11/gone.mp4")
    assert "DINOS_MEDIA_ROOT" in str(caught.value)
    assert "backup" in str(caught.value)


def test_missing_source_asks_to_choose_again(media_store, tmp_path):
    with pytest.raises(MediaStorageError, match="choose it again"):
        media_store.import_file(tmp_path / "nope.mp4", MediaKind.VIDEO)


def test_open_creates_a_missing_root(tmp_path):
    store = MediaStore.open(tmp_path / "new" / "media")
    assert store.root.is_dir()


def test_root_that_is_a_file_is_rejected(tmp_path):
    file_root = tmp_path / "media"
    file_root.write_text("not a folder", encoding="utf-8")
    with pytest.raises(MediaStorageError, match="DINOS_MEDIA_ROOT"):
        MediaStore.open(file_root)


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="POSIX permission bits do not restrict Windows or root")
def test_unwritable_root_is_reported_with_next_steps(tmp_path):
    root = tmp_path / "locked"
    root.mkdir()
    root.chmod(0o555)
    try:
        with pytest.raises(MediaStorageError) as caught:
            MediaStore.open(root)
    finally:
        root.chmod(0o755)
    message = str(caught.value)
    assert "Cannot create or write to media root" in message
    assert "DINOS_MEDIA_ROOT" in message


def test_failed_write_leaves_no_partial_file(media_store, source_video, monkeypatch):
    def disk_full(_fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(media_module.os, "fsync", disk_full)

    with pytest.raises(MediaStorageError, match="disk is full"):
        media_store.import_file(source_video, MediaKind.VIDEO)
    assert stored_files(media_store.root) == []


# --- Recorded in the database ------------------------------------------------


def test_saved_media_is_recorded_with_its_relative_path(db_session, media_store, source_video):
    row = add_media_file(db_session, media_store, source_video, MediaKind.VIDEO,
                         width=1280, height=720, fps=30.0, frame_count=108_000)
    db_session.commit()
    media_id = row.id
    db_session.expunge_all()

    saved = db_session.get(MediaFile, media_id)
    assert saved.relative_path == stored_files(media_store.root)[0].relative_to(media_store.root).as_posix()
    assert saved.original_filename == source_video.name
    assert saved.content_type == "video/mp4"
    assert saved.sha256 == hashlib.sha256(VIDEO_BYTES).hexdigest()
    assert media_path(media_store, saved).read_bytes() == VIDEO_BYTES


def test_media_root_can_move_without_changing_the_database(db_session, media_store, source_video, tmp_path):
    row = add_media_file(db_session, media_store, source_video, MediaKind.VIDEO)
    moved_root = tmp_path / "restored-media"
    media_store.root.rename(moved_root)

    assert media_path(MediaStore.open(moved_root), row).read_bytes() == VIDEO_BYTES


def test_generated_bytes_are_recorded(db_session, media_store):
    row = add_media_bytes(db_session, media_store, b"\x89PNG...", MediaKind.EXPORT, ".png",
                          width=64, height=48)
    assert row.content_type == "image/png"
    assert media_path(media_store, row).read_bytes() == b"\x89PNG..."


def test_file_is_removed_when_the_database_rejects_the_row(db_session, media_store, source_video):
    with pytest.raises(IntegrityError, match="ck_media_files_dimensions_valid"):
        # Invalid dimensions violate a CHECK constraint after the copy succeeded.
        add_media_file(db_session, media_store, source_video, MediaKind.VIDEO, width=-1, height=720)

    assert stored_files(media_store.root) == []
    assert db_session.scalars(select(MediaFile)).all() == []
