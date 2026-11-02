"""SCRUM-49: committed restart and real dump/restore acceptance tests."""

from dataclasses import replace
from pathlib import Path
import json

import pytest
from sqlalchemy import func, select, text

from app.storage.backup import backup, restore, _manifest
from app.storage.config import Settings, load_settings
from app.storage.database import create_db_engine, session_factory
from app.storage.errors import BackupError, ValidationError
from app.storage.inspections import (get_session_summary, frames_between, create_session, start_session,
    save_frame_results, complete_session, FrameResult, DetectionResult)
from app.storage.media import MediaStore, media_path
from app.storage.models import Base, Frame, Detection
from app.storage.sample_data import create_sample_data
from app.storage.schema import upgrade_to_head
from app.storage.verification import verify_media, sha256_file


@pytest.fixture
def committed_store(create_database, tmp_path):
    database = create_database()
    upgrade_to_head(database)
    settings = Settings(database, tmp_path / 'source-media', load_settings().pg_bin)
    store = MediaStore.open(settings.media_root)
    engine = create_db_engine(database)
    try:
        with session_factory(engine).begin() as session:
            id = create_sample_data(session, store).id
        yield settings, store, engine, id
    finally:
        engine.dispose()


def counts(session):
    return {name: session.scalar(select(func.count()).select_from(table))
            for name, table in Base.metadata.tables.items()}


def assert_links(session, store, id):
    summary = get_session_summary(session, id)
    onnx = summary['model_version'].onnx_file
    assert media_path(store, onnx).is_file()
    assert sha256_file(media_path(store, onnx)) == onnx.sha256
    frames = frames_between(session, id, 0, 1)
    assert len(frames) == 2 and sum(len(f.detections) for f in frames) == 1
    for frame in frames:
        assert media_path(store, frame.media_file).is_file()
        assert sha256_file(media_path(store, frame.media_file)) == frame.media_file.sha256
    assert not verify_media(session, store, full=True)
    assert summary['created_at'] <= summary['started_at'] <= summary['finished_at']
    for key in ('created_at', 'started_at', 'finished_at'):
        assert summary[key].utcoffset().total_seconds() == 0


def test_restart_uses_new_engine_and_committed_rows(committed_store):
    settings, store, engine, id = committed_store
    with session_factory(engine)() as session:
        summary = get_session_summary(session, id)
        expected = (summary['model_version'].id, summary['recording'].id,
                    summary['model_version'].onnx_file.relative_path,
                    summary['created_at'], summary['started_at'], summary['finished_at'])
        expected_frames = [(f.id, f.media_file_id, f.media_file.relative_path, f.timestamp_ms)
                           for f in frames_between(session, id, 0, 1)]
    engine.dispose()
    reopened = create_db_engine(settings.database)
    try:
        with session_factory(reopened).begin() as session:
            before = counts(session)
            assert create_sample_data(session, store).id == id
            assert counts(session) == before
            assert_links(session, store, id)
            summary = get_session_summary(session, id)
            assert (summary['model_version'].id, summary['recording'].id,
                    summary['model_version'].onnx_file.relative_path,
                    summary['created_at'], summary['started_at'], summary['finished_at']) == expected
            assert [(f.id, f.media_file_id, f.media_file.relative_path, f.timestamp_ms)
                    for f in frames_between(session, id, 0, 1)] == expected_frames
    finally:
        reopened.dispose()


def test_verification_reports_owner_and_same_size_corruption(committed_store):
    _, store, engine, id = committed_store
    with session_factory(engine)() as session:
        file = get_session_summary(session, id)['model_version'].onnx_file
        path = media_path(store, file)
        original = path.read_bytes()
        path.write_bytes(b'x' * len(original))
        assert verify_media(session, store) == []
        issue, = verify_media(session, store, full=True)
        assert issue.reason == 'SHA-256 mismatch' and any('model version' in owner for owner in issue.owners)
        path.unlink()
        assert len(verify_media(session, store)) == 1
        path.write_bytes(original)


def test_backup_restore_new_database_and_root(committed_store, create_database, tmp_path):
    settings, store, engine, id = committed_store
    destination = tmp_path / 'backup'
    manifest = backup(settings, destination)
    assert manifest['file_count'] == 3
    with session_factory(engine)() as session:
        original_counts = counts(session)
    target = replace(settings, database=create_database(), media_root=tmp_path / 'restored-media')
    restore(target, destination)
    restored = create_db_engine(target.database)
    try:
        with session_factory(restored).begin() as session:
            assert counts(session) == original_counts
            assert_links(session, MediaStore(target.media_root), id)
            summary = get_session_summary(session, id)
            another = create_session(session, summary['recording'].id, summary['model_version'].id)
            start_session(session, another.id)
            file = summary['recording'].files[0].media_file
            assert save_frame_results(session, another.id, [FrameResult(file.id, None, 0,
                (DetectionResult(0, .9, (0, 0, 10, 10)),))]) == 1
            complete_session(session, another.id)
            assert session.scalar(select(func.max(Frame.id))) > original_counts['frames']
        with pytest.raises(BackupError, match='not empty'):
            restore(replace(target, media_root=tmp_path / 'another-empty-root'), destination)
    finally:
        restored.dispose()


def test_backup_refuses_active_job_and_existing_destination(committed_store, tmp_path):
    settings, _, engine, id = committed_store
    with session_factory(engine).begin() as session:
        summary = get_session_summary(session, id)
        job = create_session(session, summary['recording'].id, summary['model_version'].id)
        start_session(session, job.id)
    with pytest.raises(BackupError, match='active'):
        backup(settings, tmp_path / 'active-backup')
    with pytest.raises(BackupError, match='new backup'):
        backup(settings, tmp_path)


def test_corrupt_backup_is_rejected_before_target_mutation(committed_store, create_database, tmp_path):
    settings, _, _, _ = committed_store
    directory = tmp_path / 'backup'
    backup(settings, directory)
    (directory / 'database.dump').write_bytes(b'corrupted')
    target = replace(settings, database=create_database(), media_root=tmp_path / 'new-media')
    with pytest.raises(BackupError, match='SHA-256'):
        restore(target, directory)
    assert not target.media_root.exists()


def test_optional_pg_bin_setting_resolves_against_env_file(tmp_path):
    env = tmp_path / 'settings.env'
    env.write_text('DINOS_DB_HOST=localhost\nDINOS_DB_NAME=test\nDINOS_DB_USER=test\n'
                   'DINOS_DB_PASSWORD=unused\nDINOS_MEDIA_ROOT=media\nDINOS_PG_BIN=tools/bin\n')
    assert load_settings(environ={}, env_file=env).pg_bin == (tmp_path / 'tools/bin').resolve()


def test_missing_tools_have_an_actionable_error(committed_store, tmp_path):
    settings, _, _, _ = committed_store
    with pytest.raises(BackupError, match='DINOS_PG_BIN'):
        backup(replace(settings, pg_bin=tmp_path / 'missing-tools'), tmp_path / 'backup')


def test_failed_restore_rolls_back_database_and_cleans_copied_media(committed_store, create_database, tmp_path, monkeypatch):
    from app.storage import backup as module
    settings, _, _, _ = committed_store
    directory = tmp_path / 'backup'
    backup(settings, directory)
    target = replace(settings, database=create_database(), media_root=tmp_path / 'failed-media')
    original_run = module._run
    def fail(tool, arguments, database):
        if '--exit-on-error' in arguments:
            raise BackupError('Simulated pg_restore failure')
        return original_run(tool, arguments, database)
    monkeypatch.setattr(module, '_run', fail)
    with pytest.raises(BackupError, match='Simulated'):
        restore(target, directory)
    assert not [p for p in target.media_root.rglob('*') if p.is_file()]
    # A retry into the same still-empty targets succeeds after the failure.
    monkeypatch.setattr(module, '_run', original_run)
    restore(target, directory)


def test_backup_restore_reopens_decodable_video(committed_store, create_database, tmp_path):
    import cv2
    import numpy as np
    from app.storage.recordings import create_recording
    settings, _, engine, id = committed_store
    video = tmp_path / 'video.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (40, 30))
    assert writer.isOpened()
    writer.write(np.zeros((30, 40, 3), dtype=np.uint8))
    writer.release()
    with session_factory(engine).begin() as session:
        recording = create_recording(session, MediaStore(settings.media_root), 'Video restore', [video])
        model = get_session_summary(session, id)['model_version']
        job = create_session(session, recording.id, model.id)
        start_session(session, job.id)
        save_frame_results(session, job.id, [FrameResult(recording.files[0].media_file_id, 0, 0)])
        complete_session(session, job.id)
        video_job_id = job.id
    directory = tmp_path / 'backup-with-video'
    backup(settings, directory)
    target = replace(settings, database=create_database(), media_root=tmp_path / 'restored-video-media')
    restore(target, directory)
    restored = create_db_engine(target.database)
    try:
        with session_factory(restored)() as session:
            frame, = frames_between(session, video_job_id, 0, 1)
            capture = cv2.VideoCapture(str(media_path(MediaStore(target.media_root), frame.media_file)))
            try:
                assert capture.read()[0]
            finally:
                capture.release()
    finally:
        restored.dispose()


def test_backup_maintenance_lock_rejects_uncommitted_writes(committed_store, tmp_path):
    from app.storage.operations import writing
    settings, _, engine, _ = committed_store
    with session_factory(engine).begin() as session:
        writing(session)
        with pytest.raises(BackupError, match='writes are active'):
            backup(settings, tmp_path / 'blocked-backup')
    assert not (tmp_path / 'blocked-backup').exists()


def test_manifest_rejects_unsafe_media_paths_before_restore(committed_store, create_database, tmp_path):
    settings, _, _, _ = committed_store
    directory = tmp_path / 'backup'
    backup(settings, directory)
    manifest_path = directory / 'backup.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][0]['path'] = '../escape.onnx'
    manifest_path.write_text(json.dumps(manifest))
    target = replace(settings, database=create_database(), media_root=tmp_path / 'untouched-media')
    with pytest.raises(BackupError, match='corrupt'):
        restore(target, directory)
    assert not target.media_root.exists()


def test_cli_commands_round_trip_in_separate_processes(committed_store, create_database, tmp_path):
    import os
    import subprocess
    import sys
    settings, _, _, id = committed_store
    def run(settings, *arguments):
        env = dict(os.environ, DINOS_ENV_FILE=str(tmp_path / 'absent.env'),
            DINOS_DB_HOST=settings.database.host, DINOS_DB_PORT=str(settings.database.port),
            DINOS_DB_NAME=settings.database.name, DINOS_DB_USER=settings.database.user,
            DINOS_DB_PASSWORD=settings.database.password, DINOS_MEDIA_ROOT=str(settings.media_root))
        if settings.pg_bin:
            env['DINOS_PG_BIN'] = str(settings.pg_bin)
        return subprocess.run([sys.executable, '-m', 'app.storage', *arguments], env=env,
                              text=True, capture_output=True, timeout=60)
    for command in (('migrate',), ('check',), ('sample-data',), ('verify-media', '--full')):
        result = run(settings, *command)
        assert result.returncode == 0, result.stderr
        if command[0] == 'sample-data':
            assert str(id) in result.stdout and 'placeholder' in result.stdout
    directory = tmp_path / 'cli-backup'
    result = run(settings, 'backup', str(directory))
    assert result.returncode == 0, result.stderr
    target = replace(settings, database=create_database(), media_root=tmp_path / 'cli-restored')
    result = run(target, 'restore', str(directory))
    assert result.returncode == 0, result.stderr
    result = run(target, 'sample-data')
    assert result.returncode == 0 and str(id) in result.stdout, result.stderr
    result = run(replace(target, media_root=tmp_path / 'empty-again'), 'restore', str(directory))
    assert result.returncode == 1 and 'not empty' in result.stderr and 'Traceback' not in result.stderr
