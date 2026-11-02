"""SCRUM-48 service acceptance: transactions, identities, results and lifecycle."""

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import func, select

from app.dataset.manifest import build_manifest
from app.dataset.records import (AcceptedImage, ClassSpec, CocoImage, DatasetInput,
    GroundTruthObject, ParsedAnnotations, RejectedImage, RejectionReason, SourcePolicy)
from app.storage.datasets import import_dataset_version, freeze_dataset_version
from app.storage.errors import ValidationError, NotFoundError
from app.storage.inspections import (FrameResult, DetectionResult, create_session, start_session,
    complete_session, fail_session, cancel_session, save_frame_results, frames_between,
    get_session_summary, list_sessions)
from app.storage.media import add_media_bytes, media_path
from app.storage.model_registry import register_model_version, get_model_version, list_model_versions
from app.storage.models import (DatasetImage, DatasetAnnotation, DatasetImageRejection,
    DatasetVersionStatus, Detection, Frame, MediaKind, ModelVersionStatus, JobStatus, TrainingStatus)
from app.storage.operations import normalized_box
from app.storage.recordings import create_recording, get_recording, probe_media
from app.storage.training import (create_training_run, start_training_run, record_epoch,
    request_stop, finish_training_run)


@pytest.fixture
def source_image(tmp_path):
    path = tmp_path / 'frame.jpg'
    assert cv2.imwrite(str(path), np.zeros((120, 200, 3), dtype=np.uint8))
    return path


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / 'test.onnx'
    path.write_bytes(b'placeholder, never executed')
    return path


def register(session, store, weights, name='test-model', **overrides):
    values = dict(onnx_path=weights, class_keys=['almond', 'rock'], architecture='test',
                  framework='test', input_size=(200, 120))
    values.update(overrides)
    return register_model_version(session, store, name, **values)


def inspection(session, store, source, weights, **kwargs):
    recording = create_recording(session, store, 'Test footage', [source])
    version = register(session, store, weights)
    row = create_session(session, recording.id, version.id, **kwargs)
    return row, recording, version


def count(session, model):
    return session.scalar(select(func.count()).select_from(model))


@pytest.mark.parametrize('box', [(20., 12., 180., 100.), (0., 0., 200., 120.)])
def test_pixel_boxes_round_trip(box):
    normalized = normalized_box(box, 200, 120)
    restored = [v * scale for v, scale in zip(normalized.values(), (200, 120, 200, 120))]
    assert restored == pytest.approx(box, abs=1e-9)


def test_edge_tolerance_is_normalized_and_large_excursions_fail():
    assert normalized_box((-0.0001, 0, 200.0001, 120), 200, 120)['x_min'] == 0
    for box in [(-1, 0, 200, 120), (0, 0, float('nan'), 120), (20, 0, 10, 120)]:
        with pytest.raises(ValidationError):
            normalized_box(box, 200, 120)


def test_recording_and_model_reopen_with_eager_media(db_session, media_store, source_image, weights):
    capture_time = datetime(2026, 1, 1, tzinfo=UTC)
    recording = create_recording(db_session, media_store, 'A shift', [source_image], started_at=capture_time)
    version = register(db_session, media_store, weights)
    ids = recording.id, version.id
    db_session.expunge_all()
    saved = get_recording(db_session, ids[0])
    model = get_model_version(db_session, ids[1])
    db_session.expunge_all()
    assert saved.started_at == capture_time
    assert saved.id.version == model.id.version == 7
    assert saved.created_at.utcoffset().total_seconds() == 0
    assert media_path(media_store, saved.files[0].media_file).read_bytes() == source_image.read_bytes()
    assert media_path(media_store, model.onnx_file).read_bytes() == weights.read_bytes()
    assert [c.category.key for c in model.classes] == ['almond', 'rock']


def test_registry_versions_filter_validation(db_session, media_store, weights):
    first = register(db_session, media_store, weights)
    second = register(db_session, media_store, weights)
    draft = register(db_session, media_store, weights, name='checkpoint-only', onnx_path=None, checkpoint_path=weights)
    assert [first.version_number, second.version_number] == [1, 2]
    assert draft.status == ModelVersionStatus.DRAFT
    assert {v.id for v in list_model_versions(db_session)} == {first.id, second.id}
    for keys in (['unknown'], ['almond', 'almond'], []):
        with pytest.raises(ValidationError):
            register(db_session, media_store, weights, class_keys=keys)


def test_recording_prevalidates_every_file(db_session, media_store, source_image, tmp_path):
    with pytest.raises(ValidationError):
        create_recording(db_session, media_store, 'Broken', [source_image, tmp_path / 'missing.jpg'])
    assert not list(media_store.root.rglob('*.jpg'))


def test_media_cleanup_tracks_nested_and_outer_rollback(db_session, media_store):
    kept = add_media_bytes(db_session, media_store, b'keep', MediaKind.IMAGE, '.jpg')
    with pytest.raises(RuntimeError):
        with db_session.begin_nested():
            nested = add_media_bytes(db_session, media_store, b'undo', MediaKind.IMAGE, '.jpg')
            nested_path = media_path(media_store, nested)
            raise RuntimeError('rollback savepoint')
    assert not nested_path.exists()
    assert media_path(media_store, kept).is_file()
    kept_path = media_path(media_store, kept)
    db_session.rollback()
    assert not kept_path.exists()


def test_failed_multi_file_registration_cleans_copies(db_session, media_store, weights, tmp_path):
    from app.storage.errors import MediaStorageError
    with pytest.raises(MediaStorageError):
        register(db_session, media_store, weights, checkpoint_path=tmp_path / 'missing.pth')
    assert not [p for p in media_store.root.rglob('*') if p.is_file()]


def test_inspection_idempotency_empty_frames_counts_and_replay(db_session, media_store, source_image, weights):
    row, recording, version = inspection(db_session, media_store, source_image, weights)
    start_session(db_session, row.id)
    file = recording.files[0].media_file
    result = FrameResult(file.id, None, 0, (DetectionResult(0, .9, (20, 12, 180, 100)),))
    assert save_frame_results(db_session, row.id, [result, result]) == 1
    assert save_frame_results(db_session, row.id, [result]) == 0
    complete_session(db_session, row.id)
    summary = get_session_summary(db_session, row.id)
    assert summary['frame_count'] == 1
    assert summary['category_counts'] == {'almond': 1, 'rock': 0}
    assert count(db_session, Detection) == 1
    assert row.created_at <= row.started_at <= row.finished_at
    assert frames_between(db_session, row.id, 1, 2) == []
    replay = frames_between(db_session, row.id, 0, 1)
    assert len(replay) == 1 and len(replay[0].detections) == 1
    assert replay[0].detections[0].x_min == .1
    assert list_sessions(db_session, search='footage', model_version_id=version.id)[0].id == row.id
    assert list_sessions(db_session, search='absent') == []
    with pytest.raises(ValidationError):
        start_session(db_session, row.id)


def test_selected_category_and_threshold_policy(db_session, media_store, source_image, weights):
    row, recording, _ = inspection(db_session, media_store, source_image, weights,
                                   category_keys=['almond'], confidence_threshold=.8)
    start_session(db_session, row.id)
    result = FrameResult(recording.files[0].media_file_id, None, 0, (
        DetectionResult(0, .7, (0, 0, 20, 20)), DetectionResult(1, .9, (0, 0, 20, 20))))
    save_frame_results(db_session, row.id, [result])
    assert count(db_session, Frame) == 1 and count(db_session, Detection) == 0


def test_categories_archived_models_and_lifecycle(db_session, media_store, source_image, weights):
    row, recording, version = inspection(db_session, media_store, source_image, weights)
    with pytest.raises(ValidationError):
        create_session(db_session, recording.id, version.id, category_keys=['shell'])
    version.status = ModelVersionStatus.ARCHIVED
    db_session.flush()
    with pytest.raises(ValidationError):
        start_session(db_session, row.id)
    with pytest.raises(ValidationError):
        create_session(db_session, recording.id, version.id)
    cancel_session(db_session, row.id)
    assert row.status == JobStatus.CANCELLED and row.finished_at
    version.status = ModelVersionStatus.READY
    failed = create_session(db_session, recording.id, version.id)
    fail_session(db_session, failed.id, 'Test failure; retry with readable media')
    assert failed.status == JobStatus.FAILED and failed.error_message


def test_result_call_is_atomic_when_later_chunk_is_invalid(db_session, media_store, tmp_path, weights):
    video = make_video(tmp_path / 'clip.avi', frames=501)
    row, recording, _ = inspection(db_session, media_store, video, weights)
    start_session(db_session, row.id)
    file_id = recording.files[0].media_file_id
    results = [FrameResult(file_id, i, i * 100) for i in range(500)]
    results.append(FrameResult(file_id, 500, 50000, (DetectionResult(99, .8, (0, 0, 20, 20)),)))
    with pytest.raises(ValidationError, match='class index'):
        save_frame_results(db_session, row.id, results)
    assert count(db_session, Frame) == 0


def make_video(path, frames=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (40, 30))
    assert writer.isOpened()
    try:
        for _ in range(frames):
            writer.write(np.zeros((30, 40, 3), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_video_offsets_metadata_and_duplicate_mixed_batch(db_session, media_store, tmp_path, weights):
    video = make_video(tmp_path / 'clip.avi')
    row = create_recording(db_session, media_store, 'Chunks', [video, video])
    assert [file.start_offset_ms for file in row.files] == [0, 1000]
    version = register(db_session, media_store, weights)
    job = create_session(db_session, row.id, version.id)
    start_session(db_session, job.id)
    first, second = [file.media_file_id for file in row.files]
    assert save_frame_results(db_session, job.id, [(first, 0, 0, [])]) == 1
    assert save_frame_results(db_session, job.id, [(first, 0, 0, []), (second, 0, 1000, [])]) == 1
    assert get_session_summary(db_session, job.id)['frame_count'] == 2
    with pytest.raises(ValidationError, match='Timestamp'):
        save_frame_results(db_session, job.id, [(second, 1, 100, [])])


def dataset_records(source_image):
    policy = SourcePolicy(False, False)
    accepted = AcceptedImage('stable-image', 1, Path(source_image.name), Path('labels.json'),
        200, 120, 'lot-1', 'camera-1', 'group-1')
    annotations = ParsedAnnotations(Path('labels.json'), (ClassSpec(7, 'almond'),),
        (CocoImage(1, source_image.name, 200, 120),),
        (GroundTruthObject(42, 1, 7, (20, 12, 160, 88)),))
    return DatasetInput(source_image.parent, policy, (accepted,),
        (RejectedImage(Path('bad.jpg'), None, (RejectionReason('unreadable', 'Cannot decode'),)),),
        (annotations,))


def test_dataset_import_and_freeze_keep_provenance(db_session, media_store, source_image):
    records = dataset_records(source_image)
    manifest = build_manifest(records, 'v1')
    row = import_dataset_version(db_session, media_store, records, manifest, dataset_name='Test dataset',
                                 category_map={('labels.json', 7): 'almond'})
    images = list(db_session.scalars(select(DatasetImage).where(DatasetImage.dataset_version_id == row.id)))
    assert len(images) == 2 and all(image.split is None for image in images)
    accepted = next(image for image in images if image.image_key)
    assert accepted.image_key == 'stable-image' and accepted.source_lot == 'lot-1'
    assert accepted.source_coco_image_id == 1 and accepted.source_annotation_path == 'labels.json'
    assert count(db_session, DatasetImageRejection) == 1
    box = db_session.scalar(select(DatasetAnnotation))
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == pytest.approx((.1, .1, .9, 100 / 120))
    assert box.source_annotation_id == 42
    frozen = freeze_dataset_version(db_session, row.id)
    assert frozen.status == DatasetVersionStatus.FROZEN and frozen.frozen_at
    assert freeze_dataset_version(db_session, row.id).frozen_at == frozen.frozen_at
    with pytest.raises(ValidationError, match='label'):
        import_dataset_version(db_session, media_store, records, manifest, dataset_name='Test dataset',
                               category_map={('labels.json', 7): 'almond'})


def test_training_lifecycle_epochs_stop_and_fail(db_session, media_store, source_image):
    records = dataset_records(source_image)
    version = import_dataset_version(db_session, media_store, records, build_manifest(records, 'v1'),
        dataset_name='Training data', category_map={('labels.json', 7): 'almond'})
    with pytest.raises(ValidationError, match='Freeze'):
        create_training_run(db_session, version.id, total_epochs=2)
    freeze_dataset_version(db_session, version.id)
    run = create_training_run(db_session, version.id, total_epochs=2)
    start_training_run(db_session, run.id)
    record_epoch(db_session, run.id, 1, train_loss=.9, val_map50=.7, is_best=True)
    with pytest.raises(ValidationError, match='once'):
        record_epoch(db_session, run.id, 1)
    request_stop(db_session, run.id)
    assert run.status == TrainingStatus.STOPPING
    finish_training_run(db_session, run.id, TrainingStatus.STOPPED)
    assert run.current_epoch == run.best_epoch == 1 and run.finished_at >= run.started_at
    queued = create_training_run(db_session, version.id, total_epochs=1)
    request_stop(db_session, queued.id)
    assert queued.status == TrainingStatus.STOPPED and queued.started_at is None
    completed = create_training_run(db_session, version.id, total_epochs=1)
    start_training_run(db_session, completed.id)
    with pytest.raises(ValidationError, match='all configured'):
        finish_training_run(db_session, completed.id, TrainingStatus.COMPLETED)
    record_epoch(db_session, completed.id, 1)
    finish_training_run(db_session, completed.id, TrainingStatus.COMPLETED)
    failed = create_training_run(db_session, version.id, total_epochs=1)
    finish_training_run(db_session, failed.id, TrainingStatus.FAILED, 'GPU unavailable; retry on CPU')
    assert failed.error_message


def test_model_number_allocation_is_safe_across_connections(create_database, tmp_path, weights):
    from concurrent.futures import ThreadPoolExecutor
    from app.storage.database import create_db_engine, session_factory
    from app.storage.media import MediaStore
    from app.storage.schema import upgrade_to_head
    settings = create_database()
    upgrade_to_head(settings)
    engine = create_db_engine(settings)
    store = MediaStore.open(tmp_path / 'concurrent-media')
    try:
        def save(_):
            with session_factory(engine).begin() as session:
                return register(session, store, weights, name='concurrent-family').version_number
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert sorted(pool.map(save, range(2))) == [1, 2]
    finally:
        engine.dispose()


def test_media_survives_commit_but_new_copy_is_removed_on_session_close(create_database, tmp_path):
    from app.storage.database import create_db_engine, session_factory
    from app.storage.media import MediaStore
    from app.storage.schema import upgrade_to_head
    settings = create_database()
    upgrade_to_head(settings)
    engine = create_db_engine(settings)
    store = MediaStore.open(tmp_path / 'committed-media')
    try:
        with session_factory(engine).begin() as session:
            saved = add_media_bytes(session, store, b'committed', MediaKind.IMAGE, '.jpg')
            path = media_path(store, saved)
        assert path.is_file()
        with session_factory(engine)() as session:
            abandoned = add_media_bytes(session, store, b'not committed', MediaKind.IMAGE, '.jpg')
            abandoned_path = media_path(store, abandoned)
        assert not abandoned_path.exists()
        assert path.read_bytes() == b'committed'
    finally:
        engine.dispose()


def test_detection_insert_failure_rolls_back_its_frames(db_session, media_store, source_image, weights):
    from sqlalchemy import event
    row, recording, _ = inspection(db_session, media_store, source_image, weights)
    start_session(db_session, row.id)
    result = FrameResult(recording.files[0].media_file_id, None, 0,
                         (DetectionResult(0, .9, (0, 0, 20, 20)),))
    def fail_insert(conn, cursor, statement, parameters, context, executemany):
        if statement.startswith('INSERT INTO detections'):
            raise RuntimeError('Simulated interruption between frame and detection inserts')
    event.listen(db_session.bind, 'before_cursor_execute', fail_insert)
    try:
        with pytest.raises(RuntimeError, match='interruption'):
            save_frame_results(db_session, row.id, [result])
    finally:
        event.remove(db_session.bind, 'before_cursor_execute', fail_insert)
    assert count(db_session, Frame) == count(db_session, Detection) == 0
    assert save_frame_results(db_session, row.id, [result]) == 1


def test_committed_ids_paths_and_utc_times_survive_new_engine(create_database, tmp_path, source_image, weights):
    from app.storage.database import create_db_engine, session_factory
    from app.storage.media import MediaStore
    from app.storage.schema import upgrade_to_head
    settings = create_database()
    upgrade_to_head(settings)
    engine = create_db_engine(settings)
    store = MediaStore.open(tmp_path / 'reopen-media')
    try:
        with session_factory(engine).begin() as session:
            row, recording, version = inspection(session, store, source_image, weights)
            start_session(session, row.id)
            save_frame_results(session, row.id, [FrameResult(recording.files[0].media_file_id, None, 0)])
            complete_session(session, row.id)
            expected = (row.id, recording.id, version.id, version.onnx_file.relative_path,
                        recording.files[0].media_file.relative_path,
                        row.created_at, row.started_at, row.finished_at)
    finally:
        engine.dispose()
    reopened = create_db_engine(settings)
    try:
        with session_factory(reopened)() as session:
            result = get_session_summary(session, expected[0])
            recording, version = result['recording'], result['model_version']
            actual = (result['id'], recording.id, version.id, version.onnx_file.relative_path,
                      recording.files[0].media_file.relative_path,
                      result['created_at'], result['started_at'], result['finished_at'])
            assert actual == expected
            assert actual[-3] <= actual[-2] <= actual[-1]
            assert all(time.utcoffset().total_seconds() == 0 for time in actual[-3:])
            assert media_path(store, version.onnx_file).read_bytes() == weights.read_bytes()
            assert media_path(store, recording.files[0].media_file).read_bytes() == source_image.read_bytes()
    finally:
        reopened.dispose()
