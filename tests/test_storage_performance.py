"""SCRUM-48 full-shift save benchmark with caller-owned batch commits."""

import time
import pytest
from sqlalchemy import func, select
from app.storage.database import create_db_engine, session_factory
from app.storage.inspections import (create_session, start_session, save_frame_results,
                                    FrameResult, DetectionResult)
from app.storage.models import Frame, Detection
from app.storage.schema import upgrade_to_head


@pytest.mark.slow
def test_full_shift_bulk_save_under_a_minute(create_database, media_store, tmp_path):
    from app.storage.media import add_media_bytes
    from app.storage.models import MediaKind, Recording, RecordingFile
    from app.storage.model_registry import register_model_version
    settings = create_database()
    upgrade_to_head(settings)
    engine = create_db_engine(settings)
    factory = session_factory(engine)
    weights = tmp_path / 'placeholder.onnx'
    weights.write_bytes(b'benchmark only')
    try:
        with factory.begin() as session:
            video = add_media_bytes(session, media_store, b'benchmark metadata fixture', MediaKind.VIDEO, '.mp4',
                width=1280, height=720, fps=30., frame_count=864000, duration_ms=28800000)
            recording = Recording(name='Benchmark', files=[RecordingFile(position=0, media_file=video)])
            session.add(recording)
            version = register_model_version(session, media_store, 'Benchmark', onnx_path=weights,
                class_keys=['almond'], architecture='benchmark', framework='fixture', input_size=(640, 640))
            session.flush()
            job = create_session(session, recording.id, version.id)
            start_session(session, job.id)
            id, media_id = job.id, video.id
        detections = tuple(DetectionResult(0, .9, (i, i, i + 10, i + 10)) for i in range(20))
        start = time.perf_counter()
        for offset in range(0, 28800, 500):
            with factory.begin() as session:
                batch = [FrameResult(media_id, i * 30, i * 1000, detections)
                         for i in range(offset, min(offset + 500, 28800))]
                assert save_frame_results(session, id, batch) == len(batch)
        elapsed = time.perf_counter() - start
        with factory() as session:
            assert session.scalar(select(func.count()).select_from(Frame)) == 28800
            assert session.scalar(select(func.count()).select_from(Detection)) == 576000
        print(f'Full-shift save: {elapsed:.2f}s (28,800 frames, 576,000 detections, commit every 500 frames)')
        assert elapsed < 60
    finally:
        engine.dispose()
