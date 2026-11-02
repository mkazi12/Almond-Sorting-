"""A repeatable persistence demo; its ONNX placeholder is not executable."""

from pathlib import Path

from sqlalchemy import select, text

from app.storage.inspections import (DetectionResult, FrameResult, create_session, start_session,
                                     save_frame_results, complete_session)
from app.storage.model_registry import register_model_version
from app.storage.models import InspectionSession
from app.storage.operations import writing
from app.storage.recordings import create_recording


SAMPLE_NAME = '[SAMPLE] SCRUM-48/49 persistence check'


def create_sample_data(session, store):
    writing(session)
    session.execute(text('SELECT pg_advisory_xact_lock(48490002)'))
    existing = session.scalar(select(InspectionSession).where(InspectionSession.name == SAMPLE_NAME))
    if existing is not None:
        return existing
    root = Path(__file__).resolve().parents[2]
    # Import the placeholder from a temporary source, retaining its informative filename.
    import tempfile
    with session.begin_nested():
        with tempfile.TemporaryDirectory() as temporary:
            placeholder = Path(temporary) / 'SAMPLE-not-a-real-model.onnx'
            placeholder.write_bytes(b'SAMPLE ONLY - not executable ONNX\n')
            version = register_model_version(session, store, '[SAMPLE] placeholder model',
                onnx_path=placeholder, class_keys=['almond', 'rock'], architecture='sample-placeholder',
                framework='sample-only', input_size=(640, 640))
        version.notes = 'SAMPLE ONLY. Never load this placeholder into an inference runtime.'
        recording = create_recording(session, store, '[SAMPLE] supplied almond images',
                                    [root / 'media' / 'Almond sample.jpg', root / 'test_almonds.jpg'])
        row = create_session(session, recording.id, version.id, name=SAMPLE_NAME)
        start_session(session, row.id)
        first, second = [f.media_file for f in recording.files]
        save_frame_results(session, row.id, [
            FrameResult(first.id, None, 0, (DetectionResult(0, .95, (
                first.width * .1, first.height * .1, first.width * .5, first.height * .5)),)),
            FrameResult(second.id, None, 0),
        ])
        complete_session(session, row.id)
    return row
