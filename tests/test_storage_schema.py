"""SCRUM-46: the schema stores inspection results and rejects invalid data."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app.storage.models import (
    Dataset,
    DatasetAnnotation,
    DatasetImage,
    DatasetImageRejection,
    DatasetImageStatus,
    DatasetSplit,
    DatasetVersion,
    DatasetVersionStatus,
    Detection,
    EvaluationOutcome,
    EvaluationResult,
    Frame,
    InspectionSession,
    JobStatus,
    MediaFile,
    MediaKind,
    ModelEvaluation,
    ModelVersion,
    ModelVersionClass,
    TrainingRun,
    TrainingStatus,
    User,
)
from tests.storage_builders import (
    CATEGORY_KEYS,
    assert_rejected,
    category,
    inspection_session,
    media,
    model_version,
    recording,
)


def dataset_version(session, **overrides):
    values = dict(
        dataset=Dataset(name=f"harvest-{uuid.uuid4().hex[:6]}"),
        label="v1",
        require_lot=True,
        require_session=True,
    )
    values.update(overrides)
    version = DatasetVersion(**values)
    session.add(version)
    session.flush()
    return version


def accepted_image(session, version, key="img-001", **overrides):
    values = dict(
        dataset_version=version,
        status=DatasetImageStatus.ACCEPTED,
        source_path=f"images/{key}.png",
        image_key=key,
        media_file=media(session, width=640, height=480),
        width=640,
        height=480,
        split=DatasetSplit.TRAIN,
    )
    values.update(overrides)
    image = DatasetImage(**values)
    session.add(image)
    session.flush()
    return image


def box(**overrides):
    values = dict(x_min=0.1, y_min=0.2, x_max=0.3, y_max=0.4)
    values.update(overrides)
    return values


# --- Identity and reference data ---------------------------------------------


def test_server_generates_uuidv7_ids_and_utc_timestamps(db_session):
    user = User(employee_id="E1001", display_name="Jordan Davis")
    db_session.add(user)
    db_session.flush()

    assert user.id.version == 7
    assert user.created_at.tzinfo is not None
    assert abs(user.created_at - datetime.now(UTC)) < timedelta(minutes=1)


def test_categories_from_the_mockup_are_seeded(db_session):
    keys = db_session.scalars(text("SELECT key FROM categories ORDER BY sort_order")).all()
    assert tuple(keys) == CATEGORY_KEYS


# --- End-to-end inspection results -------------------------------------------


def test_saved_session_resolves_model_dataset_media_and_detections(migrated_engine, db_session):
    trained_on = dataset_version(db_session)
    version = model_version(db_session, version_number=4, training_dataset_version=trained_on)
    footage = recording(db_session, video_count=2)
    session_row = inspection_session(
        db_session, recording=footage, model_version=version, interval_end_ms=1_800_000
    )
    session_row.categories = [category(db_session, key) for key in CATEGORY_KEYS]
    first_video, second_video = (item.media_file for item in footage.files)

    frames = [
        Frame(session=session_row, media_file=first_video, frame_index=0, timestamp_ms=0),
        Frame(session=session_row, media_file=first_video, frame_index=30, timestamp_ms=1_000),
        # A frame with nothing detected is still recorded as analyzed.
        Frame(session=session_row, media_file=second_video, frame_index=0, timestamp_ms=3_600_000),
    ]
    frames[0].detections = [
        Detection(category=category(db_session, "almond"), confidence=0.98, **box()),
        Detection(category=category(db_session, "rock"), confidence=0.91, **box(x_min=0.5, x_max=0.6)),
    ]
    frames[1].detections = [Detection(category=category(db_session, "almond"), confidence=0.95, **box())]
    db_session.add_all(frames)
    db_session.commit()
    session_id = session_row.id
    db_session.expunge_all()

    saved = db_session.get(InspectionSession, session_id)
    assert saved.model_version.id == version.id
    assert saved.model_version.version_number == 4
    assert saved.model_version.training_dataset_version_id == trained_on.id
    assert saved.recording.files[1].media_file.relative_path == second_video.relative_path
    assert [item.key for item in saved.categories] == list(CATEGORY_KEYS)

    timeline = db_session.scalars(
        saved.frames.select().order_by(Frame.timestamp_ms)
    ).all()
    assert [(item.frame_index, len(item.detections)) for item in timeline] == [(0, 2), (30, 1), (0, 0)]

    counts = dict(db_session.execute(
        select(Detection.category_id, func.count())
        .join(Frame)
        .where(Frame.session_id == session_id)
        .group_by(Detection.category_id)
    ).all())
    assert counts == {category(db_session, "almond").id: 2, category(db_session, "rock").id: 1}


# --- Required fields and value ranges ----------------------------------------


@pytest.mark.parametrize("missing", ["confidence", "category_id", "x_min"])
def test_detection_required_fields(db_session, missing):
    frame = Frame(session=inspection_session(db_session), media_file=media(db_session), timestamp_ms=0)
    values = dict(frame=frame, category_id=category(db_session, "almond").id, confidence=0.9, **box())
    values.pop(missing)
    assert_rejected(db_session, Detection(**values), constraint="not-null")


@pytest.mark.parametrize(
    "bbox",
    [
        box(x_min=0.3, x_max=0.3),  # zero width
        box(x_min=0.4, x_max=0.3),  # inverted
        box(y_min=0.5, y_max=0.4),
        box(x_min=-0.01),
        box(x_max=1.01),
        box(y_max=2.0),
    ],
)
def test_detection_rejects_invalid_boxes(db_session, bbox):
    frame = Frame(session=inspection_session(db_session), media_file=media(db_session), timestamp_ms=0)
    detection = Detection(frame=frame, category=category(db_session, "rock"), confidence=0.9, **bbox)
    assert_rejected(db_session, detection, constraint="ck_detections_bbox_valid")


def test_detection_accepts_full_frame_box(db_session):
    frame = Frame(session=inspection_session(db_session), media_file=media(db_session), timestamp_ms=0)
    db_session.add(Detection(frame=frame, category=category(db_session, "rock"), confidence=1.0,
                             x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0))
    db_session.flush()


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_detection_rejects_confidence_outside_unit_interval(db_session, confidence):
    frame = Frame(session=inspection_session(db_session), media_file=media(db_session), timestamp_ms=0)
    detection = Detection(frame=frame, category=category(db_session, "rock"), confidence=confidence, **box())
    assert_rejected(db_session, detection, constraint="ck_detections_confidence_in_unit_interval")


def test_status_values_are_checked_by_the_database(db_session):
    user = User(employee_id="E2", display_name="Op")
    db_session.add(user)
    db_session.flush()
    with pytest.raises(IntegrityError, match="ck_users_user_role"):
        with db_session.begin_nested():
            db_session.execute(text("UPDATE users SET role = 'bogus' WHERE id = :id"), {"id": user.id})


def test_blank_names_are_rejected(db_session):
    assert_rejected(db_session, User(employee_id="  ", display_name="X"), constraint="employee_id_not_blank")


# --- Relationships -----------------------------------------------------------


def test_session_must_reference_an_existing_model_version(db_session):
    row = InspectionSession(recording=recording(db_session), model_version_id=uuid.uuid4(),
                            confidence_threshold=0.8)
    assert_rejected(db_session, row, constraint="fk_inspection_sessions_model_version_id")


def test_frame_must_reference_an_existing_session(db_session):
    row = Frame(session_id=uuid.uuid4(), media_file=media(db_session), timestamp_ms=0)
    assert_rejected(db_session, row, constraint="fk_frames_session_id")


def test_detection_must_reference_an_existing_frame(db_session):
    row = Detection(frame_id=987654321, category=category(db_session, "almond"), confidence=0.9, **box())
    assert_rejected(db_session, row, constraint="fk_detections_frame_id")


def test_video_frame_is_recorded_once_per_session(db_session):
    session_row = inspection_session(db_session)
    video = session_row.recording.files[0].media_file
    db_session.add(Frame(session=session_row, media_file=video, frame_index=30, timestamp_ms=1000))
    db_session.flush()
    duplicate = Frame(session=session_row, media_file=video, frame_index=30, timestamp_ms=1000)
    assert_rejected(db_session, duplicate, constraint="uq_frames_session_id_media_file_id_frame_index")


def test_still_image_frame_is_recorded_once_per_session(db_session):
    session_row = inspection_session(db_session, sample_fps=None)
    snapshot = media(db_session, width=1280, height=720)
    db_session.add(Frame(session=session_row, media_file=snapshot, timestamp_ms=0))
    db_session.flush()
    # NULL frame indexes count as equal (UNIQUE NULLS NOT DISTINCT).
    duplicate = Frame(session=session_row, media_file=snapshot, timestamp_ms=0)
    assert_rejected(db_session, duplicate, constraint="uq_frames_session_id_media_file_id_frame_index")


def test_same_footage_can_be_analyzed_by_several_models(db_session):
    footage = recording(db_session)
    first = inspection_session(db_session, recording=footage)
    second = inspection_session(db_session, recording=footage, model_version=model_version(db_session))
    video = footage.files[0].media_file
    db_session.add_all([
        Frame(session=first, media_file=video, frame_index=0, timestamp_ms=0),
        Frame(session=second, media_file=video, frame_index=0, timestamp_ms=0),
    ])
    db_session.flush()


def test_deleting_a_session_removes_its_frames_and_detections(db_session):
    session_row = inspection_session(db_session)
    frame = Frame(session=session_row, media_file=session_row.recording.files[0].media_file,
                  frame_index=0, timestamp_ms=0)
    frame.detections = [Detection(category=category(db_session, "almond"), confidence=0.9, **box())]
    db_session.add(frame)
    db_session.flush()

    db_session.delete(session_row)
    db_session.flush()

    assert db_session.scalar(select(func.count()).select_from(Frame)) == 0
    assert db_session.scalar(select(func.count()).select_from(Detection)) == 0


def test_model_version_used_by_a_session_cannot_be_deleted(db_session):
    session_row = inspection_session(db_session)
    with pytest.raises(IntegrityError, match="fk_inspection_sessions_model_version_id"):
        with db_session.begin_nested():
            db_session.execute(text("DELETE FROM model_versions WHERE id = :id"),
                               {"id": session_row.model_version_id})


def test_media_file_referenced_by_a_frame_cannot_be_deleted(db_session):
    snapshot = media(db_session, width=1280, height=720)
    db_session.add(Frame(session=inspection_session(db_session, sample_fps=None),
                         media_file=snapshot, timestamp_ms=0))
    db_session.flush()
    with pytest.raises(IntegrityError, match="fk_frames_media_file_id"):
        with db_session.begin_nested():
            db_session.execute(text("DELETE FROM media_files WHERE id = :id"), {"id": snapshot.id})


def test_media_file_in_a_recording_cannot_be_deleted(db_session):
    video = recording(db_session).files[0].media_file
    with pytest.raises(IntegrityError, match="fk_recording_files_media_file_id"):
        with db_session.begin_nested():
            db_session.execute(text("DELETE FROM media_files WHERE id = :id"), {"id": video.id})


@pytest.mark.parametrize("dimensions", [dict(width=640), dict(height=480), dict(width=0, height=480)])
def test_media_dimensions_are_both_set_and_positive(db_session, dimensions):
    row = MediaFile(kind=MediaKind.IMAGE, relative_path="images/x.png", sha256="c" * 64,
                    size_bytes=1, **dimensions)
    assert_rejected(db_session, row, constraint="ck_media_files_dimensions_valid")


# --- Media references --------------------------------------------------------


@pytest.mark.parametrize(
    "relative_path",
    ["/videos/a.mp4", "C:/videos/a.mp4", "videos\\a.mp4", "../a.mp4", "videos/../a.mp4",
     "videos//a.mp4", "videos/a.mp4/", "./a.mp4", ""],
)
def test_media_paths_must_be_relative_and_contained(db_session, relative_path):
    row = MediaFile(kind=MediaKind.VIDEO, relative_path=relative_path, sha256="a" * 64, size_bytes=1)
    assert_rejected(db_session, row, constraint="ck_media_files_relative_path_relative")


def test_media_path_with_dots_in_names_is_allowed(db_session):
    media(db_session, relative_path="videos/2026/11/shift..1.v2.mp4")


@pytest.mark.parametrize("sha256", ["ABC", "g" * 64, "A" * 64])
def test_media_sha256_must_be_lowercase_hex(db_session, sha256):
    row = MediaFile(kind=MediaKind.IMAGE, relative_path="images/x.png", sha256=sha256, size_bytes=1)
    assert_rejected(db_session, row, constraint="ck_media_files_sha256_format")


def test_media_relative_paths_are_unique(db_session):
    media(db_session, relative_path="images/same.png")
    row = MediaFile(kind=MediaKind.IMAGE, relative_path="images/same.png", sha256="b" * 64, size_bytes=1)
    assert_rejected(db_session, row, constraint="uq_media_files_relative_path")


# --- Models ------------------------------------------------------------------


def test_model_version_needs_a_weights_file(db_session):
    existing = model_version(db_session)
    weightless = ModelVersion(model=existing.model, version_number=2, source=existing.source,
                              architecture="rf-detr-medium", framework="rfdetr",
                              input_width=576, input_height=576)
    assert_rejected(db_session, weightless, constraint="ck_model_versions_has_weights_file")


def test_model_version_numbers_are_unique_per_model(db_session):
    version = model_version(db_session)
    duplicate = ModelVersion(model=version.model, version_number=1, source=version.source,
                             architecture="rf-detr-medium", framework="rfdetr",
                             input_width=576, input_height=576,
                             onnx_file=media(db_session, MediaKind.MODEL_ONNX))
    assert_rejected(db_session, duplicate, constraint="uq_model_versions_model_id_version_number")


def test_model_class_map_cannot_repeat_a_category(db_session):
    version = model_version(db_session)
    duplicate = ModelVersionClass(model_version_id=version.id, class_index=99,
                                  category_id=category(db_session, "almond").id)
    assert_rejected(db_session, duplicate, constraint="uq_model_version_classes_model_version_id_category_id")


def test_training_run_links_base_and_output_models(db_session):
    base = model_version(db_session)
    trained_on = dataset_version(db_session, status=DatasetVersionStatus.FROZEN,
                                 frozen_at=datetime.now(UTC))
    run = TrainingRun(dataset_version=trained_on, base_model_version=base, total_epochs=50,
                      config={"batch_size": 8, "lr": 1e-4})
    output = model_version(db_session, training_run=run, training_dataset_version=trained_on)
    db_session.commit()
    db_session.expunge_all()

    saved = db_session.get(TrainingRun, run.id)
    assert saved.status is TrainingStatus.QUEUED
    assert saved.config == {"batch_size": 8, "lr": 1e-4}
    assert saved.base_model_version_id == base.id
    assert saved.output_model_version.id == output.id

    # A run produces at most one model version.
    with pytest.raises(IntegrityError, match="uq_model_versions_training_run_id"):
        with db_session.begin_nested():
            model_version(db_session, training_run_id=run.id)


@pytest.mark.parametrize(
    "overrides, constraint",
    [
        (dict(status=TrainingStatus.COMPLETED, started_at=datetime.now(UTC)), "finished_at_matches_status"),
        (dict(status=TrainingStatus.RUNNING), "active_has_started_at"),
        (dict(status=TrainingStatus.FAILED, finished_at=datetime.now(UTC)), "failed_has_error_message"),
        (dict(current_epoch=51), "current_epoch_in_range"),
        (dict(current_epoch=5, best_epoch=6), "best_epoch_in_range"),
    ],
)
def test_training_run_state_must_be_consistent(db_session, overrides, constraint):
    run = TrainingRun(dataset_version=dataset_version(db_session), total_epochs=50, **overrides)
    assert_rejected(db_session, run, constraint=constraint)


# --- Session lifecycle -------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, constraint",
    [
        (dict(status=JobStatus.COMPLETED), "finished_at_matches_status"),
        (dict(status=JobStatus.RUNNING), "active_has_started_at"),
        (dict(status=JobStatus.FAILED, finished_at=datetime.now(UTC)), "failed_has_error_message"),
        (dict(started_at=datetime.now(UTC), status=JobStatus.COMPLETED,
              finished_at=datetime.now(UTC) - timedelta(hours=1)), "finished_after_started"),
        (dict(interval_start_ms=5000, interval_end_ms=5000), "interval_end_after_start"),
        (dict(sample_fps=0), "sample_fps_positive"),
        (dict(confidence_threshold=1.5), "confidence_threshold_in_unit_interval"),
    ],
)
def test_session_state_must_be_consistent(db_session, overrides, constraint):
    values = dict(recording=recording(db_session), model_version=model_version(db_session),
                  confidence_threshold=0.8)
    values.update(overrides)
    assert_rejected(db_session, InspectionSession(**values), constraint=constraint)


# --- Datasets ----------------------------------------------------------------


def test_accepted_image_needs_key_file_and_dimensions(db_session):
    version = dataset_version(db_session)
    row = DatasetImage(dataset_version=version, status=DatasetImageStatus.ACCEPTED,
                       source_path="images/a.png", split=DatasetSplit.TRAIN)
    assert_rejected(db_session, row, constraint="accepted_is_complete")


def test_rejected_image_keeps_reasons_and_has_no_split(db_session):
    version = dataset_version(db_session)
    rejected = DatasetImage(dataset_version=version, status=DatasetImageStatus.REJECTED,
                            source_path="images/broken.jpg")
    rejected.rejections = [
        DatasetImageRejection(category="unreadable_image", detail="Could not decode."),
        DatasetImageRejection(category="missing_annotation", detail="No COCO entry."),
    ]
    db_session.add(rejected)
    db_session.flush()

    with_split = DatasetImage(dataset_version=version, status=DatasetImageStatus.REJECTED,
                              source_path="images/other.jpg", split=DatasetSplit.VALID)
    assert_rejected(db_session, with_split, constraint="rejected_has_no_split")


def test_image_keys_are_unique_within_a_version_only(db_session):
    first = dataset_version(db_session)
    accepted_image(db_session, first, key="img-001")
    duplicate = DatasetImage(dataset_version=first, status=DatasetImageStatus.ACCEPTED,
                             source_path="images/copy.png", image_key="img-001",
                             media_file=media(db_session), width=10, height=10)
    assert_rejected(db_session, duplicate, constraint="uq_dataset_images_dataset_version_id_image_key")

    second = dataset_version(db_session, dataset=first.dataset, label="v2")
    accepted_image(db_session, second, key="img-001")


def test_annotation_boxes_are_validated(db_session):
    image = accepted_image(db_session, dataset_version(db_session))
    row = DatasetAnnotation(dataset_image=image, category=category(db_session, "almond"),
                            **box(x_min=0.9, x_max=0.1))
    assert_rejected(db_session, row, constraint="ck_dataset_annotations_bbox_valid")


def test_frozen_versions_record_when_they_were_frozen(db_session):
    row = DatasetVersion(dataset=Dataset(name="harvest"), label="v1", require_lot=True,
                         require_session=False, status=DatasetVersionStatus.FROZEN)
    assert_rejected(db_session, row, constraint="frozen_at_matches_status")


# --- Evaluations -------------------------------------------------------------


def _evaluation(session, **overrides):
    version = dataset_version(session)
    values = dict(model_version=model_version(session), dataset_version=version,
                  split=DatasetSplit.TEST, iou_threshold=0.5, confidence_threshold=0.25)
    values.update(overrides)
    evaluation = ModelEvaluation(**values)
    session.add(evaluation)
    session.flush()
    return evaluation


def test_completed_evaluation_needs_metrics(db_session):
    version = dataset_version(db_session)
    row = ModelEvaluation(model_version=model_version(db_session), dataset_version=version,
                          split=DatasetSplit.TEST, iou_threshold=0.5, confidence_threshold=0.25,
                          status=JobStatus.COMPLETED, started_at=datetime.now(UTC),
                          finished_at=datetime.now(UTC))
    assert_rejected(db_session, row, constraint="completed_has_metrics")


def test_evaluation_results_match_their_outcome(db_session):
    evaluation = _evaluation(db_session)
    image = accepted_image(db_session, evaluation.dataset_version)
    truth = DatasetAnnotation(dataset_image=image, category=category(db_session, "rock"), **box())
    db_session.add(truth)
    db_session.flush()
    rock = category(db_session, "rock").id

    def result(**values):
        return EvaluationResult(evaluation_id=evaluation.id, dataset_image_id=image.id, **values)

    predicted = dict(predicted_category_id=rock, confidence=0.9, pred_x_min=0.1, pred_y_min=0.2,
                     pred_x_max=0.3, pred_y_max=0.4)
    db_session.add_all([
        result(outcome=EvaluationOutcome.TRUE_POSITIVE, dataset_annotation_id=truth.id, iou=0.8, **predicted),
        result(outcome=EvaluationOutcome.FALSE_POSITIVE, **predicted),
        result(outcome=EvaluationOutcome.FALSE_NEGATIVE, dataset_annotation_id=truth.id),
    ])
    db_session.flush()

    assert_rejected(db_session, result(outcome=EvaluationOutcome.FALSE_POSITIVE,
                                       dataset_annotation_id=truth.id, **predicted),
                    constraint="outcome_fields_consistent")
    assert_rejected(db_session, result(outcome=EvaluationOutcome.FALSE_NEGATIVE,
                                       dataset_annotation_id=truth.id, **predicted),
                    constraint="outcome_fields_consistent")
    partial_box = dict(predicted, pred_y_max=None)
    assert_rejected(db_session, result(outcome=EvaluationOutcome.FALSE_POSITIVE, **partial_box),
                    constraint="pred_bbox_valid")
