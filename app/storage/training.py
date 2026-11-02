"""Persist training lifecycle and metrics; no ML framework or Qt dependency."""

from sqlalchemy import func, select

from app.storage.errors import ValidationError
from app.storage.models import (DatasetVersion, DatasetVersionStatus, ModelVersion,
                                TrainingRun, TrainingStatus, TrainingEpochMetric)
from app.storage.operations import finite, integer, nonblank, require, writing


def create_training_run(session, dataset_version_id, *, total_epochs, config=None, base_model_version_id=None):
    dataset = require(session, DatasetVersion, dataset_version_id)
    if dataset.status != DatasetVersionStatus.FROZEN:
        raise ValidationError("Freeze the dataset version before starting training.")
    integer(total_epochs, "Total epochs", 1)
    if base_model_version_id is not None:
        require(session, ModelVersion, base_model_version_id)
    writing(session)
    row = TrainingRun(dataset_version_id=dataset_version_id, total_epochs=total_epochs,
                      config=config or {}, base_model_version_id=base_model_version_id)
    session.add(row)
    session.flush()
    return row


def _locked(session, id):
    writing(session)
    require(session, TrainingRun, id)
    return session.scalar(select(TrainingRun).where(TrainingRun.id == id)
                          .with_for_update().execution_options(populate_existing=True))


def start_training_run(session, run_id):
    row = _locked(session, run_id)
    if row.status != TrainingStatus.QUEUED:
        raise ValidationError("Only queued training can start. Create a new run to repeat training.")
    row.status, row.started_at = TrainingStatus.RUNNING, func.now()
    session.flush()
    return row


def record_epoch(session, run_id, epoch, *, is_best=False, **metrics):
    row = _locked(session, run_id)
    if row.status not in (TrainingStatus.RUNNING, TrainingStatus.STOPPING):
        raise ValidationError("Start training before recording epoch results.")
    integer(epoch, "Epoch", 1)
    if epoch != row.current_epoch + 1 or epoch > row.total_epochs:
        raise ValidationError("Record each epoch once, in sequence, within the configured total.")
    allowed = {'train_loss', 'val_loss', 'val_map50', 'val_map50_95', 'val_precision', 'val_recall', 'extra'}
    if not metrics.keys() <= allowed:
        raise ValidationError(f"Unknown epoch metrics: {sorted(metrics.keys() - allowed)}.")
    for key, value in metrics.items():
        if value is not None and key != 'extra':
            finite(value, key, 0 if key not in ('train_loss', 'val_loss') else None,
                   1 if key not in ('train_loss', 'val_loss') else None)
    metric = TrainingEpochMetric(training_run_id=run_id, epoch=epoch, **metrics)
    row.current_epoch = epoch
    if is_best:
        row.best_epoch = epoch
    session.add(metric)
    session.flush()
    return metric


def request_stop(session, run_id):
    row = _locked(session, run_id)
    if row.status == TrainingStatus.QUEUED:
        row.status, row.finished_at = TrainingStatus.STOPPED, func.now()
    elif row.status == TrainingStatus.RUNNING:
        row.status = TrainingStatus.STOPPING
    elif row.status != TrainingStatus.STOPPING:
        raise ValidationError("This training run has already finished. Refresh its state.")
    session.flush()
    return row


def finish_training_run(session, run_id, status, error=None):
    if status not in (TrainingStatus.COMPLETED, TrainingStatus.STOPPED, TrainingStatus.FAILED):
        raise ValidationError("Finish status must be completed, stopped, or failed.")
    if status == TrainingStatus.FAILED:
        error = nonblank(error, "Failure reason")
    row = _locked(session, run_id)
    allowed = (TrainingStatus.RUNNING, TrainingStatus.STOPPING)
    if row.status not in allowed and not (status == TrainingStatus.FAILED and row.status == TrainingStatus.QUEUED):
        raise ValidationError("This training run cannot finish from its current state.")
    if status == TrainingStatus.COMPLETED and row.current_epoch != row.total_epochs:
        raise ValidationError("Record all configured epochs before completing training.")
    row.status, row.finished_at, row.error_message = status, func.now(), error
    session.flush()
    return row
