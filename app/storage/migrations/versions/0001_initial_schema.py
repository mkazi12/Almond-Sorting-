"""initial schema

Creates every DINOS table (see app/storage/models.py and docs/database.md)
and seeds the detection categories shown in the mockup.

Revision ID: 0001
Revises:
Create Date: 2026-11-01 15:19:55.288255
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '0001'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('categories',
    sa.Column('id', sa.SmallInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('key', sa.Text(), nullable=False),
    sa.Column('display_name', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('kind', sa.Enum('acceptable', 'defect', 'foreign_material', name='category_kind', native_enum=False, create_constraint=True, length=32), nullable=False),
    sa.Column('sort_order', sa.SmallInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(display_name) <> ''", name=op.f('ck_categories_display_name_not_blank')),
    sa.CheckConstraint("key ~ '^[a-z][a-z0-9_]*$'", name=op.f('ck_categories_key_format')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_categories')),
    sa.UniqueConstraint('key', name=op.f('uq_categories_key'))
    )
    op.create_table('users',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('employee_id', sa.Text(), nullable=False),
    sa.Column('display_name', sa.Text(), nullable=False),
    sa.Column('role', sa.Enum('operator', 'engineer', 'admin', name='user_role', native_enum=False, create_constraint=True, length=32), server_default='operator', nullable=False),
    sa.Column('password_hash', sa.Text(), nullable=True),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('last_login_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(display_name) <> ''", name=op.f('ck_users_display_name_not_blank')),
    sa.CheckConstraint("btrim(employee_id) <> ''", name=op.f('ck_users_employee_id_not_blank')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('employee_id', name=op.f('uq_users_employee_id'))
    )
    op.create_table('datasets',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('almond_variety', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(name) <> ''", name=op.f('ck_datasets_name_not_blank')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_datasets_created_by')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_datasets')),
    sa.UniqueConstraint('name', name=op.f('uq_datasets_name'))
    )
    op.create_table('media_files',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('kind', sa.Enum('video', 'image', 'model_checkpoint', 'model_onnx', 'dataset_manifest', 'training_log', 'export', name='media_kind', native_enum=False, create_constraint=True, length=32), nullable=False),
    sa.Column('relative_path', sa.Text(), nullable=False),
    sa.Column('sha256', sa.Text(), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('content_type', sa.Text(), nullable=True),
    sa.Column('original_filename', sa.Text(), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True),
    sa.Column('height', sa.Integer(), nullable=True),
    sa.Column('duration_ms', sa.BigInteger(), nullable=True),
    sa.Column('fps', sa.Double(), nullable=True),
    sa.Column('frame_count', sa.Integer(), nullable=True),
    sa.Column('codec', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("relative_path ~ '^[^/\\\\]+(/[^/\\\\]+)*$' AND relative_path !~ '(^|/)\\.\\.?(/|$)' AND relative_path !~ '^[A-Za-z]:'", name=op.f('ck_media_files_relative_path_relative')),
    sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f('ck_media_files_sha256_format')),
    sa.CheckConstraint('(width IS NULL AND height IS NULL) OR (width IS NOT NULL AND height IS NOT NULL AND width > 0 AND height > 0)', name=op.f('ck_media_files_dimensions_valid')),
    sa.CheckConstraint('duration_ms IS NULL OR duration_ms >= 0', name=op.f('ck_media_files_duration_non_negative')),
    sa.CheckConstraint('fps IS NULL OR fps > 0', name=op.f('ck_media_files_fps_positive')),
    sa.CheckConstraint('frame_count IS NULL OR frame_count >= 0', name=op.f('ck_media_files_frame_count_non_negative')),
    sa.CheckConstraint('size_bytes >= 0', name=op.f('ck_media_files_size_bytes_non_negative')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_media_files_created_by')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_media_files')),
    sa.UniqueConstraint('relative_path', name=op.f('uq_media_files_relative_path'))
    )
    op.create_index(op.f('ix_media_files_sha256'), 'media_files', ['sha256'], unique=False)
    op.create_table('models',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(name) <> ''", name=op.f('ck_models_name_not_blank')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_models_created_by')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_models')),
    sa.UniqueConstraint('name', name=op.f('uq_models_name'))
    )
    op.create_table('recordings',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('line_name', sa.Text(), nullable=True),
    sa.Column('camera_name', sa.Text(), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(name) <> ''", name=op.f('ck_recordings_name_not_blank')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_recordings_created_by')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_recordings'))
    )
    op.create_table('dataset_versions',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('dataset_id', sa.UUID(), nullable=False),
    sa.Column('label', sa.Text(), nullable=False),
    sa.Column('status', sa.Enum('draft', 'frozen', name='dataset_version_status', native_enum=False, create_constraint=True, length=32), server_default='draft', nullable=False),
    sa.Column('frozen_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('content_hash', sa.Text(), nullable=True),
    sa.Column('manifest_schema_version', sa.Integer(), nullable=True),
    sa.Column('manifest_media_file_id', sa.UUID(), nullable=True),
    sa.Column('require_lot', sa.Boolean(), nullable=False),
    sa.Column('require_session', sa.Boolean(), nullable=False),
    sa.Column('source_root', sa.Text(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(status = 'frozen') = (frozen_at IS NOT NULL)", name=op.f('ck_dataset_versions_frozen_at_matches_status')),
    sa.CheckConstraint("btrim(label) <> ''", name=op.f('ck_dataset_versions_label_not_blank')),
    sa.CheckConstraint("content_hash IS NULL OR content_hash ~ '^[0-9a-f]{64}$'", name=op.f('ck_dataset_versions_content_hash_format')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_dataset_versions_created_by')),
    sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], name=op.f('fk_dataset_versions_dataset_id')),
    sa.ForeignKeyConstraint(['manifest_media_file_id'], ['media_files.id'], name=op.f('fk_dataset_versions_manifest_media_file_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_versions')),
    sa.UniqueConstraint('dataset_id', 'label', name=op.f('uq_dataset_versions_dataset_id_label'))
    )
    op.create_table('recording_files',
    sa.Column('recording_id', sa.UUID(), nullable=False),
    sa.Column('position', sa.Integer(), nullable=False),
    sa.Column('media_file_id', sa.UUID(), nullable=False),
    sa.Column('start_offset_ms', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.CheckConstraint('position >= 0', name=op.f('ck_recording_files_position_non_negative')),
    sa.CheckConstraint('start_offset_ms >= 0', name=op.f('ck_recording_files_start_offset_non_negative')),
    sa.ForeignKeyConstraint(['media_file_id'], ['media_files.id'], name=op.f('fk_recording_files_media_file_id')),
    sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], name=op.f('fk_recording_files_recording_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('recording_id', 'position', name=op.f('pk_recording_files')),
    sa.UniqueConstraint('media_file_id', name=op.f('uq_recording_files_media_file_id'))
    )
    op.create_table('dataset_images',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('dataset_version_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.Enum('accepted', 'rejected', name='dataset_image_status', native_enum=False, create_constraint=True, length=32), nullable=False),
    sa.Column('source_path', sa.Text(), nullable=False),
    sa.Column('image_key', sa.Text(), nullable=True),
    sa.Column('media_file_id', sa.UUID(), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True),
    sa.Column('height', sa.Integer(), nullable=True),
    sa.Column('split', sa.Enum('train', 'valid', 'test', name='dataset_split', native_enum=False, create_constraint=True, length=32), nullable=True),
    sa.Column('source_lot', sa.Text(), nullable=True),
    sa.Column('source_session', sa.Text(), nullable=True),
    sa.Column('near_duplicate_group_id', sa.Text(), nullable=True),
    sa.Column('source_coco_image_id', sa.Integer(), nullable=True),
    sa.Column('source_annotation_path', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("image_key IS NULL OR btrim(image_key) <> ''", name=op.f('ck_dataset_images_image_key_not_blank')),
    sa.CheckConstraint("source_path ~ '^[^/\\\\]+(/[^/\\\\]+)*$' AND source_path !~ '(^|/)\\.\\.?(/|$)' AND source_path !~ '^[A-Za-z]:'", name=op.f('ck_dataset_images_source_path_relative')),
    sa.CheckConstraint("status <> 'accepted' OR (image_key IS NOT NULL AND media_file_id IS NOT NULL AND width IS NOT NULL AND height IS NOT NULL)", name=op.f('ck_dataset_images_accepted_is_complete')),
    sa.CheckConstraint("status <> 'rejected' OR split IS NULL", name=op.f('ck_dataset_images_rejected_has_no_split')),
    sa.CheckConstraint('(width IS NULL AND height IS NULL) OR (width IS NOT NULL AND height IS NOT NULL AND width > 0 AND height > 0)', name=op.f('ck_dataset_images_dimensions_valid')),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_dataset_images_dataset_version_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['media_file_id'], ['media_files.id'], name=op.f('fk_dataset_images_media_file_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_images')),
    sa.UniqueConstraint('dataset_version_id', 'image_key', name=op.f('uq_dataset_images_dataset_version_id_image_key')),
    sa.UniqueConstraint('dataset_version_id', 'source_path', name=op.f('uq_dataset_images_dataset_version_id_source_path'))
    )
    op.create_index(op.f('ix_dataset_images_dataset_version_id_split'), 'dataset_images', ['dataset_version_id', 'split'], unique=False)
    op.create_table('model_versions',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('model_id', sa.UUID(), nullable=False),
    sa.Column('version_number', sa.Integer(), nullable=False),
    sa.Column('display_name', sa.Text(), nullable=True),
    sa.Column('source', sa.Enum('trained', 'imported', name='model_source', native_enum=False, create_constraint=True, length=32), nullable=False),
    sa.Column('status', sa.Enum('draft', 'ready', 'archived', name='model_version_status', native_enum=False, create_constraint=True, length=32), server_default='draft', nullable=False),
    sa.Column('architecture', sa.Text(), nullable=False),
    sa.Column('framework', sa.Text(), nullable=False),
    sa.Column('framework_version', sa.Text(), nullable=True),
    sa.Column('input_width', sa.Integer(), nullable=False),
    sa.Column('input_height', sa.Integer(), nullable=False),
    sa.Column('checkpoint_media_file_id', sa.UUID(), nullable=True),
    sa.Column('onnx_media_file_id', sa.UUID(), nullable=True),
    sa.Column('training_dataset_version_id', sa.UUID(), nullable=True),
    sa.Column('training_run_id', sa.UUID(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("btrim(architecture) <> ''", name=op.f('ck_model_versions_architecture_not_blank')),
    sa.CheckConstraint("btrim(framework) <> ''", name=op.f('ck_model_versions_framework_not_blank')),
    sa.CheckConstraint('checkpoint_media_file_id IS NOT NULL OR onnx_media_file_id IS NOT NULL', name=op.f('ck_model_versions_has_weights_file')),
    sa.CheckConstraint('input_width > 0 AND input_height > 0', name=op.f('ck_model_versions_input_size_positive')),
    sa.CheckConstraint('version_number > 0', name=op.f('ck_model_versions_version_number_positive')),
    sa.ForeignKeyConstraint(['checkpoint_media_file_id'], ['media_files.id'], name=op.f('fk_model_versions_checkpoint_media_file_id')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_model_versions_created_by')),
    sa.ForeignKeyConstraint(['model_id'], ['models.id'], name=op.f('fk_model_versions_model_id')),
    sa.ForeignKeyConstraint(['onnx_media_file_id'], ['media_files.id'], name=op.f('fk_model_versions_onnx_media_file_id')),
    sa.ForeignKeyConstraint(['training_dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_model_versions_training_dataset_version_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_versions')),
    sa.UniqueConstraint('model_id', 'version_number', name=op.f('uq_model_versions_model_id_version_number')),
    sa.UniqueConstraint('training_run_id', name=op.f('uq_model_versions_training_run_id'))
    )
    op.create_table('dataset_annotations',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('dataset_image_id', sa.BigInteger(), nullable=False),
    sa.Column('category_id', sa.SmallInteger(), nullable=False),
    sa.Column('x_min', sa.Double(), nullable=False),
    sa.Column('y_min', sa.Double(), nullable=False),
    sa.Column('x_max', sa.Double(), nullable=False),
    sa.Column('y_max', sa.Double(), nullable=False),
    sa.Column('source_annotation_id', sa.BigInteger(), nullable=True),
    sa.CheckConstraint('0 <= x_min AND x_min < x_max AND x_max <= 1 AND 0 <= y_min AND y_min < y_max AND y_max <= 1', name=op.f('ck_dataset_annotations_bbox_valid')),
    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_dataset_annotations_category_id')),
    sa.ForeignKeyConstraint(['dataset_image_id'], ['dataset_images.id'], name=op.f('fk_dataset_annotations_dataset_image_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_annotations'))
    )
    op.create_index(op.f('ix_dataset_annotations_dataset_image_id'), 'dataset_annotations', ['dataset_image_id'], unique=False)
    op.create_table('dataset_image_rejections',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('dataset_image_id', sa.BigInteger(), nullable=False),
    sa.Column('category', sa.Text(), nullable=False),
    sa.Column('detail', sa.Text(), nullable=False),
    sa.CheckConstraint("btrim(category) <> ''", name=op.f('ck_dataset_image_rejections_category_not_blank')),
    sa.CheckConstraint("btrim(detail) <> ''", name=op.f('ck_dataset_image_rejections_detail_not_blank')),
    sa.ForeignKeyConstraint(['dataset_image_id'], ['dataset_images.id'], name=op.f('fk_dataset_image_rejections_dataset_image_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_dataset_image_rejections'))
    )
    op.create_index(op.f('ix_dataset_image_rejections_dataset_image_id'), 'dataset_image_rejections', ['dataset_image_id'], unique=False)
    op.create_table('inspection_sessions',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('recording_id', sa.UUID(), nullable=False),
    sa.Column('model_version_id', sa.UUID(), nullable=False),
    sa.Column('name', sa.Text(), nullable=True),
    sa.Column('status', sa.Enum('queued', 'running', 'completed', 'failed', 'cancelled', name='job_status', native_enum=False, create_constraint=True, length=32), server_default='queued', nullable=False),
    sa.Column('interval_start_ms', sa.BigInteger(), server_default=sa.text('0'), nullable=False),
    sa.Column('interval_end_ms', sa.BigInteger(), nullable=True),
    sa.Column('sample_fps', sa.Double(), nullable=True),
    sa.Column('confidence_threshold', sa.Double(), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(status IN ('completed', 'failed', 'cancelled')) = (finished_at IS NOT NULL)", name=op.f('ck_inspection_sessions_finished_at_matches_status')),
    sa.CheckConstraint("status <> 'failed' OR error_message IS NOT NULL", name=op.f('ck_inspection_sessions_failed_has_error_message')),
    sa.CheckConstraint("status NOT IN ('running') OR started_at IS NOT NULL", name=op.f('ck_inspection_sessions_active_has_started_at')),
    sa.CheckConstraint('confidence_threshold >= 0 AND confidence_threshold <= 1', name=op.f('ck_inspection_sessions_confidence_threshold_in_unit_interval')),
    sa.CheckConstraint('finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at', name=op.f('ck_inspection_sessions_finished_after_started')),
    sa.CheckConstraint('interval_end_ms IS NULL OR interval_end_ms > interval_start_ms', name=op.f('ck_inspection_sessions_interval_end_after_start')),
    sa.CheckConstraint('interval_start_ms >= 0', name=op.f('ck_inspection_sessions_interval_start_non_negative')),
    sa.CheckConstraint('sample_fps IS NULL OR sample_fps > 0', name=op.f('ck_inspection_sessions_sample_fps_positive')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_inspection_sessions_created_by')),
    sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], name=op.f('fk_inspection_sessions_model_version_id')),
    sa.ForeignKeyConstraint(['recording_id'], ['recordings.id'], name=op.f('fk_inspection_sessions_recording_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_inspection_sessions'))
    )
    op.create_index(op.f('ix_inspection_sessions_created_at'), 'inspection_sessions', ['created_at'], unique=False)
    op.create_index(op.f('ix_inspection_sessions_model_version_id'), 'inspection_sessions', ['model_version_id'], unique=False)
    op.create_index(op.f('ix_inspection_sessions_recording_id'), 'inspection_sessions', ['recording_id'], unique=False)
    op.create_table('model_evaluations',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('model_version_id', sa.UUID(), nullable=False),
    sa.Column('dataset_version_id', sa.UUID(), nullable=False),
    sa.Column('split', sa.Enum('train', 'valid', 'test', name='dataset_split', native_enum=False, create_constraint=True, length=32), nullable=False),
    sa.Column('iou_threshold', sa.Double(), nullable=False),
    sa.Column('confidence_threshold', sa.Double(), nullable=False),
    sa.Column('status', sa.Enum('queued', 'running', 'completed', 'failed', 'cancelled', name='job_status', native_enum=False, create_constraint=True, length=32), server_default='queued', nullable=False),
    sa.Column('precision', sa.Double(), nullable=True),
    sa.Column('recall', sa.Double(), nullable=True),
    sa.Column('f1', sa.Double(), nullable=True),
    sa.Column('map50', sa.Double(), nullable=True),
    sa.Column('map50_95', sa.Double(), nullable=True),
    sa.Column('median_latency_ms', sa.Double(), nullable=True),
    sa.Column('hardware', sa.Text(), nullable=True),
    sa.Column('confusion_matrix', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(status IN ('completed', 'failed', 'cancelled')) = (finished_at IS NOT NULL)", name=op.f('ck_model_evaluations_finished_at_matches_status')),
    sa.CheckConstraint("status <> 'completed' OR (precision IS NOT NULL AND recall IS NOT NULL AND map50 IS NOT NULL)", name=op.f('ck_model_evaluations_completed_has_metrics')),
    sa.CheckConstraint("status <> 'failed' OR error_message IS NOT NULL", name=op.f('ck_model_evaluations_failed_has_error_message')),
    sa.CheckConstraint("status NOT IN ('running') OR started_at IS NOT NULL", name=op.f('ck_model_evaluations_active_has_started_at')),
    sa.CheckConstraint('confidence_threshold >= 0 AND confidence_threshold <= 1', name=op.f('ck_model_evaluations_confidence_threshold_in_unit_interval')),
    sa.CheckConstraint('f1 IS NULL OR (f1 >= 0 AND f1 <= 1)', name=op.f('ck_model_evaluations_f1_in_unit_interval')),
    sa.CheckConstraint('finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at', name=op.f('ck_model_evaluations_finished_after_started')),
    sa.CheckConstraint('iou_threshold > 0 AND iou_threshold <= 1', name=op.f('ck_model_evaluations_iou_threshold_range')),
    sa.CheckConstraint('map50 IS NULL OR (map50 >= 0 AND map50 <= 1)', name=op.f('ck_model_evaluations_map50_in_unit_interval')),
    sa.CheckConstraint('map50_95 IS NULL OR (map50_95 >= 0 AND map50_95 <= 1)', name=op.f('ck_model_evaluations_map50_95_in_unit_interval')),
    sa.CheckConstraint('median_latency_ms IS NULL OR median_latency_ms >= 0', name=op.f('ck_model_evaluations_latency_non_negative')),
    sa.CheckConstraint('precision IS NULL OR (precision >= 0 AND precision <= 1)', name=op.f('ck_model_evaluations_precision_in_unit_interval')),
    sa.CheckConstraint('recall IS NULL OR (recall >= 0 AND recall <= 1)', name=op.f('ck_model_evaluations_recall_in_unit_interval')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_model_evaluations_created_by')),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_model_evaluations_dataset_version_id')),
    sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], name=op.f('fk_model_evaluations_model_version_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_model_evaluations'))
    )
    op.create_index(op.f('ix_model_evaluations_model_version_id'), 'model_evaluations', ['model_version_id'], unique=False)
    op.create_table('model_version_classes',
    sa.Column('model_version_id', sa.UUID(), nullable=False),
    sa.Column('class_index', sa.Integer(), nullable=False),
    sa.Column('category_id', sa.SmallInteger(), nullable=False),
    sa.CheckConstraint('class_index >= 0', name=op.f('ck_model_version_classes_class_index_non_negative')),
    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_model_version_classes_category_id')),
    sa.ForeignKeyConstraint(['model_version_id'], ['model_versions.id'], name=op.f('fk_model_version_classes_model_version_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('model_version_id', 'class_index', name=op.f('pk_model_version_classes')),
    sa.UniqueConstraint('model_version_id', 'category_id', name=op.f('uq_model_version_classes_model_version_id_category_id'))
    )
    op.create_table('training_runs',
    sa.Column('id', sa.UUID(), server_default=sa.text('uuidv7()'), nullable=False),
    sa.Column('dataset_version_id', sa.UUID(), nullable=False),
    sa.Column('base_model_version_id', sa.UUID(), nullable=True),
    sa.Column('status', sa.Enum('queued', 'running', 'stopping', 'completed', 'stopped', 'failed', name='training_status', native_enum=False, create_constraint=True, length=32), server_default='queued', nullable=False),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('total_epochs', sa.Integer(), nullable=False),
    sa.Column('current_epoch', sa.Integer(), server_default=sa.text('0'), nullable=False),
    sa.Column('best_epoch', sa.Integer(), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('log_media_file_id', sa.UUID(), nullable=True),
    sa.Column('created_by', sa.UUID(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("(status IN ('completed', 'stopped', 'failed')) = (finished_at IS NOT NULL)", name=op.f('ck_training_runs_finished_at_matches_status')),
    sa.CheckConstraint("status <> 'failed' OR error_message IS NOT NULL", name=op.f('ck_training_runs_failed_has_error_message')),
    sa.CheckConstraint("status NOT IN ('running', 'stopping') OR started_at IS NOT NULL", name=op.f('ck_training_runs_active_has_started_at')),
    sa.CheckConstraint('best_epoch IS NULL OR (best_epoch >= 1 AND best_epoch <= current_epoch)', name=op.f('ck_training_runs_best_epoch_in_range')),
    sa.CheckConstraint('current_epoch >= 0 AND current_epoch <= total_epochs', name=op.f('ck_training_runs_current_epoch_in_range')),
    sa.CheckConstraint('finished_at IS NULL OR started_at IS NULL OR finished_at >= started_at', name=op.f('ck_training_runs_finished_after_started')),
    sa.CheckConstraint('total_epochs > 0', name=op.f('ck_training_runs_total_epochs_positive')),
    sa.ForeignKeyConstraint(['base_model_version_id'], ['model_versions.id'], name=op.f('fk_training_runs_base_model_version_id')),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], name=op.f('fk_training_runs_created_by')),
    sa.ForeignKeyConstraint(['dataset_version_id'], ['dataset_versions.id'], name=op.f('fk_training_runs_dataset_version_id')),
    sa.ForeignKeyConstraint(['log_media_file_id'], ['media_files.id'], name=op.f('fk_training_runs_log_media_file_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_training_runs'))
    )
    # model_versions and training_runs reference each other, so this side
    # of the cycle is added once both tables exist.
    op.create_foreign_key(
        op.f('fk_model_versions_training_run_id'),
        'model_versions', 'training_runs', ['training_run_id'], ['id'],
    )
    op.create_table('evaluation_category_metrics',
    sa.Column('evaluation_id', sa.UUID(), nullable=False),
    sa.Column('category_id', sa.SmallInteger(), nullable=False),
    sa.Column('precision', sa.Double(), nullable=True),
    sa.Column('recall', sa.Double(), nullable=True),
    sa.Column('f1', sa.Double(), nullable=True),
    sa.Column('ap50', sa.Double(), nullable=True),
    sa.Column('support', sa.Integer(), nullable=False),
    sa.CheckConstraint('ap50 IS NULL OR (ap50 >= 0 AND ap50 <= 1)', name=op.f('ck_evaluation_category_metrics_ap50_in_unit_interval')),
    sa.CheckConstraint('f1 IS NULL OR (f1 >= 0 AND f1 <= 1)', name=op.f('ck_evaluation_category_metrics_f1_in_unit_interval')),
    sa.CheckConstraint('precision IS NULL OR (precision >= 0 AND precision <= 1)', name=op.f('ck_evaluation_category_metrics_precision_in_unit_interval')),
    sa.CheckConstraint('recall IS NULL OR (recall >= 0 AND recall <= 1)', name=op.f('ck_evaluation_category_metrics_recall_in_unit_interval')),
    sa.CheckConstraint('support >= 0', name=op.f('ck_evaluation_category_metrics_support_non_negative')),
    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_evaluation_category_metrics_category_id')),
    sa.ForeignKeyConstraint(['evaluation_id'], ['model_evaluations.id'], name=op.f('fk_evaluation_category_metrics_evaluation_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('evaluation_id', 'category_id', name=op.f('pk_evaluation_category_metrics'))
    )
    op.create_table('evaluation_results',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('evaluation_id', sa.UUID(), nullable=False),
    sa.Column('dataset_image_id', sa.BigInteger(), nullable=False),
    sa.Column('outcome', sa.Enum('true_positive', 'false_positive', 'false_negative', name='evaluation_outcome', native_enum=False, create_constraint=True, length=32), nullable=False),
    sa.Column('dataset_annotation_id', sa.BigInteger(), nullable=True),
    sa.Column('predicted_category_id', sa.SmallInteger(), nullable=True),
    sa.Column('confidence', sa.Double(), nullable=True),
    sa.Column('pred_x_min', sa.Double(), nullable=True),
    sa.Column('pred_y_min', sa.Double(), nullable=True),
    sa.Column('pred_x_max', sa.Double(), nullable=True),
    sa.Column('pred_y_max', sa.Double(), nullable=True),
    sa.Column('iou', sa.Double(), nullable=True),
    sa.CheckConstraint("(outcome = 'true_positive' AND dataset_annotation_id IS NOT NULL AND predicted_category_id IS NOT NULL AND confidence IS NOT NULL AND pred_x_min IS NOT NULL) OR (outcome = 'false_positive' AND dataset_annotation_id IS NULL AND predicted_category_id IS NOT NULL AND confidence IS NOT NULL AND pred_x_min IS NOT NULL) OR (outcome = 'false_negative' AND dataset_annotation_id IS NOT NULL AND predicted_category_id IS NULL AND confidence IS NULL AND pred_x_min IS NULL)", name=op.f('ck_evaluation_results_outcome_fields_consistent')),
    sa.CheckConstraint('(pred_x_min IS NULL AND pred_y_min IS NULL AND pred_x_max IS NULL AND pred_y_max IS NULL) OR (pred_x_min IS NOT NULL AND pred_y_min IS NOT NULL AND pred_x_max IS NOT NULL AND pred_y_max IS NOT NULL AND 0 <= pred_x_min AND pred_x_min < pred_x_max AND pred_x_max <= 1 AND 0 <= pred_y_min AND pred_y_min < pred_y_max AND pred_y_max <= 1)', name=op.f('ck_evaluation_results_pred_bbox_valid')),
    sa.CheckConstraint('confidence IS NULL OR (confidence >= 0 AND confidence <= 1)', name=op.f('ck_evaluation_results_confidence_in_unit_interval')),
    sa.CheckConstraint('iou IS NULL OR (iou >= 0 AND iou <= 1)', name=op.f('ck_evaluation_results_iou_in_unit_interval')),
    sa.ForeignKeyConstraint(['dataset_annotation_id'], ['dataset_annotations.id'], name=op.f('fk_evaluation_results_dataset_annotation_id')),
    sa.ForeignKeyConstraint(['dataset_image_id'], ['dataset_images.id'], name=op.f('fk_evaluation_results_dataset_image_id')),
    sa.ForeignKeyConstraint(['evaluation_id'], ['model_evaluations.id'], name=op.f('fk_evaluation_results_evaluation_id'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['predicted_category_id'], ['categories.id'], name=op.f('fk_evaluation_results_predicted_category_id')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_evaluation_results'))
    )
    op.create_index(op.f('ix_evaluation_results_evaluation_id_outcome'), 'evaluation_results', ['evaluation_id', 'outcome'], unique=False)
    op.create_table('frames',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('media_file_id', sa.UUID(), nullable=False),
    sa.Column('frame_index', sa.Integer(), nullable=True),
    sa.Column('timestamp_ms', sa.BigInteger(), nullable=False),
    sa.CheckConstraint('frame_index IS NULL OR frame_index >= 0', name=op.f('ck_frames_frame_index_non_negative')),
    sa.CheckConstraint('timestamp_ms >= 0', name=op.f('ck_frames_timestamp_non_negative')),
    sa.ForeignKeyConstraint(['media_file_id'], ['media_files.id'], name=op.f('fk_frames_media_file_id')),
    sa.ForeignKeyConstraint(['session_id'], ['inspection_sessions.id'], name=op.f('fk_frames_session_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_frames')),
    sa.UniqueConstraint('session_id', 'media_file_id', 'frame_index', name=op.f('uq_frames_session_id_media_file_id_frame_index'), postgresql_nulls_not_distinct=True)
    )
    op.create_index(op.f('ix_frames_session_id_timestamp_ms'), 'frames', ['session_id', 'timestamp_ms'], unique=False)
    op.create_table('inspection_session_categories',
    sa.Column('session_id', sa.UUID(), nullable=False),
    sa.Column('category_id', sa.SmallInteger(), nullable=False),
    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_inspection_session_categories_category_id')),
    sa.ForeignKeyConstraint(['session_id'], ['inspection_sessions.id'], name=op.f('fk_inspection_session_categories_session_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('session_id', 'category_id', name=op.f('pk_inspection_session_categories'))
    )
    op.create_table('training_epoch_metrics',
    sa.Column('training_run_id', sa.UUID(), nullable=False),
    sa.Column('epoch', sa.Integer(), nullable=False),
    sa.Column('train_loss', sa.Double(), nullable=True),
    sa.Column('val_loss', sa.Double(), nullable=True),
    sa.Column('val_map50', sa.Double(), nullable=True),
    sa.Column('val_map50_95', sa.Double(), nullable=True),
    sa.Column('val_precision', sa.Double(), nullable=True),
    sa.Column('val_recall', sa.Double(), nullable=True),
    sa.Column('extra', postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
    sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('epoch > 0', name=op.f('ck_training_epoch_metrics_epoch_positive')),
    sa.CheckConstraint('val_map50 IS NULL OR (val_map50 >= 0 AND val_map50 <= 1)', name=op.f('ck_training_epoch_metrics_val_map50_in_unit_interval')),
    sa.CheckConstraint('val_map50_95 IS NULL OR (val_map50_95 >= 0 AND val_map50_95 <= 1)', name=op.f('ck_training_epoch_metrics_val_map50_95_in_unit_interval')),
    sa.CheckConstraint('val_precision IS NULL OR (val_precision >= 0 AND val_precision <= 1)', name=op.f('ck_training_epoch_metrics_val_precision_in_unit_interval')),
    sa.CheckConstraint('val_recall IS NULL OR (val_recall >= 0 AND val_recall <= 1)', name=op.f('ck_training_epoch_metrics_val_recall_in_unit_interval')),
    sa.ForeignKeyConstraint(['training_run_id'], ['training_runs.id'], name=op.f('fk_training_epoch_metrics_training_run_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('training_run_id', 'epoch', name=op.f('pk_training_epoch_metrics'))
    )
    op.create_table('detections',
    sa.Column('id', sa.BigInteger(), sa.Identity(always=False), nullable=False),
    sa.Column('frame_id', sa.BigInteger(), nullable=False),
    sa.Column('category_id', sa.SmallInteger(), nullable=False),
    sa.Column('confidence', sa.Double(), nullable=False),
    sa.Column('x_min', sa.Double(), nullable=False),
    sa.Column('y_min', sa.Double(), nullable=False),
    sa.Column('x_max', sa.Double(), nullable=False),
    sa.Column('y_max', sa.Double(), nullable=False),
    sa.CheckConstraint('0 <= x_min AND x_min < x_max AND x_max <= 1 AND 0 <= y_min AND y_min < y_max AND y_max <= 1', name=op.f('ck_detections_bbox_valid')),
    sa.CheckConstraint('confidence >= 0 AND confidence <= 1', name=op.f('ck_detections_confidence_in_unit_interval')),
    sa.ForeignKeyConstraint(['category_id'], ['categories.id'], name=op.f('fk_detections_category_id')),
    sa.ForeignKeyConstraint(['frame_id'], ['frames.id'], name=op.f('fk_detections_frame_id'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_detections'))
    )
    op.create_index(op.f('ix_detections_frame_id'), 'detections', ['frame_id'], unique=False)
    _seed_categories()


def downgrade() -> None:
    op.drop_constraint(op.f('fk_model_versions_training_run_id'), 'model_versions', type_='foreignkey')
    op.drop_index(op.f('ix_detections_frame_id'), table_name='detections')
    op.drop_table('detections')
    op.drop_table('training_epoch_metrics')
    op.drop_table('inspection_session_categories')
    op.drop_index(op.f('ix_frames_session_id_timestamp_ms'), table_name='frames')
    op.drop_table('frames')
    op.drop_index(op.f('ix_evaluation_results_evaluation_id_outcome'), table_name='evaluation_results')
    op.drop_table('evaluation_results')
    op.drop_table('evaluation_category_metrics')
    op.drop_table('training_runs')
    op.drop_table('model_version_classes')
    op.drop_index(op.f('ix_model_evaluations_model_version_id'), table_name='model_evaluations')
    op.drop_table('model_evaluations')
    op.drop_index(op.f('ix_inspection_sessions_recording_id'), table_name='inspection_sessions')
    op.drop_index(op.f('ix_inspection_sessions_model_version_id'), table_name='inspection_sessions')
    op.drop_index(op.f('ix_inspection_sessions_created_at'), table_name='inspection_sessions')
    op.drop_table('inspection_sessions')
    op.drop_index(op.f('ix_dataset_image_rejections_dataset_image_id'), table_name='dataset_image_rejections')
    op.drop_table('dataset_image_rejections')
    op.drop_index(op.f('ix_dataset_annotations_dataset_image_id'), table_name='dataset_annotations')
    op.drop_table('dataset_annotations')
    op.drop_table('model_versions')
    op.drop_index(op.f('ix_dataset_images_dataset_version_id_split'), table_name='dataset_images')
    op.drop_table('dataset_images')
    op.drop_table('recording_files')
    op.drop_table('dataset_versions')
    op.drop_table('recordings')
    op.drop_table('models')
    op.drop_index(op.f('ix_media_files_sha256'), table_name='media_files')
    op.drop_table('media_files')
    op.drop_table('datasets')
    op.drop_table('users')
    op.drop_table('categories')


# Frozen copy of the categories at this revision. Later changes to the
# list belong in a new migration, not here.
SEED_CATEGORIES = [
    ('almond', 'Almond', 'Acceptable almonds', 'acceptable', 10),
    ('defect', 'Defect', 'Damaged or defective almonds', 'defect', 20),
    ('rock', 'Rock', 'Stones and mineral fragments', 'foreign_material', 30),
    ('stick', 'Stick', 'Twigs and woody material', 'foreign_material', 40),
    ('shell', 'Shell', 'Hard shell fragments', 'foreign_material', 50),
    ('hull', 'Hull', 'Outer husk material', 'foreign_material', 60),
]


def _seed_categories() -> None:
    categories = sa.table(
        'categories',
        sa.column('key', sa.Text()),
        sa.column('display_name', sa.Text()),
        sa.column('description', sa.Text()),
        sa.column('kind', sa.Text()),
        sa.column('sort_order', sa.SmallInteger()),
    )
    op.bulk_insert(
        categories,
        [
            dict(key=key, display_name=name, description=description, kind=kind, sort_order=order)
            for key, name, description, kind, order in SEED_CATEGORIES
        ],
    )
