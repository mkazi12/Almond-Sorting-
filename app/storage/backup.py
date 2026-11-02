"""Portable custom-format backups with verified media and empty-target restore."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import UTC, datetime

from sqlalchemy import text, select

from app.storage.database import create_db_engine, check_connection, session_factory
from app.storage.errors import BackupError, StorageError
from app.storage.media import MediaStore, PARTIAL_SUFFIX
from app.storage.models import MediaFile, InspectionSession, JobStatus, TrainingRun, TrainingStatus, ModelEvaluation
from app.storage.operations import MAINTENANCE_LOCK
from app.storage.schema import current_revision, ensure_schema_current, head_revision
from app.storage.verification import sha256_file, verify_media


def _run(tool, arguments, database):
    env = dict(os.environ, PGPASSWORD=database.password, PGCONNECT_TIMEOUT='10')
    try:
        return subprocess.run([str(tool), *arguments], env=env, check=True,
                              capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        # Avoid leaking credentials if libpq or a wrapper echoes its environment.
        detail = (getattr(error, 'stderr', None) or str(error)).replace(database.password, '***')
        raise BackupError(f'PostgreSQL tool failed: {detail.strip()}. Check client tools, connection, and target permissions.') from error


def _tool(settings, name, server_major):
    candidate = settings.pg_bin / (name + ('.exe' if os.name == 'nt' else '')) if settings.pg_bin else shutil.which(name)
    if candidate is None or not Path(candidate).is_file():
        raise BackupError(f'{name} was not found. Install PostgreSQL {server_major} client tools and set DINOS_PG_BIN.')
    result = _run(candidate, ['--version'], settings.database)
    version = re.search(r'(\d+)\.', result.stdout)
    if version is None or int(version[1]) < server_major:
        raise BackupError(f'{name} must be PostgreSQL {server_major} or newer. Set DINOS_PG_BIN to compatible tools.')
    return Path(candidate)


def _connection_args(database):
    return ['--host', database.host, '--port', str(database.port), '--username', database.user,
            '--dbname', database.name, '--no-password']


def _issue_error(issues):
    return '; '.join(f'{i.relative_path}: {i.reason} ({", ".join(i.owners)})' for i in issues)


def backup(settings, destination):
    destination = Path(destination).resolve()
    media_root = settings.media_root.resolve()
    if destination.exists() or destination.is_relative_to(media_root) or media_root.is_relative_to(destination):
        raise BackupError('Choose a new backup directory outside the media root; existing backups are never overwritten.')
    engine = create_db_engine(settings.database)
    try:
        major = check_connection(engine, settings.database)
        ensure_schema_current(engine)
        tool = _tool(settings, 'pg_dump', major)
        with engine.connect() as lock:
            acquired = lock.scalar(text('SELECT pg_try_advisory_lock(:key)'), {'key': MAINTENANCE_LOCK})
            if not acquired:
                raise BackupError('Storage writes are active. Wait for them to finish and retry backup.')
            try:
                with session_factory(engine)() as session:
                    for model, statuses in ((InspectionSession, [JobStatus.RUNNING]),
                        (TrainingRun, [TrainingStatus.RUNNING, TrainingStatus.STOPPING]),
                        (ModelEvaluation, [JobStatus.RUNNING])):
                        if session.scalar(select(model.id).where(model.status.in_(statuses)).limit(1)):
                            raise BackupError('Analysis, training, or evaluation is active. Stop jobs before backing up.')
                    issues = verify_media(session, MediaStore(media_root), full=True)
                    if issues:
                        raise BackupError('Repair media before backup: ' + _issue_error(issues))
                    registered = {m.relative_path: (m.size_bytes, m.sha256) for m in session.scalars(select(MediaFile))}
                destination.mkdir(parents=True)
                _run(tool, [*_connection_args(settings.database), '--format=custom',
                            '--file', str(destination / 'database.dump')], settings.database)
                copied = destination / 'media'
                copied.mkdir()
                entries = []
                for source in sorted(media_root.rglob('*')):
                    if not source.is_file() or source.name.endswith(PARTIAL_SUFFIX):
                        continue
                    relative = source.relative_to(media_root).as_posix()
                    source = MediaStore(media_root).resolve(relative)
                    target = copied / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    size, digest = target.stat().st_size, sha256_file(target)
                    if relative in registered and registered[relative] != (size, digest):
                        raise BackupError(f'Media changed during backup: {relative}. Stop writers and retry into a new directory.')
                    entries.append(dict(path=relative, size_bytes=size, sha256=digest))
                if not registered.keys() <= {e['path'] for e in entries}:
                    raise BackupError('A registered media file disappeared during backup. Repair media and retry.')
                manifest = dict(format_version=1, schema_revision=current_revision(engine),
                    postgres_version=major, created_at=datetime.now(UTC).isoformat(),
                    postgres_server_version=lock.scalar(text('SHOW server_version')),
                    file_count=len(entries), files=entries,
                    database_sha256=sha256_file(destination / 'database.dump'))
                (destination / 'backup.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
                return manifest
            finally:
                lock.execute(text('SELECT pg_advisory_unlock(:key)'), {'key': MAINTENANCE_LOCK})
    except OSError as error:
        raise BackupError(f'Backup could not be written: {error}. Check free space and permissions; retry in a new directory.') from error
    finally:
        engine.dispose()


def _manifest(source):
    try:
        manifest = json.loads((source / 'backup.json').read_text(encoding='utf-8'))
        if (manifest['format_version'] != 1 or manifest['schema_revision'] != head_revision()
                or manifest['file_count'] != len(manifest['files'])
                or not isinstance(manifest['postgres_version'], int)):
            raise ValueError('Unsupported backup format or schema revision')
        if sha256_file(source / 'database.dump') != manifest['database_sha256']:
            raise ValueError('Database dump SHA-256 mismatch')
        store = MediaStore(source / 'media')
        seen = set()
        for entry in manifest['files']:
            path = store.resolve(entry['path'])
            if entry['path'] in seen or path.stat().st_size != entry['size_bytes'] or sha256_file(path) != entry['sha256']:
                raise ValueError(f'Duplicate or corrupt backup media: {entry["path"]}')
            seen.add(entry['path'])
        return manifest
    except (OSError, ValueError, KeyError, TypeError, StorageError) as error:
        raise BackupError(f'Backup is incomplete, incompatible, or corrupt: {error}. Select a complete backup for this app version.') from error


def restore(settings, source):
    source = Path(source).resolve()
    manifest = _manifest(source)
    root = settings.media_root.resolve()
    if root.is_relative_to(source) or source.is_relative_to(root):
        raise BackupError('Restore media root and backup directory must not overlap.')
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise BackupError('Restore requires an empty media directory. Set DINOS_MEDIA_ROOT to a new folder.')
    engine = create_db_engine(settings.database)
    try:
        major = check_connection(engine, settings.database)
        if major < manifest['postgres_version']:
            raise BackupError('Restore server is older than the backup. Use the same or a newer PostgreSQL major version.')
        tool = _tool(settings, 'pg_restore', max(major, manifest['postgres_version']))
        with engine.connect() as lock:
            if not lock.scalar(text('SELECT pg_try_advisory_lock(:key)'), {'key': MAINTENANCE_LOCK}):
                raise BackupError('Storage writes are active. Stop the application before restore.')
            try:
                objects = lock.scalar(text("SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg_toast%'"))
                routines = lock.scalar(text("SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                    "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')"))
                types = lock.scalar(text("SELECT count(*) FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
                    "WHERE n.nspname NOT IN ('pg_catalog', 'information_schema') AND n.nspname NOT LIKE 'pg_toast%'"))
                schemas = lock.scalar(text("SELECT count(*) FROM pg_namespace WHERE nspname NOT IN "
                    "('public', 'pg_catalog', 'information_schema') AND nspname NOT LIKE 'pg_%'"))
                if objects or routines or types or schemas:
                    raise BackupError('Restore target database is not empty. Create a fresh database; do not migrate it before restore.')
                root.mkdir(parents=True, exist_ok=True)
                try:
                    # Verify/copy media before restoring SQL; failed SQL is rolled back.
                    for entry in manifest['files']:
                        target = MediaStore(root).path_for(entry['path'])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(MediaStore(source / 'media').resolve(entry['path']), target)
                        if target.stat().st_size != entry['size_bytes'] or sha256_file(target) != entry['sha256']:
                            raise BackupError(f'Copy verification failed for {entry["path"]}. Retry into fresh targets.')
                    _run(tool, [*_connection_args(settings.database), '--no-owner', '--no-acl',
                        '--exit-on-error', '--single-transaction', str(source / 'database.dump')], settings.database)
                except BaseException:
                    # Remove only files this restore copied, never an existing destination.
                    for entry in manifest['files']:
                        MediaStore(root).path_for(entry['path']).unlink(missing_ok=True)
                    for directory in sorted((p for p in root.rglob('*') if p.is_dir()),
                                            key=lambda p: len(p.parts), reverse=True):
                        try:
                            directory.rmdir()  # Empty directories only; never recurse-delete.
                        except OSError:
                            pass
                    raise
                ensure_schema_current(engine)
                with session_factory(engine)() as session:
                    issues = verify_media(session, MediaStore(root), full=True)
                    if issues:
                        raise BackupError('Restore verification failed; keep the app stopped: ' + _issue_error(issues))
                return manifest
            finally:
                lock.execute(text('SELECT pg_advisory_unlock(:key)'), {'key': MAINTENANCE_LOCK})
    except OSError as error:
        raise BackupError(f'Restore could not finish: {error}. Check permissions and space; retry into fresh targets.') from error
    finally:
        engine.dispose()
