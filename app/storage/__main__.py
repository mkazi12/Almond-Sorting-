"""Database and media setup commands. Run from the project root:

    python -m app.storage migrate   create or update the schema (safe to repeat)
    python -m app.storage check     verify settings, database, schema, and media root
    python -m app.storage sample-data       create/reopen the labeled sample
    python -m app.storage verify-media --full   verify all files and SHA-256
    python -m app.storage backup DIRECTORY  create a new complete backup
    python -m app.storage restore DIRECTORY restore into empty database/media

Exit codes: 0 = success, 1 = a check failed with an actionable message.
"""

import argparse
import sys

from app.storage.config import load_settings
from app.storage.database import check_connection, create_db_engine
from app.storage.errors import StorageError
from app.storage.media import MediaStore
from app.storage.schema import ensure_schema_current, upgrade_to_head
from app.storage.database import session_factory
from app.storage.backup import backup, restore
from app.storage.sample_data import create_sample_data
from app.storage.verification import verify_media
from app.storage.inspections import get_session_summary, frames_between
from app.storage.media import media_path
from app.storage.errors import ValidationError


def migrate() -> None:
    settings = load_settings()
    revision = upgrade_to_head(settings.database)
    print(f"Database {settings.database.describe()} is at revision {revision}.")


def check() -> None:
    settings = load_settings()
    print(f"Settings: database {settings.database.describe()}, media root {settings.media_root}")
    engine = create_db_engine(settings.database)
    try:
        major = check_connection(engine, settings.database)
        print(f"Database: connected (PostgreSQL {major}).")
        ensure_schema_current(engine)
        print("Schema: up to date.")
    finally:
        engine.dispose()
    MediaStore.open(settings.media_root)
    print("Media root: exists and is writable.")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.storage", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("migrate", "check", "sample-data"):
        subparsers.add_parser(name)
    verify = subparsers.add_parser("verify-media")
    verify.add_argument("--full", action="store_true", help="Verify SHA-256 as well as size")
    for name in ("backup", "restore"):
        command = subparsers.add_parser(name)
        command.add_argument("directory")
    args = parser.parse_args()
    try:
        if args.command in ("migrate", "check"):
            {"migrate": migrate, "check": check}[args.command]()
        elif args.command in ("backup", "restore"):
            manifest = {"backup": backup, "restore": restore}[args.command](load_settings(), args.directory)
            print(f"{args.command.capitalize()} verified: {manifest['file_count']} media files.")
        else:
            settings = load_settings()
            engine = create_db_engine(settings.database)
            try:
                check_connection(engine, settings.database)
                ensure_schema_current(engine)
                store = MediaStore.open(settings.media_root)
                with session_factory(engine).begin() as session:
                    if args.command == "sample-data":
                        row = create_sample_data(session, store)
                        summary = get_session_summary(session, row.id)
                        print(f"Sample session: {row.id}\nModel: {summary['model_label']} (placeholder only)")
                        print(f"ONNX: {media_path(store, summary['model_version'].onnx_file)}")
                        for frame in frames_between(session, row.id, 0, 1):
                            print(f"Frame {frame.id}: {media_path(store, frame.media_file)}")
                    else:
                        issues = verify_media(session, store, full=args.full)
                        for issue in issues:
                            print(f"{issue.relative_path}: {issue.reason}; used by {', '.join(issue.owners) or 'unreferenced media'}")
                        if issues:
                            raise ValidationError(f"{len(issues)} media file(s) failed verification. Restore missing/corrupt files from backup.")
                        print("All registered media files verified.")
            finally:
                engine.dispose()
    except StorageError as error:
        # Keep progress lines above the error when both streams share a pipe.
        sys.stdout.flush()
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
