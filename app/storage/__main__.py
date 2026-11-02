"""Database and media setup commands. Run from the project root:

    python -m app.storage migrate   create or update the schema (safe to repeat)
    python -m app.storage check     verify settings, database, schema, and media root

Exit codes: 0 = success, 1 = a check failed with an actionable message.
"""

import argparse
import sys

from app.storage.config import load_settings
from app.storage.database import check_connection, create_db_engine
from app.storage.errors import StorageError
from app.storage.media import MediaStore
from app.storage.schema import ensure_schema_current, upgrade_to_head


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
    parser.add_argument("command", choices=("migrate", "check"))
    args = parser.parse_args()
    try:
        {"migrate": migrate, "check": check}[args.command]()
    except StorageError as error:
        # Keep progress lines above the error when both streams share a pipe.
        sys.stdout.flush()
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
