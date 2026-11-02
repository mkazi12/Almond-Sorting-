"""Apply and inspect Alembic migrations from Python.

``upgrade_to_head`` is safe to run on every start: on an up-to-date database
it does nothing, and it never deletes or duplicates rows. Concurrent
processes (two app windows starting at once) are serialized by an advisory
lock in ``migrations/env.py``. Alembic's command API is process-global, so
call this from one thread at a time within a process.
"""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

from app.storage.config import DatabaseSettings
from app.storage.database import check_connection, create_db_engine
from app.storage.errors import SchemaVersionError


MIGRATIONS_DIRECTORY = Path(__file__).resolve().parent / "migrations"


def alembic_config(engine: Engine | None = None) -> Config:
    """Alembic configuration that does not depend on the working directory.

    When ``engine`` is given, migrations run on it; otherwise ``env.py``
    loads settings from the environment like the ``alembic`` CLI does.
    """
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIRECTORY))
    config.set_main_option("path_separator", "os")
    if engine is not None:
        config.attributes["engine"] = engine
    return config


def head_revision() -> str:
    heads = ScriptDirectory.from_config(alembic_config()).get_heads()
    if len(heads) != 1:
        raise SchemaVersionError(
            f"Expected one migration head but found {len(heads)}: {', '.join(heads)}. "
            f"Merge the branches with `alembic merge heads` before running the app."
        )
    return heads[0]


def current_revision(engine: Engine) -> str | None:
    """Revision recorded in the database, or None for an empty database."""
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def upgrade_to_head(settings: DatabaseSettings) -> str:
    """Connect, apply any pending migrations, and return the head revision."""
    engine = create_db_engine(settings)
    try:
        check_connection(engine, settings)
        command.upgrade(alembic_config(engine), "head")
        return current_revision(engine) or ""
    finally:
        engine.dispose()


def ensure_schema_current(engine: Engine) -> None:
    """Raise SchemaVersionError unless the database is at the app's head.

    Call at startup so an old or empty database fails with instructions
    instead of an obscure "relation does not exist" error later.
    """
    head = head_revision()
    current = current_revision(engine)
    if current == head:
        return
    if current is None:
        raise SchemaVersionError(
            "The database has no DINOS tables yet. Run `python -m app.storage migrate` "
            "to create them."
        )
    known = {script.revision for script in ScriptDirectory.from_config(alembic_config()).walk_revisions()}
    if current not in known:
        raise SchemaVersionError(
            f"The database schema ({current}) is newer than this app ({head}). "
            f"Update the app to the latest version before using this database."
        )
    raise SchemaVersionError(
        f"The database schema ({current}) is older than this app ({head}). "
        f"Run `python -m app.storage migrate` to update it."
    )
