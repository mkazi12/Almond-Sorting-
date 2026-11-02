"""SCRUM-47: migrations create the schema and are safe to run repeatedly."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text

from app.storage.database import create_db_engine
from app.storage.errors import SchemaVersionError
from app.storage.models import Base
from app.storage.schema import (
    alembic_config,
    current_revision,
    ensure_schema_current,
    head_revision,
    upgrade_to_head,
)


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def engine_for(empty_database):
    engine = create_db_engine(empty_database)
    yield engine
    engine.dispose()


def table_names(engine):
    return set(inspect(engine).get_table_names())


def scalar(engine, sql):
    with engine.connect() as connection:
        return connection.execute(text(sql)).scalar_one()


def test_upgrade_creates_the_full_schema_on_an_empty_database(empty_database, engine_for):
    assert table_names(engine_for) == set()

    assert upgrade_to_head(empty_database) == head_revision()

    assert table_names(engine_for) == set(Base.metadata.tables) | {"alembic_version"}
    assert scalar(engine_for, "SELECT count(*) FROM categories") == 6


def test_rerunning_upgrade_keeps_existing_rows(empty_database, engine_for):
    upgrade_to_head(empty_database)
    with engine_for.begin() as connection:
        connection.execute(text("INSERT INTO users (employee_id, display_name) VALUES ('E1', 'Op')"))
        connection.execute(text("INSERT INTO recordings (name) VALUES ('Morning inspection')"))

    upgrade_to_head(empty_database)
    upgrade_to_head(empty_database)

    assert scalar(engine_for, "SELECT count(*) FROM users") == 1
    assert scalar(engine_for, "SELECT count(*) FROM recordings") == 1
    assert scalar(engine_for, "SELECT count(*) FROM categories") == 6
    assert current_revision(engine_for) == head_revision()


def test_concurrent_migrate_commands_do_not_conflict_or_duplicate_seed_rows(empty_database, engine_for):
    # Separate processes, like two app windows starting at once. (Alembic's
    # command API is process-global, so threads are not a valid stand-in.)
    environment = {
        **os.environ,
        "DINOS_DB_HOST": empty_database.host,
        "DINOS_DB_PORT": str(empty_database.port),
        "DINOS_DB_NAME": empty_database.name,
        "DINOS_DB_USER": empty_database.user,
        "DINOS_DB_PASSWORD": empty_database.password,
        "DINOS_MEDIA_ROOT": "unused",
    }
    processes = [
        subprocess.Popen([sys.executable, "-m", "app.storage", "migrate"], env=environment,
                         cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for _ in range(3)
    ]
    results = [process.communicate(timeout=120) + (process.returncode,) for process in processes]

    for stdout, stderr, returncode in results:
        assert returncode == 0, stderr
        assert f"is at revision {head_revision()}" in stdout
    assert scalar(engine_for, "SELECT count(*) FROM categories") == 6


def test_downgrade_to_base_and_upgrade_again(empty_database, engine_for):
    upgrade_to_head(empty_database)

    command.downgrade(alembic_config(engine_for), "base")
    assert table_names(engine_for) == {"alembic_version"}

    upgrade_to_head(empty_database)
    assert table_names(engine_for) == set(Base.metadata.tables) | {"alembic_version"}


def test_models_and_migrations_describe_the_same_schema(migrated_engine):
    """Fails when models.py changes without a matching migration.

    Fix by running `alembic revision --autogenerate -m "..."` and reviewing it.
    """
    with migrated_engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert differences == []


def test_startup_check_explains_how_to_fix_an_empty_database(engine_for):
    with pytest.raises(SchemaVersionError, match=r"python -m app.storage migrate"):
        ensure_schema_current(engine_for)


def test_startup_check_passes_on_a_migrated_database(migrated_engine):
    ensure_schema_current(migrated_engine)


def test_startup_check_rejects_a_newer_database(empty_database, engine_for):
    upgrade_to_head(empty_database)
    with engine_for.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num = 'from_the_future'"))

    with pytest.raises(SchemaVersionError, match="newer than this app"):
        ensure_schema_current(engine_for)
