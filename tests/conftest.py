"""Shared fixtures for PostgreSQL-backed storage tests.

Database tests create throwaway databases on the server configured in
``.env`` (for developers, the docker compose server), migrate them, and drop
them afterwards. Your development database is never touched.

When no server is reachable these tests are skipped with the reason shown by
``pytest -rs``. Set ``DINOS_REQUIRE_DB=1`` (as CI should) to fail instead.
"""

import dataclasses
import os
import uuid
from collections.abc import Callable, Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

from app.storage.config import DatabaseSettings, load_settings
from app.storage.database import check_connection, create_db_engine
from app.storage.errors import StorageError
from app.storage.media import MediaStore
from app.storage.schema import upgrade_to_head


def _database_unavailable(reason: str) -> None:
    if os.environ.get("DINOS_REQUIRE_DB") == "1":
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def server_settings() -> DatabaseSettings:
    """Connection settings from .env, verified against a running server."""
    try:
        settings = load_settings().database
        engine = create_db_engine(settings, database="postgres")
        try:
            check_connection(engine, settings)
        finally:
            engine.dispose()
    except StorageError as error:
        _database_unavailable(f"PostgreSQL is not available for database tests. {error}")
    return settings


@pytest.fixture(scope="session")
def create_database(server_settings: DatabaseSettings) -> Iterator[Callable[[], DatabaseSettings]]:
    """Factory for empty databases; every one is dropped at the end of the run."""
    admin = create_db_engine(server_settings, database="postgres", isolation_level="AUTOCOMMIT")
    created: list[str] = []

    def create() -> DatabaseSettings:
        name = f"dinos_test_{uuid.uuid4().hex[:12]}"
        with admin.connect() as connection:
            connection.execute(text(
                f'CREATE DATABASE "{name}" TEMPLATE template0 ENCODING \'UTF8\' '
                f"LOCALE_PROVIDER builtin BUILTIN_LOCALE 'C.UTF-8'"
            ))
        created.append(name)
        return dataclasses.replace(server_settings, name=name)

    yield create

    with admin.connect() as connection:
        for name in created:
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def empty_database(create_database: Callable[[], DatabaseSettings]) -> DatabaseSettings:
    return create_database()


@pytest.fixture(scope="session")
def migrated_engine(create_database: Callable[[], DatabaseSettings]) -> Iterator[Engine]:
    settings = create_database()
    upgrade_to_head(settings)
    engine = create_db_engine(settings)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(migrated_engine: Engine) -> Iterator[Session]:
    """A session whose work is rolled back after each test.

    ``session.commit()`` inside a test only releases a savepoint, so tests
    can commit and re-query without leaking rows into other tests.
    """
    connection = migrated_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def media_store(tmp_path) -> MediaStore:
    return MediaStore.open(tmp_path / "media")
