"""SCRUM-47: invalid database settings produce actionable errors."""

import dataclasses

import pytest

from app.storage import database
from app.storage.config import DatabaseSettings
from app.storage.database import check_connection, create_db_engine
from app.storage.errors import DatabaseConnectionError, UnsupportedServerError


def connect(settings: DatabaseSettings) -> None:
    engine = create_db_engine(settings)
    try:
        check_connection(engine, settings)
    finally:
        engine.dispose()


def test_unreachable_server_says_how_to_start_it():
    # Port 1 on localhost is never a PostgreSQL server, so no server is needed.
    settings = DatabaseSettings(host="127.0.0.1", port=1, name="dinos", user="dinos",
                                password="secret-value")
    with pytest.raises(DatabaseConnectionError) as caught:
        connect(settings)

    message = str(caught.value)
    assert "127.0.0.1:1" in message
    assert "docker compose up -d" in message
    assert "DINOS_DB_PORT" in message
    assert "secret-value" not in message


def test_unknown_host_points_at_the_host_setting():
    settings = DatabaseSettings(host="no-such-host.invalid", port=5432, name="dinos",
                                user="dinos", password="x")
    with pytest.raises(DatabaseConnectionError, match="DINOS_DB_HOST"):
        connect(settings)


def test_wrong_password_points_at_the_credentials(server_settings):
    settings = dataclasses.replace(server_settings, password="definitely-wrong-password")
    with pytest.raises(DatabaseConnectionError) as caught:
        connect(settings)

    message = str(caught.value)
    assert "DINOS_DB_PASSWORD" in message
    assert "definitely-wrong-password" not in message


def test_missing_database_points_at_the_name_setting(server_settings):
    settings = dataclasses.replace(server_settings, name="dinos_no_such_database")
    with pytest.raises(DatabaseConnectionError, match="DINOS_DB_NAME"):
        connect(settings)


def test_old_server_versions_are_rejected(server_settings, monkeypatch):
    monkeypatch.setattr(database, "MINIMUM_SERVER_VERSION", 99)
    with pytest.raises(UnsupportedServerError, match="PostgreSQL 99 or newer"):
        connect(server_settings)


def test_valid_settings_report_the_server_version(server_settings):
    engine = create_db_engine(server_settings)
    try:
        assert check_connection(engine, server_settings) >= 18
    finally:
        engine.dispose()
