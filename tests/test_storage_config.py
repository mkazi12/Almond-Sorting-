"""SCRUM-47: connection settings and the media root come from configuration."""

from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

from app.storage.config import load_settings
from app.storage.errors import ConfigError


COMPLETE = {
    "DINOS_DB_HOST": "db.local",
    "DINOS_DB_PORT": "5433",
    "DINOS_DB_NAME": "dinos",
    "DINOS_DB_USER": "operator",
    "DINOS_DB_PASSWORD": "p@ss:w/rd#1",
    "DINOS_MEDIA_ROOT": "media",
}


def write_env(path: Path, values: dict[str, str]) -> Path:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return path


def test_settings_load_from_an_env_file(tmp_path):
    env_file = write_env(tmp_path / ".env", COMPLETE)

    settings = load_settings(environ={}, env_file=env_file)

    assert settings.database.host == "db.local"
    assert settings.database.port == 5433
    assert settings.database.user == "operator"
    assert settings.media_root == (tmp_path / "media").resolve()


def test_environment_variables_override_the_file(tmp_path):
    env_file = write_env(tmp_path / ".env", COMPLETE)

    settings = load_settings(environ={"DINOS_DB_HOST": "override.local"}, env_file=env_file)

    assert settings.database.host == "override.local"
    assert settings.database.name == "dinos"


def test_env_file_location_can_be_configured(tmp_path):
    env_file = write_env(tmp_path / "custom.env", COMPLETE)

    settings = load_settings(environ={"DINOS_ENV_FILE": str(env_file)})

    assert settings.database.host == "db.local"


def test_relative_media_root_from_the_environment_uses_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings = load_settings(environ=COMPLETE, env_file=tmp_path / "missing.env")
    assert settings.media_root == (tmp_path / "media").resolve()


def test_absolute_media_root_is_kept(tmp_path):
    root = tmp_path / "elsewhere"
    settings = load_settings(environ={**COMPLETE, "DINOS_MEDIA_ROOT": str(root)},
                             env_file=tmp_path / "missing.env")
    assert settings.media_root == root.resolve()


def test_default_port_is_5432(tmp_path):
    values = {key: value for key, value in COMPLETE.items() if key != "DINOS_DB_PORT"}
    settings = load_settings(environ=values, env_file=tmp_path / "missing.env")
    assert settings.database.port == 5432


def test_missing_settings_are_named_with_next_steps(tmp_path):
    values = {key: value for key, value in COMPLETE.items()
              if key not in {"DINOS_DB_PASSWORD", "DINOS_MEDIA_ROOT"}}

    with pytest.raises(ConfigError) as caught:
        load_settings(environ=values, env_file=tmp_path / "missing.env")

    message = str(caught.value)
    assert "DINOS_DB_PASSWORD" in message
    assert "DINOS_MEDIA_ROOT" in message
    assert ".env.example" in message


def test_blank_values_count_as_missing(tmp_path):
    with pytest.raises(ConfigError, match="DINOS_DB_HOST"):
        load_settings(environ={**COMPLETE, "DINOS_DB_HOST": "  "}, env_file=tmp_path / "missing.env")


@pytest.mark.parametrize("port", ["abc", "0", "70000"])
def test_invalid_port_is_rejected(tmp_path, port):
    with pytest.raises(ConfigError, match="DINOS_DB_PORT"):
        load_settings(environ={**COMPLETE, "DINOS_DB_PORT": port}, env_file=tmp_path / "missing.env")


def test_env_file_path_must_be_a_file(tmp_path):
    with pytest.raises(ConfigError, match="DINOS_ENV_FILE"):
        load_settings(environ=COMPLETE, env_file=tmp_path)


def test_passwords_with_special_characters_survive_the_url(tmp_path):
    settings = load_settings(environ=COMPLETE, env_file=tmp_path / "missing.env")

    url = make_url(settings.database.url().render_as_string(hide_password=False))

    assert url.password == "p@ss:w/rd#1"
    assert url.host == "db.local"
    assert url.database == "dinos"


def test_password_is_never_shown(tmp_path):
    settings = load_settings(environ=COMPLETE, env_file=tmp_path / "missing.env")

    assert "p@ss" not in repr(settings)
    assert "p@ss" not in settings.database.describe()
    assert "p@ss" not in str(settings.database.url())
