"""Load database and media settings from the environment or a ``.env`` file.

Nothing here is hardcoded: connection details and the media root come from
``DINOS_*`` variables, so the same code runs on a developer Mac (Docker) and
on the client's Windows PC (native PostgreSQL install). Credentials live in
the untracked ``.env`` file; ``.env.example`` documents every variable.

Precedence: process environment first, then the ``.env`` file. The file is
``DINOS_ENV_FILE`` when set, otherwise ``.env`` in the current directory. A
missing file is fine as long as every required variable is set elsewhere.

A relative ``DINOS_MEDIA_ROOT`` is resolved against the directory of the
file that defined it, or the current directory when it came from the
process environment. That keeps ``./data/media`` stable no matter where the
app is launched from.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values
from sqlalchemy.engine import URL

from app.storage.errors import ConfigError


ENV_FILE_VARIABLE = "DINOS_ENV_FILE"
DEFAULT_ENV_FILE = ".env"
DEFAULT_PORT = 5432

REQUIRED_VARIABLES = (
    "DINOS_DB_HOST",
    "DINOS_DB_NAME",
    "DINOS_DB_USER",
    "DINOS_DB_PASSWORD",
    "DINOS_MEDIA_ROOT",
)


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str = field(repr=False)

    def url(self, database: str | None = None) -> URL:
        """SQLAlchemy URL for ``database`` (default: the configured one).

        ``URL.create`` escapes special characters in the password, so values
        such as ``p@ss:word/1`` need no manual quoting in ``.env``.
        """
        return URL.create(
            "postgresql+psycopg",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=database or self.name,
        )

    def describe(self) -> str:
        """Location for error messages. Never includes the password."""
        return f"{self.user}@{self.host}:{self.port}/{self.name}"


@dataclass(frozen=True)
class Settings:
    database: DatabaseSettings
    media_root: Path
    pg_bin: Path | None = None


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: str | Path | None = None,
) -> Settings:
    """Return validated settings, or raise ConfigError naming what to fix.

    ``environ`` defaults to ``os.environ``. ``env_file`` overrides the
    ``DINOS_ENV_FILE`` lookup; tests pass both to stay isolated.
    """
    environ = os.environ if environ is None else environ
    file_path = _env_file_path(environ, env_file)
    file_values = _read_env_file(file_path)

    values: dict[str, str] = {}
    origins: dict[str, Path] = {}
    for name in (*REQUIRED_VARIABLES, "DINOS_DB_PORT", "DINOS_PG_BIN"):
        if environ.get(name, "").strip():
            values[name] = environ[name].strip()
            origins[name] = Path.cwd()
        elif (file_values.get(name) or "").strip():
            values[name] = file_values[name].strip()
            origins[name] = file_path.parent

    missing = [name for name in REQUIRED_VARIABLES if name not in values]
    if missing:
        raise ConfigError(
            f"Missing required setting(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill them in, or set them as "
            f"environment variables. Looked for a settings file at {file_path}."
        )

    database = DatabaseSettings(
        host=values["DINOS_DB_HOST"],
        port=_port(values.get("DINOS_DB_PORT")),
        name=values["DINOS_DB_NAME"],
        user=values["DINOS_DB_USER"],
        password=values["DINOS_DB_PASSWORD"],
    )
    media_root = Path(values["DINOS_MEDIA_ROOT"]).expanduser()
    if not media_root.is_absolute():
        media_root = origins["DINOS_MEDIA_ROOT"] / media_root
    pg_bin = Path(values["DINOS_PG_BIN"]).expanduser() if "DINOS_PG_BIN" in values else None
    if pg_bin is not None and not pg_bin.is_absolute():
        pg_bin = origins["DINOS_PG_BIN"] / pg_bin
    return Settings(database=database, media_root=media_root.resolve(),
                    pg_bin=pg_bin.resolve() if pg_bin is not None else None)


def _env_file_path(environ: Mapping[str, str], env_file: str | Path | None) -> Path:
    if env_file is not None:
        return Path(env_file).expanduser().resolve()
    configured = environ.get(ENV_FILE_VARIABLE, "").strip()
    return Path(configured or DEFAULT_ENV_FILE).expanduser().resolve()


def _read_env_file(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {}
    if not path.is_file():
        raise ConfigError(f"Settings path {path} is not a file. Point DINOS_ENV_FILE at a .env file.")
    try:
        return dotenv_values(path, encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ConfigError(
            f"Could not read settings file {path}: {error}. "
            f"Check its permissions and save it as UTF-8."
        ) from error


def _port(raw: str | None) -> int:
    if raw is None:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        port = 0
    if not 1 <= port <= 65535:
        raise ConfigError(
            f"DINOS_DB_PORT must be a number from 1 to 65535, not {raw!r}. "
            f"PostgreSQL's default port is {DEFAULT_PORT}."
        )
    return port
