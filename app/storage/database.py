"""Create database engines and turn connection failures into actionable errors.

All database access goes through ``create_db_engine`` and ``session_factory``
so a future API service can reuse this package unchanged. UI code should not
build its own engines or SQL strings.
"""

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.storage.config import DatabaseSettings
from app.storage.errors import DatabaseConnectionError, UnsupportedServerError


# uuidv7() and the built-in C.UTF-8 collation both need PostgreSQL 18.
MINIMUM_SERVER_VERSION = 18
CONNECT_TIMEOUT_SECONDS = 5


def create_db_engine(settings: DatabaseSettings, database: str | None = None, **kwargs) -> Engine:
    """Return an engine for ``database`` (default: the configured database).

    Creating an engine does not connect. Call ``check_connection`` to fail
    early with a clear message instead of on the first query.
    """
    connect_args = {
        "connect_timeout": CONNECT_TIMEOUT_SECONDS,
        "application_name": "dinos",
        **kwargs.pop("connect_args", {}),
    }
    return create_engine(
        settings.url(database),
        pool_pre_ping=True,
        connect_args=connect_args,
        **kwargs,
    )


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions that keep loaded attributes usable after commit.

    Qt widgets often read a row after the transaction that saved it has
    ended; ``expire_on_commit=False`` avoids surprise reloads there.
    """
    return sessionmaker(engine, expire_on_commit=False)


def check_connection(engine: Engine, settings: DatabaseSettings) -> int:
    """Connect once and return the server's major version.

    Raises DatabaseConnectionError or UnsupportedServerError with a message
    the user can act on.
    """
    try:
        with engine.connect() as connection:
            version_num = int(connection.execute(text("SHOW server_version_num")).scalar_one())
    except OperationalError as error:
        raise connection_error(error, settings) from error

    major = version_num // 10000
    if major < MINIMUM_SERVER_VERSION:
        raise UnsupportedServerError(
            f"PostgreSQL {major} is running at {settings.host}:{settings.port}, but DINOS "
            f"needs PostgreSQL {MINIMUM_SERVER_VERSION} or newer. Install PostgreSQL "
            f"{MINIMUM_SERVER_VERSION} and point DINOS_DB_HOST/DINOS_DB_PORT at it."
        )
    return major


def connection_error(error: DBAPIError, settings: DatabaseSettings) -> DatabaseConnectionError:
    """Map a driver connection failure to a message that says what to fix.

    psycopg does not attach SQLSTATE codes to connection failures, so this
    matches libpq's English messages and falls back to the original text.
    """
    detail = str(error.orig).strip() if error.orig is not None else str(error)
    message = detail.lower()
    location = f"{settings.host}:{settings.port}"

    if "password authentication failed" in message or "no pg_hba.conf entry" in message:
        hint = (
            f"PostgreSQL at {location} rejected user {settings.user!r}. Check "
            f"DINOS_DB_USER and DINOS_DB_PASSWORD in your .env file."
        )
    elif "does not exist" in message and "database" in message:
        hint = (
            f"Database {settings.name!r} does not exist on {location}. Create it "
            f"(see docs/database.md) or fix DINOS_DB_NAME in your .env file."
        )
    elif "failed to resolve host" in message or "could not translate host name" in message:
        hint = (
            f"Could not find host {settings.host!r}. Check DINOS_DB_HOST in your "
            f".env file; use localhost when PostgreSQL runs on this computer."
        )
    elif "connection refused" in message or "timeout expired" in message or "could not connect" in message:
        hint = (
            f"Could not reach PostgreSQL at {location}. Start the server "
            f"(developers: `docker compose up -d`; Windows: start the "
            f"postgresql-x64-{MINIMUM_SERVER_VERSION} service) and check DINOS_DB_HOST "
            f"and DINOS_DB_PORT in your .env file."
        )
    else:
        hint = (
            f"Could not connect to PostgreSQL at {location}. Check the server is "
            f"running and the DINOS_DB_* settings in your .env file are correct."
        )
    # Keep the first line of the driver's message for troubleshooting; the
    # rest repeats the same failure for each address localhost resolves to.
    first_line = detail.splitlines()[0] if detail else ""
    return DatabaseConnectionError(f"{hint}\nDetails: {first_line}")
