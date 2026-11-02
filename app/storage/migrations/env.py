"""Alembic environment for the DINOS schema.

Runs from the ``alembic`` CLI (reads settings from ``.env``) or from
``app.storage.schema`` (passes an engine in ``config.attributes``).
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import text

from app.storage.config import load_settings
from app.storage.database import check_connection, create_db_engine
from app.storage.models import Base


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Shared by every DINOS process. Holding this transaction-scoped advisory
# lock makes concurrent migration runs wait for each other, so the second
# one sees the finished schema and does nothing.
MIGRATION_LOCK_ID = 4046470001


def run_migrations_offline() -> None:
    """Print SQL instead of executing it (``alembic upgrade head --sql``)."""
    context.configure(
        url=load_settings().database.url().render_as_string(hide_password=False),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = config.attributes.get("engine")
    owns_engine = engine is None
    if owns_engine:
        settings = load_settings().database
        engine = create_db_engine(settings)
        check_connection(engine, settings)
    try:
        with engine.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                connection.execute(
                    text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": MIGRATION_LOCK_ID}
                )
                context.run_migrations()
    finally:
        if owns_engine:
            engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
