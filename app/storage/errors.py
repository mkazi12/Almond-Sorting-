"""Expected storage failures with messages suitable for the user.

Every message says what went wrong and what to do next, so callers can
show ``str(error)`` directly instead of a stack trace.
"""


class StorageError(Exception):
    """Base class for expected database and media storage failures."""


class ConfigError(StorageError):
    """A required setting is missing or invalid."""


class DatabaseConnectionError(StorageError):
    """PostgreSQL could not be reached or rejected the connection."""


class UnsupportedServerError(StorageError):
    """The PostgreSQL server is older than the version the schema needs."""


class SchemaVersionError(StorageError):
    """The database schema does not match the version this app expects."""


class MediaStorageError(StorageError):
    """A media file could not be saved or read beneath the media root."""


class MediaPathError(MediaStorageError):
    """A stored relative path is absolute or escapes the media root."""


class MediaMissingError(MediaStorageError):
    """A database row points at a media file that is no longer on disk."""


class ValidationError(StorageError):
    """An operation's inputs or state are invalid."""


class NotFoundError(StorageError):
    """The requested stored record does not exist."""
