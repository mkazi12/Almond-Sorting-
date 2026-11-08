"""Portable dataset paths shared by records, import and split tooling."""

from pathlib import Path, PureWindowsPath


class DatasetPathError(ValueError):
    """A dataset path is absolute or escapes the dataset root."""


def normalize_relative_path(value: str | Path, *, allow_root: bool = False) -> str:
    """Return a normalized POSIX path without a drive, traversal or NUL.

    Both slash conventions are understood on every operating system, so a
    manifest copied from Windows has the same identity on Linux.
    """
    if isinstance(value, Path):
        value = str(value)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise DatasetPathError(f"Invalid relative dataset path: {value!r}.")
    if value.startswith(("/", "\\")) or PureWindowsPath(value).drive:
        raise DatasetPathError(f"Absolute or network path is forbidden: {value!r}.")
    parts = value.replace("\\", "/").split("/")
    if any(part == ".." or ":" in part for part in parts):
        raise DatasetPathError(f"Dataset path escapes its root: {value!r}.")
    clean = [part for part in parts if part not in ("", ".")]
    if not clean:
        if allow_root:
            return "."
        raise DatasetPathError(f"Empty image or annotation path: {value!r}.")
    return "/".join(clean)


def relative_dataset_path(path: str | Path) -> Path:
    """Return one portable, nonempty dataset-relative path."""
    return Path(normalize_relative_path(path))


def relative_path_identity(path: str | Path) -> str:
    """Identity of an image or annotation path, independent of its root."""
    return normalize_relative_path(path)


def resolve_dataset_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve one path and reject symlinks that leave the dataset root."""
    relative = normalize_relative_path(relative_path)
    try:
        base = Path(root).resolve()
        candidate = (base / relative).resolve()
    except (OSError, RuntimeError) as error:
        raise DatasetPathError(f"Cannot resolve dataset path {relative!r}: {error}") from error
    if not candidate.is_relative_to(base):
        raise DatasetPathError(f"File {relative!r} resolves outside dataset root.")
    return candidate
