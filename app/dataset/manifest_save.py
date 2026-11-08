"""Save a built manifest as one JSON file per dataset version.

``save_manifest`` reuses ``manifest_json``. It does not rebuild records,
summarize the dataset, or modify source images and annotation files.
An existing destination is left unchanged. The final name is published
only after the temporary file holds the complete document.
"""

import os
import uuid
from pathlib import Path

from app.dataset.manifest import DatasetManifest, manifest_json


class ManifestSaveError(ValueError):
    """The manifest could not be saved."""


class ManifestDestinationError(ManifestSaveError):
    """A manifest for this version already exists."""


def save_manifest(manifest: DatasetManifest, output_directory: Path) -> Path:
    """Write ``manifest`` to ``{output_directory}/{version}.json`` and return that path.

    The version label must be a single file name. Labels that contain a
    path separator, a drive, or ``..`` are rejected before anything is
    written. Publication links a finished temporary file into place and
    removes that temporary file afterward, including when publication fails.
    """
    root = Path(output_directory).resolve()
    destination = _destination(root, manifest.dataset_version)
    root.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ManifestDestinationError(
            f"A manifest for version {manifest.dataset_version.strip()} already exists."
        )

    temporary = root / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        _write_complete(temporary, manifest_json(manifest) + "\n")
        _publish(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _destination(output_directory: Path, dataset_version: str) -> Path:
    filename = _version_filename(dataset_version)
    destination = (output_directory / filename).resolve()
    if destination.parent != output_directory:
        raise ManifestSaveError(
            "Dataset version labels must stay inside the output directory."
        )
    return destination


def _version_filename(dataset_version: str) -> str:
    label = dataset_version.strip()
    path = Path(label)
    if (
        not label
        or label in {".", ".."}
        or any(character in label for character in "/\\:\x00")
        or path.is_absolute()
        or path.drive
        or path.root
        or ".." in path.parts
        or path.parts != (path.name,)
        or path.name != label
    ):
        raise ManifestSaveError(
            "Dataset version labels must be a single file name inside the output directory."
        )
    return f"{label}.json"


def _write_complete(path: Path, text: str) -> None:
    with path.open("xb") as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise ManifestDestinationError(
            f"A manifest already exists at {destination}."
        ) from error
