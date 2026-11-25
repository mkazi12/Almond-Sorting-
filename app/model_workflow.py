"""Transactional local package transfer and active-model selection."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from threading import RLock
from types import MappingProxyType
import zipfile

from app.model_registry import DEFAULT_REGISTRY_ROOT, _read_model, _unique_object


class ModelError(ValueError):
    """A package, checkpoint or saved selection cannot be used."""


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def package_checkpoint_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or name != str(PurePosixPath(name)) or name in {".", "model.json"}):
        raise ModelError("Invalid checkpoint path in package.")
    return name


def compatible_record(path):
    """Validate the package contract and hash without deserializing weights."""
    record = read_json(path)
    model = _read_model(Path(path), record=record)
    package_checkpoint_name(record["checkpoint"])
    if Path(path).stat().st_size > 1024 * 1024:
        raise ModelError("model.json exceeds the 1 MiB metadata limit.")
    if record.get("format") != "rfdetr-state-dict-v1" or record.get("rfdetr_version") != "1.3.0":
        raise ModelError("Incompatible format: expected rfdetr-state-dict-v1 for RF-DETR 1.3.0.")
    if model.variant not in {"nano", "small", "medium", "base", "large"}:
        raise ModelError(f"Unsupported RF-DETR variant: {model.variant}")
    digest = record.get("checkpoint_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ModelError("Missing or invalid checkpoint_sha256.")
    if not model.checkpoint_available:
        raise ModelError(f"Missing checkpoint file: {model.checkpoint.name}")
    if not 0 < model.checkpoint.stat().st_size <= 4 * 1024**3:
        raise ModelError("Checkpoint must be a nonempty file of at most 4 GiB.")
    if sha256(model.checkpoint) != digest:
        raise ModelError("Checkpoint SHA-256 mismatch: package is corrupt or weights changed.")
    if not isinstance(record.get("training_provenance"), dict) or not record["training_provenance"]:
        raise ModelError("Missing training_provenance object.")
    if "validation_summary" not in record:
        raise ModelError("Missing validation_summary (use null when unavailable).")
    if record["validation_summary"] is not None and model.validation_summary is None:
        raise ModelError("Invalid validation_summary: expected a nonempty object or null.")
    outputs = record.get("output_class_map")
    if not isinstance(outputs, list) or not outputs or len(outputs) > 10000:
        raise ModelError("Missing output_class_map: supply one category ID or null per classifier output.")
    ids = [value for value in outputs if value is not None]
    if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)) or set(ids) != set(model.class_map):
        raise ModelError("output_class_map must map each recorded class exactly once; use null for unused outputs.")
    try:
        json.dumps(record, allow_nan=False)
    except ValueError as error:
        raise ModelError("Metadata contains non-finite numbers.") from error
    return model, record


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".selection-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(record, stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def default_loader(model, record):
    from app.rfdetr_detector import load_detector
    return load_detector(model, record)


@dataclass(frozen=True)
class ActiveModel:
    model_id: str
    model_version: str
    checkpoint_sha256: str
    detector: object
    class_map: object

    @property
    def label(self):
        return f"{self.model_id} / {self.model_version} / {self.checkpoint_sha256}"


class ModelWorkflow:
    def __init__(self, root=DEFAULT_REGISTRY_ROOT, *, loader=None):
        self.root = Path(root).resolve()
        self.loader = loader or default_loader
        self.active = None
        self._lock = RLock()

    def _path(self, model_id):
        if not isinstance(model_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", model_id):
            raise ModelError("Invalid model ID.")
        path = self.root / model_id / "model.json"
        if not path.resolve().is_relative_to(self.root):
            raise ModelError("Model metadata must stay inside the registry.")
        return path

    def _candidate(self, path):
        model, record = compatible_record(path)
        detector = self.loader(model, record)
        if not callable(getattr(detector, "predict", None)):
            raise ModelError("Model loader did not return a detector.")
        # Detect a replacement during load, before publishing active state.
        if read_json(path) != record or sha256(model.checkpoint) != record["checkpoint_sha256"]:
            raise ModelError("Model files changed during loading. Retry the operation.")
        return ActiveModel(model.model_id, model.model_version, record["checkpoint_sha256"],
                           detector, MappingProxyType(dict(model.class_map)))

    def activate(self, model_id):
        with self._lock:
            try:
                path = self._path(model_id)
                candidate = self._candidate(path)
                atomic_json(self.root / "active_model.json", {
                    "schema_version": 1, "model_id": model_id,
                    "checkpoint_sha256": candidate.checkpoint_sha256,
                    "metadata_sha256": sha256(path),
                })
                self.active = candidate
                return candidate
            except Exception as error:
                raise ModelError(f"Could not activate {model_id}: {error}") from error

    def restore(self):
        with self._lock:
            selection = self.root / "active_model.json"
            if not selection.exists():
                return None
            try:
                saved = read_json(selection)
                if not isinstance(saved, dict) or saved.get("schema_version") != 1:
                    raise ModelError("Invalid saved active-model selection.")
                path = self._path(saved.get("model_id"))
                if sha256(path) != saved.get("metadata_sha256"):
                    raise ModelError("Active model metadata changed since selection.")
                candidate = self._candidate(path)
                if candidate.checkpoint_sha256 != saved.get("checkpoint_sha256"):
                    raise ModelError("Active checkpoint changed since selection.")
                self.active = candidate
                return candidate
            except Exception as error:
                raise ModelError(f"Could not restore active model: {error}") from error

    def import_package(self, source):
        """Validate a staged ZIP and load its candidate before publishing once."""
        with self._lock:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".import-", dir=self.root) as staging:
                    stage = Path(staging)
                    with zipfile.ZipFile(source) as archive:
                        infos = archive.infolist()
                        names = [item.filename for item in infos]
                        if len(names) != len(set(names)) or len(names) > 3:
                            raise ModelError("Package must contain model.json and one checkpoint, without duplicate members.")
                        if "model.json" not in names:
                            raise ModelError("Package is missing model.json.")
                        if archive.getinfo("model.json").file_size > 1024 * 1024:
                            raise ModelError("model.json exceeds the 1 MiB metadata limit.")
                        raw = archive.read("model.json")
                        record = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_object)
                        if not isinstance(record, dict):
                            raise ModelError("model.json must contain an object.")
                        model_id = record.get("model_id")
                        destination = self._path(model_id).parent
                        if destination.exists():
                            raise ModelError(f"Model ID {model_id} already exists; it was not imported again.")
                        name = package_checkpoint_name(record.get("checkpoint"))
                        if name not in names:
                            raise ModelError(f"Package is missing checkpoint: {name}")
                        if set(names) != {"model.json", name}:
                            raise ModelError("Unexpected package members; expected only metadata and checkpoint.")
                        info = archive.getinfo(name)
                        if info.is_dir() or not 0 < info.file_size <= 4 * 1024**3:
                            raise ModelError("Checkpoint must be a nonempty file of at most 4 GiB.")
                        folder = stage / model_id
                        folder.mkdir()
                        (folder / "model.json").write_bytes(raw)
                        target = folder / name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    candidate = self._candidate(folder / "model.json")
                    del candidate  # Import validates but never activates the model.
                    # rename must not replace a pre-existing model directory.
                    if destination.exists():
                        raise ModelError(f"Model ID {model_id} already exists.")
                    folder.rename(destination)
                return _read_model(destination / "model.json")
            except Exception as error:
                raise ModelError(f"Could not import package: {error}") from error

    def export_package(self, model_id, destination):
        """Keep the exact metadata bytes and verify the streamed checkpoint hash."""
        with self._lock:
            temporary = None
            try:
                path = self._path(model_id)
                model, record = compatible_record(path)
                destination = Path(destination).resolve()
                if destination.is_relative_to(self.root):
                    raise ModelError("Export outside the registry to avoid replacing model files.")
                # No overwrite: selecting a new name also protects existing packages.
                if destination.exists():
                    raise ModelError("Export destination already exists; choose a new filename.")
                # A reusable export must also pass the actual checkpoint loader.
                self._candidate(path)
                fd, temporary = tempfile.mkstemp(prefix=".export-", dir=destination.parent)
                os.close(fd)
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                    archive.write(path, "model.json")
                    archive.write(model.checkpoint, record["checkpoint"])
                with zipfile.ZipFile(temporary) as archive:
                    with archive.open(record["checkpoint"]) as stream:
                        digest = hashlib.file_digest(stream, "sha256").hexdigest()
                    if digest != record["checkpoint_sha256"] or json.loads(archive.read("model.json")) != record:
                        raise ModelError("Model files changed during export. Retry the operation.")
                # Hard-link publication is atomic and refuses an existing destination.
                os.link(temporary, destination)
                return destination
            except Exception as error:
                raise ModelError(f"Could not export {model_id}: {error}") from error
            finally:
                if temporary is not None:
                    Path(temporary).unlink(missing_ok=True)
