"""Read saved model records without loading weights or changing active state."""

from dataclasses import dataclass
import json
from pathlib import Path
import re


DEFAULT_REGISTRY_ROOT = Path(__file__).resolve().parents[1] / "models"


@dataclass(frozen=True)
class SavedModel:
    model_id: str
    variant: str
    model_version: str
    class_map: dict[int, str]
    dataset_version: str
    checkpoint: Path
    metadata: Path
    validation_summary: dict | None
    training_provenance: dict | None = None

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint.is_file()


@dataclass(frozen=True)
class RegistrySnapshot:
    models: tuple[SavedModel, ...]
    errors: tuple[str, ...]


def _text(record, field):
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_model(path, *, record=None):
    if record is None:
        record = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    if not isinstance(record, dict):
        raise ValueError("record must be an object")
    if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
        raise ValueError("unsupported schema_version (expected 1)")
    model_id = _text(record, "model_id")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", model_id):
        raise ValueError("model_id must use 1–100 letters, digits, underscores or hyphens")
    if model_id != path.parent.name:
        raise ValueError("model_id must match its registry directory")
    classes = record.get("class_map")
    if not isinstance(classes, dict) or not classes:
        raise ValueError("class_map must be a nonempty object")
    for key, value in classes.items():
        if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError("class_map keys must be canonical nonnegative integer IDs")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("class_map names must be nonempty strings")
    relative = Path(_text(record, "checkpoint"))
    checkpoint = (path.parent / relative).resolve()
    if relative.is_absolute() or not checkpoint.is_relative_to(path.parent.resolve()):
        raise ValueError("checkpoint must stay inside the model directory")
    summary = record.get("validation_summary")
    # Invalid/missing validation must never be replaced by another model's metrics.
    if not isinstance(summary, dict) or not summary:
        summary = None
    else:
        try:
            json.dumps(summary, allow_nan=False)
        except ValueError:
            summary = None
    return SavedModel(
        model_id, _text(record, "variant"), _text(record, "model_version"),
        {int(key): value for key, value in classes.items()},
        _text(record, "dataset_version"), checkpoint, path.resolve(), summary,
        record.get("training_provenance") if isinstance(record.get("training_provenance"), dict) else None,
    )


def read_registry(root: str | Path = DEFAULT_REGISTRY_ROOT) -> RegistrySnapshot:
    """Read models/<model_id>/model.json afresh; isolate invalid records."""
    root = Path(root).resolve()
    models, errors = [], []
    try:
        if not root.exists():
            return RegistrySnapshot((), ())
        directories = sorted(root.iterdir())
        for directory in directories:
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            path = directory / "model.json"
            try:
                if not path.resolve().is_relative_to(root):
                    raise ValueError("metadata must stay inside the registry")
                models.append(_read_model(path))
            except (OSError, ValueError, TypeError, RecursionError) as error:
                errors.append(f"{directory.name}: {error}")
    except OSError as error:
        errors.append(f"Cannot read registry {root}: {error}")
    return RegistrySnapshot(tuple(models), tuple(errors))
