"""Read RF-DETR checkpoint files and check them against a configured class map.

Shared by the run-artifact writer (SCRUM-59), which verifies saved checkpoints
before reporting them, and the inference adapter (SCRUM-60), which refuses an
incompatible checkpoint before any prediction is returned.

RF-DETR 1.3.0 uses COCO category IDs directly as classification-logit
indices and records the training class names, in category order, in the
checkpoint's ``args.class_names``. A checkpoint is therefore compatible with
a class map only when its head has a logit for every configured category ID
and its recorded names equal the configured names in ascending ID order.

Checkpoints are pickles (``torch.load(weights_only=False)``, as RF-DETR itself
loads them). Only open checkpoints from runs you trust.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.inference import validate_class_map


class CheckpointError(ValueError):
    """A checkpoint file is missing, empty or cannot be read."""

    def __init__(self, path, message: str) -> None:
        self.path = Path(path)
        super().__init__(f"{message}: {self.path}")


class IncompatibleCheckpointError(CheckpointError):
    """The file is readable but its weights or class metadata do not fit."""


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    sha256: str
    size_bytes: int
    epoch: int | None
    logit_count: int
    class_names: tuple[str, ...] | None
    resolution: int | None
    state: Mapping = field(repr=False, compare=False)


def torch_loader(path: Path):
    """Load a checkpoint exactly as RF-DETR does (CPU, full pickle)."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(size) for size in tensor.shape)


def read_checkpoint(path, *, loader=None) -> CheckpointInfo:
    """Load and describe a checkpoint, failing with the path on any problem."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise CheckpointError(path, "Checkpoint file is missing")
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except OSError as error:
        raise CheckpointError(path, f"Checkpoint file cannot be read ({error})") from error
    if size == 0:
        raise CheckpointError(path, "Checkpoint file is empty")
    try:
        payload = (loader or torch_loader)(path)
    except Exception as error:  # torch raises many types for truncated pickles/zips.
        raise CheckpointError(
            path, f"Checkpoint is unreadable or incomplete ({type(error).__name__}: {error})"
        ) from error
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise IncompatibleCheckpointError(path, "Not an RF-DETR checkpoint: no 'model' weights")
    state = payload["model"]
    bias = state.get("class_embed.bias")
    if bias is None:
        raise IncompatibleCheckpointError(path, "Not an RF-DETR detector checkpoint: no class_embed.bias")
    try:
        shape = _shape(bias)
    except (AttributeError, TypeError, ValueError) as error:
        raise IncompatibleCheckpointError(path, "class_embed.bias is not a tensor") from error
    if len(shape) != 1 or shape[0] < 1:
        raise IncompatibleCheckpointError(path, f"class_embed.bias has unexpected shape {shape}")

    args = payload.get("args")
    names = getattr(args, "class_names", None)
    if names is not None:
        if not isinstance(names, (list, tuple)) or not all(isinstance(n, str) for n in names):
            raise IncompatibleCheckpointError(path, "Checkpoint class_names is not a list of strings")
        names = tuple(names)
    resolution = getattr(args, "resolution", None)
    epoch = payload.get("epoch")
    return CheckpointInfo(
        path=path, sha256=digest, size_bytes=size,
        epoch=epoch if type(epoch) is int else None,
        logit_count=shape[0], class_names=names,
        resolution=resolution if type(resolution) is int else None,
        state=state,
    )


def check_class_map(info: CheckpointInfo, class_map: Mapping[int, str]) -> None:
    """Raise IncompatibleCheckpointError unless the checkpoint fits the class map."""
    try:
        validate_class_map(class_map)
    except ValueError as error:
        raise IncompatibleCheckpointError(info.path, f"Invalid configured class map ({error})") from error
    highest = max(class_map)
    if info.logit_count <= highest:
        raise IncompatibleCheckpointError(
            info.path,
            f"Classification head has {info.logit_count} outputs (indices 0-{info.logit_count - 1}) "
            f"but the class map uses category ID {highest}",
        )
    if info.class_names is None:
        raise IncompatibleCheckpointError(
            info.path, "Checkpoint records no class names, so its class map cannot be confirmed"
        )
    expected = [class_map[key] for key in sorted(class_map)]
    if list(info.class_names) != expected:
        raise IncompatibleCheckpointError(
            info.path,
            f"Checkpoint classes {list(info.class_names)} do not match configured classes "
            f"{expected} (ascending category ID order)",
        )
