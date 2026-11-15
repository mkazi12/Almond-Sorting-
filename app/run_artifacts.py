"""Record one training run's identity, inputs, checkpoints and logs (SCRUM-59).

Every run directory receives ``run.json`` keyed by a single run ID. It holds the
resolved parameters, seed, dependency versions, dataset version and hashes,
log file hashes, and references to two checkpoints kept separately:

* ``latest`` -> ``checkpoint.pth``: RF-DETR's most recent completed save.
* ``best`` -> ``checkpoint_best_total.pth``: selected by validation mAP@[.50:.95].

RF-DETR 1.3.0 names validation metrics ``test_*`` in ``log.txt`` because its
evaluator is shared; the launcher runs with ``run_test=False`` and redirects
the upstream test loader to the validation split, so no test-set input is
used for selection. Checkpoint references are written only after each file
has been reloaded and checked. Any failure sets ``status`` to ``failed`` with
the operation and path, and leaves both references ``null``.
"""

import hashlib
import json
import os
import platform
import secrets
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from app.checkpoint import CheckpointError, check_class_map, read_checkpoint, sha256_file


SCHEMA_VERSION = 1
RUN_RECORD = "run.json"
RESOLVED_CONFIG = "resolved_config.json"
LATEST_CHECKPOINT = "checkpoint.pth"
BEST_CHECKPOINT = "checkpoint_best_total.pth"
METRICS_LOG = "log.txt"
CONSOLE_LOG = "console.log"
SPLITS = ("train", "valid")
SELECTION = {
    "metric": "validation mAP@[.50:.95]",
    "split": "valid",
    "log_keys": {"regular": "test_coco_eval_bbox", "ema": "ema_test_coco_eval_bbox"},
    "rule": "highest value; EMA only if strictly higher than regular; earliest epoch on ties",
}


class RunArtifactError(RuntimeError):
    """A run file or checkpoint could not be written, read or verified."""

    def __init__(self, operation: str, path, detail) -> None:
        self.operation = operation
        self.path = Path(path)
        self.detail = str(detail)
        super().__init__(f"{operation} failed for {self.path}: {self.detail}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(now: datetime | None = None) -> str:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return f"{now:%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"


def write_json_atomic(path, document, *, operation: str = "write", overwrite: bool = True) -> Path:
    """Publish complete JSON: replace by default, or atomically require a new file.

    No-overwrite publication uses a same-directory hard link and fails closed
    on filesystems without hard-link support.
    """
    path = Path(path)
    temporary = None
    try:
        text = json.dumps(document, indent=2, allow_nan=False) + "\n"
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}-", suffix=".tmp",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
            temporary = None
        else:
            os.link(temporary, path)
        return path
    except (OSError, TypeError, ValueError) as error:
        raise RunArtifactError(operation, path, error) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _hash(path, operation: str) -> str:
    try:
        return sha256_file(path)
    except OSError as error:
        raise RunArtifactError(operation, path, error) from error


def dataset_fingerprint(dataset_dir) -> dict:
    """Hash the exported COCO files, provenance and every train/valid image."""
    root = Path(dataset_dir)
    files, image_digests, image_counts, versions, class_map = {}, {}, {}, set(), None
    for split in SPLITS:
        annotation = root / split / "_annotations.coco.json"
        files[f"{split}/{annotation.name}"] = _hash(annotation, "hash dataset")
        try:
            document = json.loads(annotation.read_text(encoding="utf-8"))
            info = document.get("info", {})
            versions.add((info.get("dataset_version"), info.get("dataset_content_hash")))
            class_map = {item["id"]: item["name"] for item in document["categories"]}
            names = sorted(item["file_name"] for item in document["images"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RunArtifactError("read dataset", annotation, error) from error
        combined = "".join(f"{name}\0{_hash(root / split / name, 'hash dataset')}\n" for name in names)
        image_digests[split] = sha256_text(combined)
        image_counts[split] = len(names)
    if len(versions) != 1:
        raise RunArtifactError("read dataset", root, "train and valid record different dataset versions")
    provenance = root / "dataset_provenance.json"
    if provenance.is_file():
        files[provenance.name] = _hash(provenance, "hash dataset")
    version, content_hash = versions.pop()
    return {
        "path": str(root),
        "dataset_version": version,
        "dataset_content_hash": content_hash,
        "class_map": {str(key): name for key, name in sorted(class_map.items())},
        "file_sha256": files,
        "image_set_sha256": image_digests,
        "image_counts": image_counts,
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def start_run(resolved: dict) -> dict:
    """Write the initial ``run.json`` (status ``running``) before fitting."""
    output = Path(resolved["output_dir"])
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": resolved["run_id"],
        "status": "running",
        "created_at": _now(),
        "completed_at": None,
        "variant": resolved["variant"],
        "device": resolved["device"],
        "seed": resolved["seed"],
        "parameters": resolved["training"],
        "model_kwargs": resolved.get("model_kwargs"),
        "train_kwargs": resolved.get("train_kwargs"),
        "dependencies": {
            "pinned": resolved.get("dependencies"),
            "installed": resolved.get("installed_dependencies"),
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "initial_checkpoint": {
            "path": resolved["initial_checkpoint"],
            "sha256": _hash(resolved["initial_checkpoint"], "hash initial checkpoint"),
        },
        "dataset": dataset_fingerprint(resolved["dataset_dir"]),
        "test_set_policy": resolved.get("test_loader_policy"),
        "files": {"resolved_config": RESOLVED_CONFIG, "metrics_log": METRICS_LOG,
                  "console_log": CONSOLE_LOG},
        "logs": None,
        "checkpoints": {"latest": None, "best": None},
        "failure": None,
    }
    write_json_atomic(output / RUN_RECORD, record, operation="write run record")
    return record


def _read_record(output: Path) -> dict:
    path = output / RUN_RECORD
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RunArtifactError("read run record", path, error) from error


def record_failure(output_dir, operation: str, path, error, *, run_id: str | None = None) -> None:
    """Mark the run failed. Never reports checkpoints from a failed run.

    If ``run.json`` itself cannot be written, the caller's exception (which
    carries the operation and path) remains the failure report.
    """
    output = Path(output_dir)
    try:
        record = _read_record(output)
    except RunArtifactError:
        record = {"schema_version": SCHEMA_VERSION, "run_id": run_id}
    record.update(
        status="failed", completed_at=_now(),
        checkpoints={"latest": None, "best": None},
        failure={"operation": operation, "path": str(path),
                 "error": f"{type(error).__name__}: {error}"},
    )
    try:
        write_json_atomic(output / RUN_RECORD, record, operation="write run record")
    except RunArtifactError as write_error:
        print(f"Warning: {write_error}", file=sys.stderr)


def read_metrics_log(path) -> list[dict]:
    path = Path(path)
    entries = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RunArtifactError("read training log", path, error) from error
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError as error:
            raise RunArtifactError("read training log", path, f"line {number}: {error}") from error
        if not isinstance(entry, dict) or type(entry.get("epoch")) is not int:
            raise RunArtifactError("read training log", path, f"line {number}: no integer epoch")
        entries.append(entry)
    if not entries:
        raise RunArtifactError("read training log", path, "no completed epochs were logged")
    return entries


def select_best(entries: list[dict], log_path) -> dict:
    """Reproduce RF-DETR 1.3.0's validation-only best-total choice from log.txt."""
    best = {}
    for source, key in SELECTION["log_keys"].items():
        for entry in entries:
            stats = entry.get(key)
            if not isinstance(stats, list) or not stats:
                continue
            value = stats[0]
            if not isinstance(value, (int, float)):
                continue
            if source not in best or value > best[source]["value"]:
                best[source] = {"source": source, "epoch": entry["epoch"], "value": float(value)}
    if "regular" not in best:
        raise RunArtifactError("select best checkpoint", log_path, "no validation mAP was logged")
    chosen = best["regular"]
    if "ema" in best and best["ema"]["value"] > chosen["value"]:
        chosen = best["ema"]
    return {**chosen, "metric": SELECTION["metric"], "split": SELECTION["split"],
            "log_key": SELECTION["log_keys"][chosen["source"]], "rule": SELECTION["rule"]}


def _verified(path: Path, operation: str, loader):
    try:
        return read_checkpoint(path, loader=loader)
    except CheckpointError as error:
        raise RunArtifactError(operation, path, error) from error


def finalize_run(output_dir, *, loader=None) -> dict:
    """Verify outputs after fitting and publish checkpoint references.

    The best checkpoint must reload with the run's class map, which is the
    same compatibility check the inference adapter applies.
    """
    output = Path(output_dir)
    record = _read_record(output)
    try:
        log_path = output / METRICS_LOG
        entries = read_metrics_log(log_path)
        last_epoch = max(entry["epoch"] for entry in entries)

        latest_path = output / LATEST_CHECKPOINT
        latest = _verified(latest_path, "verify latest checkpoint", loader)
        if latest.epoch != last_epoch:
            raise RunArtifactError(
                "verify latest checkpoint", latest_path,
                f"checkpoint epoch {latest.epoch} does not match last logged epoch {last_epoch}",
            )

        selection = select_best(entries, log_path)
        best_path = output / BEST_CHECKPOINT
        best = _verified(best_path, "verify best checkpoint", loader)
        class_map = {int(key): name for key, name in record["dataset"]["class_map"].items()}
        try:
            check_class_map(best, class_map)
        except CheckpointError as error:
            raise RunArtifactError("verify best checkpoint", best_path, error) from error

        logs = {METRICS_LOG: {"sha256": _hash(log_path, "hash training log"), "epochs": len(entries)}}
        console = output / CONSOLE_LOG
        if console.is_file():
            logs[CONSOLE_LOG] = {"sha256": _hash(console, "hash console log")}

        record.update(
            status="completed", completed_at=_now(), logs=logs, failure=None,
            checkpoints={
                "latest": {"path": LATEST_CHECKPOINT, "sha256": latest.sha256,
                           "size_bytes": latest.size_bytes, "epoch": latest.epoch},
                "best": {"path": BEST_CHECKPOINT, "sha256": best.sha256,
                         "size_bytes": best.size_bytes, "selected_by": selection,
                         "class_names": list(best.class_names)},
            },
        )
        write_json_atomic(output / RUN_RECORD, record, operation="write run record")
        return record
    except RunArtifactError as error:
        record_failure(output, error.operation, error.path, error, run_id=record.get("run_id"))
        raise


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False


@contextmanager
def capture_console(path):
    """Copy stdout/stderr to the run's console log while still showing them."""
    path = Path(path)
    try:
        handle = open(path, "a", encoding="utf-8")
    except OSError as error:
        raise RunArtifactError("open console log", path, error) from error
    with handle, redirect_stdout(_Tee(sys.stdout, handle)), redirect_stderr(_Tee(sys.stderr, handle)):
        yield
