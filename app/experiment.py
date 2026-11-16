"""Save and relaunch RF-DETR experiments derived from a completed run (SCRUM-68).

Save an experiment that changes some parameters of a previous run::

    python -m app.experiment save runs/almond-001 experiments/lower-lr.json \\
        --output-dir runs/almond-002 --set lr=0.00005 --set epochs=20

Launch it through the normal training runner (``--dry-run`` validates only)::

    python -m app.experiment launch experiments/lower-lr.json

The experiment file records the parent run ID, the dataset version and hashes
the parent trained on, the starting checkpoint and its SHA-256, the changed
parameters and the resulting parameters. Paths are relative to the file.

Loading re-checks every reference before a model is built, then validates the
result with ``train_rfdetr.resolve_config``, the same check as a training
configuration file. Launching creates a new run ID in a new directory, so the
parent run's files are only read.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.checkpoint import sha256_file
from app.run_artifacts import (
    RESOLVED_CONFIG, RUN_RECORD, SPLITS, RunArtifactError, dataset_fingerprint, write_json_atomic,
)
from app.train_rfdetr import DEFAULTS, TrainingConfigError, launch_resolved, resolve_config


# best/latest come from the parent's verified checkpoints; initial is its starting weights.
CHECKPOINT_CHOICES = ("best", "latest", "initial")
# The variant is fixed by the starting checkpoint's architecture, so it cannot change.
CHANGEABLE = frozenset((*DEFAULTS, "device", "seed"))
DATASET_IDENTITY = ("dataset_version", "dataset_content_hash", "class_map",
                    "file_sha256", "image_set_sha256", "image_counts")


class ExperimentError(TrainingConfigError):
    """An experiment reference or setting prevents saving or launching."""


def fail(setting, detail):
    raise ExperimentError(f"{setting}: {detail}")


def _relative(path, base: Path) -> str:
    try:
        return Path(os.path.relpath(Path(path).resolve(), base)).as_posix()
    except ValueError:  # Different Windows drives have no relative path.
        return str(Path(path).resolve())


def _read_json(path: Path, setting: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(setting, f"file is missing: {path}")
    except (OSError, ValueError) as error:
        fail(setting, f"cannot read {path}: {error}")


def _hash(path: Path, setting: str) -> str:
    if not path.is_file():
        fail(setting, f"file is missing: {path}")
    return sha256_file(path)


def read_parent(parent_dir) -> tuple[Path, dict, dict]:
    """Return a completed parent run's directory, ``run.json`` and resolved configuration."""
    directory = Path(parent_dir).expanduser().resolve()
    if not directory.is_dir():
        fail("parent_run", f"run directory is missing: {directory}")
    record = _read_json(directory / RUN_RECORD, "parent_run")
    config = _read_json(directory / RESOLVED_CONFIG, "parent_run")
    if not record.get("run_id") or config.get("run_id") != record["run_id"]:
        fail("parent_run", f"run.json and {RESOLVED_CONFIG} do not identify one run: {directory}")
    if record.get("status") != "completed":
        fail("parent_run", f"run {record['run_id']} is {record.get('status')!r}; "
                           "only completed runs can be parents")
    return directory, record, config


def _check_changes(changes: dict) -> dict:
    if "variant" in changes:
        fail("changes.variant", "cannot change; the initial checkpoint fixes the model variant")
    for key in sorted(changes.keys() - CHANGEABLE):
        fail(f"changes.{key}", "unsupported parameter")
    return changes


def _apply(config: dict, changes: dict) -> dict:
    """The parent's variant, device, seed and training values with ``changes`` applied."""
    resolved = {key: config[key] for key in ("variant", "device", "seed")}
    resolved["training"] = dict(config["training"])
    for key, value in changes.items():
        if key in resolved:
            resolved[key] = value
        else:
            resolved["training"][key] = value
    return resolved


def _parent_checkpoint(directory: Path, record: dict, choice: str) -> Path:
    """Locate the chosen checkpoint and confirm it is still the file the parent recorded."""
    if choice not in CHECKPOINT_CHOICES:
        fail("initial_checkpoint", f"choose one of {', '.join(CHECKPOINT_CHOICES)}")
    if choice == "initial":
        reference = record["initial_checkpoint"]
        path = Path(reference["path"])
    else:
        reference = record["checkpoints"][choice]
        if not reference:
            fail("initial_checkpoint", f"run {record['run_id']} records no {choice} checkpoint")
        path = directory / reference["path"]
    if _hash(path, "initial_checkpoint") != reference["sha256"]:
        fail("initial_checkpoint", f"{path} no longer matches the SHA-256 recorded by run {record['run_id']}")
    return path


def save_experiment(parent_run_dir, experiment_path, *, output_dir, changes=None,
                    checkpoint: str = "best") -> dict:
    """Validate and write a new experiment file; return its contents."""
    target = Path(experiment_path).expanduser().resolve()
    if target.exists():
        fail("experiment", f"file already exists: {target}")
    base = target.parent
    directory, record, config = read_parent(parent_run_dir)
    changes = _check_changes(dict(changes or {}))
    initial = _parent_checkpoint(directory, record, checkpoint)
    document = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_run": {
            "run_id": record["run_id"],
            "path": _relative(directory, base),
            # Pins every parent parameter: any later edit to the parent config is detected.
            "resolved_config_sha256": _hash(directory / RESOLVED_CONFIG, "parent_run"),
        },
        "dataset": {"path": _relative(record["dataset"]["path"], base),
                    **{key: record["dataset"][key] for key in DATASET_IDENTITY}},
        "initial_checkpoint": {"source": checkpoint, "path": _relative(initial, base),
                               "sha256": _hash(initial, "initial_checkpoint")},
        "output_dir": _relative(Path(output_dir).expanduser(), base),
        "changes": changes,
        "resolved": _apply(config, changes),
    }
    resolve_experiment(document, base)  # Never save an experiment that could not launch.
    base.mkdir(parents=True, exist_ok=True)
    try:
        write_json_atomic(target, document, operation="write experiment", overwrite=False)
    except RunArtifactError as error:
        fail("experiment", str(error))
    return document


def resolve_experiment(document: dict, base_dir) -> dict:
    """Check every reference, then return the normally validated training configuration."""
    base = Path(base_dir).resolve()
    try:
        if document["schema_version"] != 1:
            fail("schema_version", "expected 1")
        parent = document["parent_run"]
        directory, record, config = read_parent(base / parent["path"])
        if record["run_id"] != parent["run_id"]:
            fail("parent_run.run_id", f"expected {parent['run_id']}, but {directory} holds run {record['run_id']}")
        if _hash(directory / RESOLVED_CONFIG, "parent_run") != parent["resolved_config_sha256"]:
            fail("parent_run", f"{RESOLVED_CONFIG} of run {record['run_id']} changed since saving")
        changes = _check_changes(document["changes"])
        resolved = document["resolved"]
        if resolved != _apply(config, changes):
            fail("resolved", "does not equal the parent parameters with changes applied; save again")

        checkpoint = document["initial_checkpoint"]
        initial = (base / checkpoint["path"]).resolve()
        if _hash(initial, "initial_checkpoint") != checkpoint["sha256"]:
            fail("initial_checkpoint", f"{initial} does not match its saved SHA-256")

        dataset = document["dataset"]
        dataset_dir = (base / dataset["path"]).resolve()
        if not dataset_dir.is_dir():
            fail("dataset", f"directory is missing: {dataset_dir}")
        if (dataset_dir / "test").exists():
            fail("dataset", f"{dataset_dir} contains a test split; experiments use only train/valid")
        try:
            current = dataset_fingerprint(dataset_dir)
        except RunArtifactError as error:
            fail("dataset", str(error))
        for key in DATASET_IDENTITY:
            expected = record["dataset"][key]
            if dataset[key] != expected:
                fail(f"dataset.{key}", f"differs from parent run {record['run_id']}")
            if current[key] != expected:
                changed = (sorted(name for name in expected.keys() | current[key].keys()
                                  if expected.get(name) != current[key].get(name))
                           if isinstance(expected, dict) else [f"{expected} -> {current[key]}"])
                fail(f"dataset.{key}",
                     f"changed since parent run {record['run_id']}: {', '.join(changed)}")

        output = (base / document["output_dir"]).resolve()
        if output.is_relative_to(directory):
            fail("output_dir", "must be outside the parent run directory")
        raw = {**{key: resolved[key] for key in ("variant", "device", "seed")},
               "dataset_dir": str(dataset_dir), "initial_checkpoint": str(initial),
               "output_dir": str(output), "training": resolved["training"]}
    except (KeyError, TypeError, AttributeError) as error:
        fail("experiment", f"malformed or missing field {error}")
    config = resolve_config(raw, base)  # The normal training-configuration validation.
    config["lineage"] = {
        "parent_run_id": record["run_id"],
        "parent_run_dir": str(directory),
        "initial_checkpoint_source": checkpoint["source"],
        "initial_checkpoint_sha256": checkpoint["sha256"],
        "changes": changes,
        "splits": list(SPLITS),
    }
    return config


def load_experiment(experiment_path) -> dict:
    """Read an experiment file and return its validated training configuration."""
    path = Path(experiment_path).expanduser().resolve()
    config = resolve_experiment(_read_json(path, "experiment"), path.parent)
    config["lineage"]["experiment_path"] = str(path)
    config["lineage"]["experiment_sha256"] = _hash(path, "experiment")
    return config


def launch_experiment(experiment_path, *, artifact_writer=None, dry_run=False,
                      checkpoint_loader=None) -> dict:
    """Launch a saved experiment through the normal training runner."""
    return launch_resolved(load_experiment(experiment_path), artifact_writer=artifact_writer,
                           dry_run=dry_run, checkpoint_loader=checkpoint_loader)


def _setting(text: str) -> tuple[str, object]:
    """Parse ``KEY=VALUE``; VALUE is JSON (``20``, ``0.001``, ``true``) or else text."""
    key, separator, value = text.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    try:
        return key.strip(), json.loads(value)
    except ValueError:
        return key.strip(), value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    save = commands.add_parser("save", help="save an experiment derived from a completed run")
    save.add_argument("parent_run", type=Path, help="parent run directory containing run.json")
    save.add_argument("experiment", type=Path, help="new experiment JSON file")
    save.add_argument("--output-dir", required=True, type=Path, help="new directory for the experiment run")
    save.add_argument("--set", dest="changes", action="append", type=_setting, default=[],
                      metavar="KEY=VALUE", help="changed parameter (repeatable)")
    save.add_argument("--checkpoint", default="best", choices=CHECKPOINT_CHOICES,
                      help="starting weights from the parent run (default: best)")
    launch = commands.add_parser("launch", help="validate and launch a saved experiment")
    launch.add_argument("experiment", type=Path)
    launch.add_argument("--dry-run", action="store_true",
                        help="validate references, dependencies and device without training")
    args = parser.parse_args(argv)
    try:
        if args.command == "save":
            document = save_experiment(args.parent_run, args.experiment, output_dir=args.output_dir,
                                       changes=dict(args.changes), checkpoint=args.checkpoint)
            print(json.dumps(document, indent=2))
            return 0
        resolved = launch_experiment(args.experiment, dry_run=args.dry_run)
    except (TrainingConfigError, RunArtifactError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(resolved, indent=2))
    if not args.dry_run:
        print(f"Run {resolved['run_id']} (parent {resolved['lineage']['parent_run_id']}) completed; "
              f"record: {resolved['run_record']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
