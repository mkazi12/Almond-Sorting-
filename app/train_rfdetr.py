"""Launch pinned RF-DETR training: python -m app.train_rfdetr CONFIG.json."""

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
from unittest.mock import patch

from app.run_artifacts import (
    CONSOLE_LOG, RESOLVED_CONFIG, RUN_RECORD, RunArtifactError, capture_console,
    finalize_run, new_run_id, record_failure, start_run, write_json_atomic,
)

VARIANTS = {name: f"RFDETR{name.title()}" for name in ("nano", "small", "medium", "base", "large")}
PIN_FILE = Path(__file__).resolve().parents[1] / "requirements-training.txt"
DEFAULTS = {
    "epochs": 100, "batch_size": 4, "lr": 0.0001,
    "lr_encoder": 0.00015, "grad_accum_steps": 4, "num_workers": 0,
    "weight_decay": 0.0001, "checkpoint_interval": 10,
    "use_ema": True, "early_stopping": False,
    "early_stopping_patience": 10, "early_stopping_min_delta": 0.001,
    "multi_scale": True, "expanded_scales": True, "warmup_epochs": 0.0,
}


class TrainingConfigError(ValueError):
    """A specific setting prevents training from starting."""


def fail(setting, detail):
    raise TrainingConfigError(f"{setting}: {detail}")


def pinned_versions():
    return dict(line.strip().split("==") for line in PIN_FILE.read_text().splitlines()
                if line.strip() and not line.startswith("#"))


def check_dependencies():
    installed = {}
    for package, expected in pinned_versions().items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            fail(f"dependencies.{package}", "missing; install requirements-training.txt")
        # PyTorch wheels may carry a CUDA/CPU local build suffix.
        if actual.split("+")[0] != expected:
            fail(f"dependencies.{package}", f"expected {expected}, installed {actual}")
        installed[package] = actual
    return installed


def load_config(config_path):
    """Return a JSON-serializable, typed configuration with absolute paths."""
    path = Path(config_path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        fail("config", str(error))
    if not isinstance(raw, dict):
        fail("config", "expected a JSON object")
    required = {"variant", "dataset_dir", "initial_checkpoint", "device", "seed", "output_dir"}
    for field in sorted(required - raw.keys()):
        fail(field, "required setting is missing")
    for field in sorted(raw.keys() - required - {"training"}):
        fail(field, "unsupported setting")
    if not isinstance(raw["variant"], str) or raw["variant"] not in VARIANTS:
        fail("variant", f"supported variants: {', '.join(VARIANTS)}")
    if raw["device"] not in ("cpu", "cuda", "mps"):
        fail("device", "supported values: cpu, cuda, mps")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**32:
        fail("seed", "expected an integer between 0 and 4294967295")
    resolved = dict(raw)
    for field in ("dataset_dir", "initial_checkpoint", "output_dir"):
        value = raw[field]
        if not isinstance(value, str) or not value.strip():
            fail(field, "expected a nonempty path")
        resolved[field] = str((path.parent / value).resolve())
    if not Path(resolved["initial_checkpoint"]).is_file():
        fail("initial_checkpoint", f"file is missing: {resolved['initial_checkpoint']}")
    output = Path(resolved["output_dir"])
    if output.exists():
        fail("output_dir", "must be a new directory to avoid overwriting another run")
    supplied = raw.get("training", {})
    if not isinstance(supplied, dict):
        fail("training", "expected an object")
    for field in sorted(supplied.keys() - DEFAULTS.keys()):
        fail(f"training.{field}", "unsupported parameter")
    params = {**DEFAULTS, **supplied}
    for field, default in DEFAULTS.items():
        value = params[field]
        if type(default) is bool:
            if type(value) is not bool:
                fail(f"training.{field}", "expected a boolean")
        elif type(default) is int:
            minimum = 0 if field == "num_workers" else 1
            if type(value) is not int or value < minimum:
                fail(f"training.{field}", f"expected an integer >= {minimum}")
        else:
            minimum_inclusive = field in {"weight_decay", "warmup_epochs", "early_stopping_min_delta"}
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (value == 0 and not minimum_inclusive)):
                fail(f"training.{field}", "expected a finite " + ("nonnegative" if minimum_inclusive else "positive") + " number")
            params[field] = float(value)
    if params["warmup_epochs"] > params["epochs"]:
        fail("training.warmup_epochs", "cannot exceed epochs")
    resolved["training"] = params
    resolved["dependencies"] = pinned_versions()
    validate_dataset(Path(resolved["dataset_dir"]))
    return resolved


def validate_dataset(root):
    """Check the exporter contract before allocating a model."""
    categories = None
    ids = set()
    for split in ("train", "valid"):
        try:
            document = json.loads((root / split / "_annotations.coco.json").read_text())
            if not document["images"]:
                fail(f"dataset_dir.{split}", "split must contain images")
            current = document["categories"]
            if categories is not None and current != categories:
                fail("dataset_dir.categories", "train and valid category maps differ")
            categories = current
            for category in current:
                if "supercategory" not in category:
                    fail("dataset_dir.categories", "re-export dataset: supercategory field is required")
            for image in document["images"]:
                if image["id"] in ids:
                    fail("dataset_dir", f"duplicate image ID across splits: {image['id']}")
                ids.add(image["id"])
                image_path = (root / split / image["file_name"]).resolve()
                if not image_path.is_relative_to((root / split).resolve()) or not image_path.is_file():
                    fail(f"dataset_dir.{split}", f"missing or unsafe image: {image['file_name']}")
        except (OSError, ValueError, KeyError, TypeError) as error:
            if isinstance(error, TrainingConfigError):
                raise
            fail(f"dataset_dir.{split}", str(error))


def check_device(device, torch):
    if device == "cuda" and not torch.cuda.is_available():
        fail("device", "CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        fail("device", "MPS was requested but is unavailable")


def launch(config_path, *, artifact_writer=None, dry_run=False, checkpoint_loader=None):
    """Validate first; expose resolved inputs to the artifact writer before fit.

    ``artifact_writer(resolved)`` receives all explicit adapter parameters,
    fixed safety settings, dependency versions and canonical paths.
    A real run gets one run ID and a ``run.json`` record (see
    ``app.run_artifacts``); ``checkpoint_loader`` overrides ``torch.load``
    when verifying saved checkpoints.
    """
    resolved = load_config(config_path)
    resolved["installed_dependencies"] = check_dependencies()
    torch = importlib.import_module("torch")
    check_device(resolved["device"], torch)
    train_kwargs = {
        **resolved["training"], "dataset_dir": resolved["dataset_dir"],
        "output_dir": resolved["output_dir"], "seed": resolved["seed"],
        "device": resolved["device"], "dataset_file": "roboflow",
        "run_test": False, "tensorboard": False, "wandb": False, "do_benchmark": False,
    }
    model_kwargs = {"pretrain_weights": resolved["initial_checkpoint"], "device": resolved["device"]}
    resolved["model_kwargs"] = model_kwargs
    resolved["train_kwargs"] = train_kwargs
    resolved["test_loader_policy"] = "validation-only; frozen test never opened"
    if dry_run:
        return resolved
    random.seed(resolved["seed"])
    importlib.import_module("numpy").random.seed(resolved["seed"])
    torch.manual_seed(resolved["seed"])
    rfdetr = importlib.import_module("rfdetr")
    model = getattr(rfdetr, VARIANTS[resolved["variant"]])(**model_kwargs)
    output = Path(resolved["output_dir"])
    resolved["run_id"] = new_run_id()
    resolved["run_record"] = str(output / RUN_RECORD)
    output.mkdir(parents=True, exist_ok=False)
    try:
        write_json_atomic(output / RESOLVED_CONFIG, resolved, operation="write resolved config")
        start_run(resolved)
    except RunArtifactError as error:
        record_failure(output, error.operation, error.path, error, run_id=resolved["run_id"])
        raise
    if artifact_writer is not None:
        artifact_writer(resolved)
    # RF-DETR 1.3.0 constructs a test dataset even with run_test=False.
    # Scope this compatibility shim to this single-process training invocation.
    main = importlib.import_module("rfdetr.main")
    original = main.build_dataset

    def validation_only(image_set, **kwargs):
        return original(image_set="val" if image_set == "test" else image_set, **kwargs)

    try:
        with patch.object(main, "build_dataset", validation_only), capture_console(output / CONSOLE_LOG):
            model.train(**train_kwargs)
    except BaseException as error:  # includes Ctrl+C: never leave "running"
        record_failure(output, "train", output, error, run_id=resolved["run_id"])
        raise
    finalize_run(output, loader=checkpoint_loader)
    return resolved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate dependencies, device and inputs without training")
    args = parser.parse_args(argv)
    try:
        resolved = launch(args.config, dry_run=args.dry_run)
    except (TrainingConfigError, RunArtifactError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(resolved, indent=2))
    if not args.dry_run:
        print(f"Run {resolved['run_id']} completed; record: {resolved['run_record']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
