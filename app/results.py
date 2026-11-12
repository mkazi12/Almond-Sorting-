"""Save successful desktop analyses as portable JSON and detection CSV files.

An analysis references its original image and SHA-256 rather than embedding
pixels. Loading, saving and exporting verify that image before using results.
An empty detection list is a successful analysis; its CSV contains headers.
"""

import csv
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

from PIL import Image

from app.inference import Detection, InferenceResult, validate_class_map, validate_result


class ResultFileError(ValueError):
    """A saved analysis, its source image or an output file is unusable."""


@dataclass(frozen=True)
class AnalysisRecord:
    image_path: Path
    image_sha256: str
    result: InferenceResult
    class_names: Mapping[int, str]
    created_at: datetime
    model_label: str | None = None

    def __post_init__(self):
        try:
            path = Path(self.image_path).expanduser().resolve()
            if not isinstance(self.image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.image_sha256):
                raise ValueError("Image SHA-256 must contain 64 lowercase hexadecimal characters.")
            if not isinstance(self.result, InferenceResult):
                raise ValueError("A saved analysis requires a successful InferenceResult.")
            validate_class_map(self.class_names)
            validate_result(
                self.result, image_width=self.result.image_width,
                image_height=self.result.image_height,
                confidence_floor=self.result.confidence_floor, classes=self.class_names,
            )
            if not isinstance(self.created_at, datetime) or self.created_at.utcoffset() is None:
                raise ValueError("Analysis time must include a timezone.")
            if self.model_label is not None and (
                not isinstance(self.model_label, str) or not self.model_label.strip()
            ):
                raise ValueError("Model label must be a nonblank string or null when unknown.")
            object.__setattr__(self, "image_path", path)
            object.__setattr__(self, "class_names", MappingProxyType(dict(self.class_names)))
            object.__setattr__(self, "created_at", self.created_at.astimezone(timezone.utc))
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise ResultFileError(str(error)) from error


def _require_image_file(path: Path) -> None:
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ResultFileError("Source image must have a JPG/JPEG/PNG filename.")
    if not path.is_file():
        raise ResultFileError("Source image must be an existing regular file.")


def _open_image(path: Path):
    """Open a regular source; nonblocking open also rejects racing FIFO swaps."""
    _require_image_file(path)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ResultFileError("Source image must be a regular file.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_image(path) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_image_sha256(image_path) -> str:
    """Capture source bytes before decoding/prediction to detect later changes."""
    try:
        path = Path(image_path).expanduser().resolve()
        _require_image_file(path)
        return _sha256(path)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ResultFileError(f"Cannot read source image {image_path}: {error}. Choose a readable JPG/PNG image.") from error


def _source_fingerprint(path: Path) -> tuple[str, tuple[int, int]]:
    try:
        _require_image_file(path)
        before = _sha256(path)
        # Decode an already verified descriptor, avoiding a second path-open
        # race if the source gets replaced with a pipe or device.
        with _open_image(path) as handle, Image.open(handle) as image:
            if image.format not in {"JPEG", "PNG"}:
                raise ResultFileError("Source file contents must be JPEG or PNG.")
            image.load()
            dimensions = image.size
        if _sha256(path) != before:
            raise ResultFileError("The source image changed while it was being read.")
        return before, dimensions
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise ResultFileError(f"Cannot verify source image {path}: {error}") from error


def create_analysis(image_path, result, class_names, *, model_label=None, created_at=None,
                    expected_image_sha256=None) -> AnalysisRecord:
    """Capture successful predictions, checking a pre-prediction source digest."""
    try:
        path = Path(image_path).expanduser().resolve()
        digest, dimensions = _source_fingerprint(path)
        if expected_image_sha256 is not None:
            if not isinstance(expected_image_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_image_sha256):
                raise ResultFileError("Expected image SHA-256 must contain 64 lowercase hexadecimal characters.")
            if digest != expected_image_sha256:
                raise ResultFileError("The source image changed during analysis; its SHA-256 no longer matches the predicted image.")
        if not isinstance(result, InferenceResult):
            raise ResultFileError("A saved analysis requires a successful InferenceResult.")
        if dimensions != (result.image_width, result.image_height):
            raise ResultFileError("Prediction dimensions do not match the source image.")
        timestamp = datetime.now(timezone.utc) if created_at is None else created_at
        return AnalysisRecord(path, digest, result, class_names, timestamp, model_label)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ResultFileError(str(error)) from error


def _verify_source(analysis: AnalysisRecord) -> None:
    if not isinstance(analysis, AnalysisRecord):
        raise ResultFileError("A validated AnalysisRecord is required.")
    digest, dimensions = _source_fingerprint(analysis.image_path)
    if digest != analysis.image_sha256:
        raise ResultFileError("The source image has changed; its SHA-256 no longer matches this analysis.")
    if dimensions != (analysis.result.image_width, analysis.result.image_height):
        raise ResultFileError("Saved dimensions do not match the source image.")


def _write(destination, text: str, analysis: AnalysisRecord, overwrite: bool) -> Path:
    temporary = None
    try:
        target = Path(destination).expanduser().absolute()
        if target.resolve() == analysis.image_path:
            raise ResultFileError("Choose an output file different from the source image.")
        if type(overwrite) is not bool:
            raise ResultFileError("overwrite must be True or False.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=target.parent, prefix=".dinos-result-",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # A hard link publishes the complete file without clobbering a
            # destination that another process created after a file dialog.
            os.link(temporary, target)
        return target
    except FileExistsError as error:
        raise ResultFileError(f"Output already exists: {destination}. Choose a new file or confirm replacement.") from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ResultFileError(f"Cannot save {destination}: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_analysis(destination, analysis: AnalysisRecord, *, overwrite=False) -> Path:
    """Atomically save a JSON analysis; replacement requires explicit consent."""
    _verify_source(analysis)
    try:
        target = Path(destination).expanduser().absolute()
        try:
            image_path = Path(os.path.relpath(analysis.image_path, target.parent.resolve())).as_posix()
        except ValueError:  # Different Windows drives have no relative path.
            image_path = str(analysis.image_path)
        result = analysis.result
        payload = {
            "schema_version": 1,
            "created_at": analysis.created_at.isoformat(),
            "model_label": analysis.model_label,
            "image": {"path": image_path, "sha256": analysis.image_sha256,
                      "width": result.image_width, "height": result.image_height},
            "classes": [{"category_id": key, "name": name}
                        for key, name in sorted(analysis.class_names.items())],
            "prediction": {
                "confidence_floor": result.confidence_floor,
                "detections": [{"category_id": item.category_id, "confidence": item.confidence,
                                "bbox_xyxy": list(item.bbox_xyxy)} for item in result.detections],
            },
        }
        return _write(target, json.dumps(payload, indent=2, allow_nan=False) + "\n", analysis, overwrite)
    except (OSError, TypeError, ValueError) as error:
        raise ResultFileError(str(error)) from error


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResultFileError(f"Repeated JSON field: {key}.")
        result[key] = value
    return result


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ResultFileError(f"{label} has missing or unsupported fields.")


def load_analysis(path) -> AnalysisRecord:
    """Load and validate JSON, then verify the referenced image's bytes."""
    try:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8-sig"), object_pairs_hook=_object)
        _keys(payload, ("schema_version", "created_at", "model_label", "image", "classes", "prediction"), "Analysis")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
            raise ResultFileError("Unsupported analysis schema_version; expected 1.")
        image, prediction = payload["image"], payload["prediction"]
        _keys(image, ("path", "sha256", "width", "height"), "Image")
        _keys(prediction, ("confidence_floor", "detections"), "Prediction")
        if not isinstance(image["path"], str) or not image["path"].strip() or "\x00" in image["path"]:
            raise ResultFileError("Saved image path must be a nonempty string.")
        if not isinstance(payload["classes"], list) or not isinstance(prediction["detections"], list):
            raise ResultFileError("Classes and detections must be arrays.")
        classes = {}
        for item in payload["classes"]:
            _keys(item, ("category_id", "name"), "Class")
            key = item["category_id"]
            if type(key) is not int or key in classes:
                raise ResultFileError("Saved class IDs must be unique integers.")
            classes[key] = item["name"]
        detections = []
        for item in prediction["detections"]:
            _keys(item, ("category_id", "confidence", "bbox_xyxy"), "Detection")
            detections.append(Detection(item["category_id"], item["confidence"], item["bbox_xyxy"]))
        result = InferenceResult(image["width"], image["height"], prediction["confidence_floor"], detections)
        timestamp = datetime.fromisoformat(payload["created_at"])
        analysis = AnalysisRecord(source.parent / image["path"], image["sha256"], result,
                                  classes, timestamp, payload["model_label"])
        _verify_source(analysis)
        return analysis
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ResultFileError(f"Cannot open analysis {path}: {error}") from error


def _csv_text(value: str) -> str:
    """Keep text literal in spreadsheets, preserving the original after a quote."""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


def export_csv(destination, analysis: AnalysisRecord, *, overwrite=False) -> Path:
    """Export detections, or headers only for a successful empty result.

    CSV text fields whose first non-whitespace character is =, +, - or @
    receive a leading apostrophe to prevent spreadsheet formula evaluation.
    Record and JSON text remain unchanged; numerical fields remain numerical.
    """
    _verify_source(analysis)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("created_at", "image_path", "image_sha256", "image_width", "image_height",
                     "model_label", "confidence_floor", "category_id", "class_name", "confidence",
                     "x_min", "y_min", "x_max", "y_max"))
    result = analysis.result
    for item in result.detections:
        writer.writerow((analysis.created_at.isoformat(), _csv_text(str(analysis.image_path)), analysis.image_sha256,
                         result.image_width, result.image_height, _csv_text(analysis.model_label or ""),
                         result.confidence_floor, item.category_id, _csv_text(analysis.class_names[item.category_id]),
                         item.confidence, *item.bbox_xyxy))
    return _write(destination, output.getvalue(), analysis, overwrite)
