"""RF-DETR checkpoint adapter implementing ``app.inference.Detector`` (SCRUM-60).

Load a checkpoint and class map, refuse anything incompatible before the
first prediction, then return original-image pixel boxes with the requested
confidence floor recorded::

    detector = load_detector("runs/almond-001/checkpoint_best_total.pth",
                             {1: "almond", 2: "foreign_material"}, variant="nano")
    result = detector.predict(rgb_image, confidence_floor=0.5)

or from a configuration file (see ``configs/inference.example.json``)::

    python -m app.rfdetr_adapter configs/inference.example.json IMAGE.jpg

Requires the RF-DETR environment from ``requirements-training.txt``; RF-DETR
and torch are imported only when a real model is loaded.

Preprocessing: RF-DETR 1.3.0 ``predict`` scales pixels to [0, 1], normalizes
with ImageNet statistics and resizes to a square ``resolution`` x
``resolution`` input, which changes the aspect ratio. Its post-processing
multiplies normalized boxes by the size of the image it was given. The
adapter therefore always passes the full original image (never a resized
copy) and treats returned boxes as original-image pixels, then clips them to
the image bounds. Detection category IDs are COCO IDs, which RF-DETR uses
directly as logit indices; outputs at indices that are not configured
category IDs (such as index 0) were never trained and are dropped.
"""

import argparse
import importlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from app.checkpoint import (
    CheckpointError, CheckpointInfo, IncompatibleCheckpointError, check_class_map, read_checkpoint,
)
from app.inference import (
    Detection, InferenceError, InferenceResult, RGBImage, confidence_value, validate_class_map,
)


VARIANTS = {name: f"RFDETR{name.title()}" for name in ("nano", "small", "medium", "base", "large")}
DEVICES = ("cpu", "cuda", "mps")


def _check_device(device: str) -> None:
    torch = importlib.import_module("torch")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device: CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device: MPS was requested but is unavailable")


def _check_weights(model, info: CheckpointInfo, variant: str) -> None:
    """Every parameter of the constructed network must come from the checkpoint.

    RF-DETR loads pretrained weights with ``strict=False``, which silently
    skips missing tensors; compare names and shapes so a checkpoint from
    another variant or architecture fails here instead.
    """
    network = model.model.model
    missing, mismatched = [], []
    for name, tensor in network.state_dict().items():
        saved = info.state.get(name)
        if saved is None:
            missing.append(name)
        elif tuple(saved.shape) != tuple(tensor.shape):
            mismatched.append(f"{name} {tuple(saved.shape)} != {tuple(tensor.shape)}")
    if missing or mismatched:
        details = []
        if missing:
            details.append(f"{len(missing)} missing tensors (e.g. {missing[0]})")
        if mismatched:
            details.append(f"{len(mismatched)} shape mismatches (e.g. {mismatched[0]})")
        raise IncompatibleCheckpointError(
            info.path, f"Weights do not match RF-DETR {variant}: {'; '.join(details)}"
        )


class RFDETRDetector:
    """A loaded, compatibility-checked RF-DETR model."""

    def __init__(self, model, class_map: Mapping[int, str], *, checkpoint: CheckpointInfo,
                 variant: str, device: str) -> None:
        self._model = model
        self.class_map = MappingProxyType(dict(class_map))
        self.checkpoint = checkpoint
        self.variant = variant
        self.device = device

    @property
    def model_label(self) -> str:
        return f"rfdetr-{self.variant}:{self.checkpoint.path.name}@{self.checkpoint.sha256[:12]}"

    def predict(self, image: RGBImage, *, confidence_floor: float) -> InferenceResult:
        """Return detections at or above ``confidence_floor``.

        Raises ValueError for an invalid request and InferenceError when the
        model fails or returns unusable output. An empty result is a success.
        """
        floor = confidence_value(confidence_floor)
        if not isinstance(image, RGBImage):
            raise ValueError("predict() requires an RGBImage.")
        from PIL import Image

        pixels = Image.frombytes("RGB", (image.width, image.height), image.pixels)
        try:
            raw = self._model.predict(pixels, threshold=floor)
        except Exception as error:
            raise InferenceError(f"RF-DETR prediction failed: {type(error).__name__}: {error}") from error
        return self._convert(raw, image.width, image.height, floor)

    def _convert(self, raw, width: int, height: int, floor: float) -> InferenceResult:
        if isinstance(raw, list):
            if len(raw) != 1:
                raise InferenceError(f"Expected one prediction set for one image, got {len(raw)}.")
            raw = raw[0]
        try:
            boxes = [[float(value) for value in box] for box in raw.xyxy]
            scores = [float(value) for value in raw.confidence]
            labels = list(raw.class_id)
        except (AttributeError, TypeError, ValueError) as error:
            raise InferenceError(f"RF-DETR returned unexpected output: {error}") from error
        if not len(boxes) == len(scores) == len(labels):
            raise InferenceError("RF-DETR returned boxes, scores and classes of different lengths.")
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                raise InferenceError(f"RF-DETR returned a non-finite or malformed box: {box}")
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise InferenceError(f"RF-DETR returned an invalid confidence: {score}")
            try:
                category = int(label)
                if category != label:
                    raise ValueError
            except (TypeError, ValueError) as error:
                raise InferenceError(f"RF-DETR returned a non-integer class: {label!r}") from error
            if category not in self.class_map or score < floor:
                continue
            left, top, right, bottom = box
            left, right = max(0.0, left), min(float(width), right)
            top, bottom = max(0.0, top), min(float(height), bottom)
            if right <= left or bottom <= top:
                continue  # entirely outside the image or zero area after clipping
            detections.append(Detection(category, score, (left, top, right, bottom)))
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return InferenceResult(width, height, floor, tuple(detections))


def load_detector(checkpoint_path, class_map: Mapping[int, str], *, variant: str,
                  device: str = "cpu", loader=None, model_factory=None) -> RFDETRDetector:
    """Check the checkpoint and class map, then construct the RF-DETR model.

    Raises CheckpointError for a missing or unreadable file,
    IncompatibleCheckpointError for wrong weights or class metadata, and
    ValueError for an invalid variant, device or class map.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant: supported variants: {', '.join(VARIANTS)}")
    if device not in DEVICES:
        raise ValueError(f"device: supported values: {', '.join(DEVICES)}")
    validate_class_map(class_map)
    info = read_checkpoint(checkpoint_path, loader=loader)
    check_class_map(info, class_map)
    if model_factory is None:
        _check_device(device)
        model_factory = getattr(importlib.import_module("rfdetr"), VARIANTS[variant])
    try:
        model = model_factory(pretrain_weights=str(info.path), device=device)
    except Exception as error:  # RF-DETR raises RuntimeError on tensor size mismatches.
        raise IncompatibleCheckpointError(
            info.path, f"RF-DETR {variant} cannot load these weights ({type(error).__name__}: {error})"
        ) from error
    _check_weights(model, info, variant)
    return RFDETRDetector(model, class_map, checkpoint=info, variant=variant, device=device)


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: Path
    class_map: Mapping[int, str]
    variant: str
    device: str
    confidence_floor: float


def load_inference_config(path) -> InferenceConfig:
    """Read a JSON config; relative paths are relative to the config file."""
    from app.dataset.rfdetr_export import load_class_map

    path = Path(path).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise ValueError(f"config: {error}") from error
    fields = {"checkpoint", "class_map", "variant", "device", "confidence_floor"}
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"config: expected exactly these fields: {', '.join(sorted(fields))}")
    try:
        floor = confidence_value(raw["confidence_floor"])
    except ValueError as error:
        raise ValueError(f"confidence_floor: {error}") from error
    return InferenceConfig(
        checkpoint=(path.parent / raw["checkpoint"]).resolve(),
        class_map=load_class_map(path.parent / raw["class_map"]),
        variant=raw["variant"], device=raw["device"], confidence_floor=floor,
    )


def _read_rgb(path) -> RGBImage:
    from PIL import Image

    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
    return RGBImage(rgb.width, rgb.height, rgb.tobytes())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the RF-DETR adapter on one image.")
    parser.add_argument("config", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args(argv)
    try:
        config = load_inference_config(args.config)
        detector = load_detector(config.checkpoint, config.class_map,
                                 variant=config.variant, device=config.device)
        image = _read_rgb(args.image)
    except (CheckpointError, ValueError, OSError, ImportError) as error:
        parser.exit(2, f"Error: {error}\n")
    try:
        result = detector.predict(image, confidence_floor=config.confidence_floor)
    except InferenceError as error:
        parser.exit(3, f"Inference failed: {error}\n")
    print(json.dumps({
        "model_label": detector.model_label,
        "image": {"path": str(args.image), "width": result.image_width, "height": result.image_height},
        "confidence_floor": result.confidence_floor,
        "detections": [{"category_id": item.category_id, "class_name": config.class_map[item.category_id],
                        "confidence": item.confidence, "bbox_xyxy": list(item.bbox_xyxy)}
                       for item in result.detections],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
