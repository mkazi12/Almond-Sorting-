"""Strict CPU loader for the pinned RF-DETR 1.3.0 state-dict package format."""

from argparse import Namespace
from importlib.metadata import version
import math
from types import SimpleNamespace

from app.inference import Detection, InferenceError, InferenceResult, RGBImage, validate_result
from app.model_workflow import ModelError


class RFDETRDetector:
    def __init__(self, backend, output_class_map, class_map):
        self.backend = backend
        self.output_class_map = tuple(output_class_map)
        self.class_map = dict(class_map)

    def predict(self, image: RGBImage, *, confidence_floor: float):
        from PIL import Image
        pixels = Image.frombytes("RGB", (image.width, image.height), image.pixels)
        prediction = self.backend.predict(pixels, threshold=confidence_floor)
        detections = []
        for index, score, box in zip(prediction.class_id, prediction.confidence, prediction.xyxy, strict=True):
            index = int(index)
            if not 0 <= index < len(self.output_class_map):
                raise InferenceError(f"Unexpected classifier output: {index}")
            category = self.output_class_map[index]
            if category is None:
                continue
            left, top, right, bottom = map(float, box)
            if not all(math.isfinite(value) for value in (left, top, right, bottom)):
                raise InferenceError("Model returned a non-finite bounding box.")
            clipped = (max(0., min(image.width, left)), max(0., min(image.height, top)),
                       max(0., min(image.width, right)), max(0., min(image.height, bottom)))
            if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
                continue
            detections.append(Detection(category, float(score), clipped))
        result = InferenceResult(image.width, image.height, confidence_floor, tuple(detections))
        validate_result(result, image_width=image.width, image_height=image.height,
                        confidence_floor=confidence_floor, classes=self.class_map)
        return result


def load_detector(model, record):
    try:
        if version("rfdetr") != "1.3.0" or version("torch").split("+")[0] != "2.8.0":
            raise ModelError("Install the pinned RF-DETR 1.3.0 / torch 2.8.0 runtime from requirements-training.txt.")
        import torch
        import rfdetr
        from rfdetr.main import populate_args, build_model, PostProcess

        # Training checkpoints include argparse.Namespace. Allow only this passive
        # metadata container, never unrestricted pickle or package-supplied globals.
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.load(model.checkpoint, map_location="cpu", weights_only=True)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict) or not state or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items()
        ):
            raise ModelError("Checkpoint must contain a 'model' tensor state dictionary.")
        if any(not torch.isfinite(value).all().item() for value in state.values()):
            raise ModelError("Checkpoint contains non-finite tensor values.")
        outputs = record["output_class_map"]
        bias = state.get("class_embed.bias")
        if bias is None or tuple(bias.shape) != (len(outputs),):
            raise ModelError("Classifier shape does not match output_class_map.")

        variant_class = getattr(rfdetr, "RFDETR" + model.variant.title())

        class LocalWeightsModel(variant_class):
            def get_model(self, config):
                args = populate_args(**config.model_dump())
                # build_model uses this flag only to suppress backbone downloads.
                # We bypass Model's permissive/unrestricted pretrained loader.
                args.pretrain_weights = "local-state-dict"
                network = build_model(args)
                network.eval()
                return SimpleNamespace(model=network, device=torch.device("cpu"),
                                       resolution=config.resolution,
                                       postprocess=PostProcess(num_select=args.num_select))

        backend = LocalWeightsModel(pretrain_weights=None, device="cpu", num_classes=len(outputs) - 1)
        backend.model.model.load_state_dict(state, strict=True)
        # Do not retain a second copy of weights or training optimizer state in
        # the loaded detector's class closure while inference is running.
        del checkpoint, state
        detector = RFDETRDetector(backend, outputs, model.class_map)
        # A successful load includes a real forward pass before activation.
        detector.predict(RGBImage(32, 32, bytes(32 * 32 * 3)), confidence_floor=1.0)
        return detector
    except ModelError:
        raise
    except ImportError as error:
        raise ModelError("RF-DETR runtime is unavailable; install requirements-models.txt in the desktop environment.") from error
    except Exception as error:
        raise ModelError(f"Corrupt or incompatible RF-DETR checkpoint: {error}") from error
