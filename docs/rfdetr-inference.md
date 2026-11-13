# RF-DETR inference adapter

`app.rfdetr_adapter` implements the `app.inference.Detector` protocol for a
trained RF-DETR checkpoint. It is shared by the desktop display, saved
results and evaluation, so every consumer receives the same
`InferenceResult`: original-image pixel boxes plus the confidence floor that
was requested.

Use the RF-DETR environment from [rfdetr-training.md](rfdetr-training.md);
RF-DETR and torch are imported only when a real model is loaded.

## Configuration

Copy `configs/inference.example.json`. Paths are relative to the config file.

```json
{
  "checkpoint": "../runs/almond-001/checkpoint_best_total.pth",
  "class_map": "../manifests/class-map.json",
  "variant": "nano",
  "device": "cpu",
  "confidence_floor": 0.5
}
```

The class map uses the exporter's format (`{"1": "almond"}` or COCO
categories). Use the run's best checkpoint reference from `run.json`
(see [run-artifacts.md](run-artifacts.md)).

```powershell
.\.venv-training\Scripts\python.exe -m app.rfdetr_adapter configs/inference.example.json IMAGE.jpg
```

The command prints JSON with the model label, image size, confidence floor
and detections. It exits 2 for a configuration or checkpoint problem and 3
for an inference failure.

From Python:

```python
from app.rfdetr_adapter import load_detector

detector = load_detector(path, {1: "almond", 2: "foreign_material"}, variant="nano", device="cpu")
result = detector.predict(rgb_image, confidence_floor=0.5)
```

## Compatibility checks (before any prediction)

| Problem | Error |
| --- | --- |
| File missing, empty, truncated or unreadable | `CheckpointError` |
| No RF-DETR `model` weights or `class_embed.bias` | `IncompatibleCheckpointError` |
| Head has no logit for a configured category ID | `IncompatibleCheckpointError` |
| Checkpoint class names missing or not equal to configured names in ascending ID order | `IncompatibleCheckpointError` |
| RF-DETR cannot load the weights, or the variant's network has tensors missing from or shaped differently in the checkpoint | `IncompatibleCheckpointError` |
| Unknown variant or device, unavailable accelerator, invalid class map | `ValueError` |

RF-DETR loads weights with `strict=False`, so the adapter compares every
tensor name and shape itself; a checkpoint from another variant is rejected
rather than partly loaded. All errors include the checkpoint path.

## Predictions

- **Classes:** RF-DETR uses COCO category IDs directly as logit indices.
  Outputs at indices that are not configured category IDs (for example
  index 0, which has no training targets) are dropped.
- **Confidence:** every detection has a finite confidence in [0, 1] and at or
  above the floor. RF-DETR keeps scores strictly above its threshold; the
  adapter passes the floor as that threshold and also filters itself.
- **Boxes:** `[left, top, right, bottom]` in original-image pixels, with
  `left < right` and `top < bottom`, clipped to the image. Boxes entirely
  outside the image are dropped.
- **Resize:** RF-DETR 1.3.0 resizes the input to a square
  `resolution x resolution` tensor (changing the aspect ratio) and scales its
  normalized output boxes by the size of the image it was given. The adapter
  always passes the full original image, so boxes come back in original
  pixels; tests use a fake that performs the same square resize on a
  non-square image to confirm this.
- **Floor recorded:** `InferenceResult.confidence_floor` is the requested floor.
- **Empty vs failure:** no detections returns a result with
  `detections == ()`. A model exception or unusable output (non-finite
  box, out-of-range score, mismatched lengths) raises `InferenceError`.
  An invalid floor or image raises `ValueError` before prediction.

Checkpoints are pickles and are loaded with `torch.load(weights_only=False)`,
as RF-DETR does. Only load checkpoints from trusted runs.

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_rfdetr_adapter.py -q
```
