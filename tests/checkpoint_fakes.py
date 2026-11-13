"""Torch-free stand-ins for RF-DETR checkpoint files.

Fake checkpoints are JSON with the same structure RF-DETR saves; ``load_fake``
returns objects shaped like ``torch.load`` output (tensors expose ``shape``,
``args`` exposes attributes).
"""

import json
from pathlib import Path
from types import SimpleNamespace


class FakeTensor:
    def __init__(self, *shape):
        self.shape = shape


def write_fake_checkpoint(path, *, epoch=None, class_names=("A", "B"), logits=None,
                          state=None, stripped=False):
    """Write a JSON checkpoint; ``state`` maps tensor names to shapes."""
    shapes = dict(state or {"backbone.weight": [4, 4]})
    shapes["class_embed.bias"] = [logits if logits is not None else len(class_names) + 1]
    payload = {"model": shapes, "args": {"class_names": list(class_names), "resolution": 384}}
    if not stripped:
        payload["epoch"] = epoch
    Path(path).write_text(json.dumps(payload))
    return Path(path)


def load_fake(path):
    payload = json.loads(Path(path).read_text())
    if "model" in payload:
        payload["model"] = {name: FakeTensor(*shape) for name, shape in payload["model"].items()}
    if "args" in payload:
        payload["args"] = SimpleNamespace(**payload["args"])
    return payload


def write_metrics_log(path, values, ema_values=None):
    """Write RF-DETR-style log.txt lines (validation metrics use test_* keys)."""
    lines = []
    for epoch, value in enumerate(values):
        entry = {"epoch": epoch, "train_loss": 1.0 / (epoch + 1),
                 "test_coco_eval_bbox": [value, value + 0.1]}
        if ema_values is not None:
            entry["ema_test_coco_eval_bbox"] = [ema_values[epoch], ema_values[epoch] + 0.1]
        lines.append(json.dumps(entry))
    Path(path).write_text("\n".join(lines) + "\n")
