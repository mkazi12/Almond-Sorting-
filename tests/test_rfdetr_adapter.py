"""SCRUM-60: compatible checkpoint loading, floor-recorded predictions, original-pixel boxes.

The fake model mirrors RF-DETR 1.3.0 ``predict``: it resizes whatever image it
receives to a square input, finds the object in that resized input, and
scales normalized boxes by the size of the image it was given.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from app.checkpoint import CheckpointError, IncompatibleCheckpointError
from app.inference import InferenceError, RGBImage, validate_result
from app.rfdetr_adapter import load_detector, load_inference_config, main
from app.run_artifacts import finalize_run
from tests.checkpoint_fakes import FakeTensor, load_fake, write_fake_checkpoint, write_metrics_log


CLASSES = {1: "A", 2: "B"}
STATE = {"backbone.weight": [4, 4], "class_embed.weight": [3, 8]}


class FakeNetwork:
    def __init__(self, shapes):
        self.shapes = shapes

    def state_dict(self):
        return {name: FakeTensor(*shape) for name, shape in self.shapes.items()}


class FakeRFDETR:
    """Behaves like RF-DETR 1.3.0's predict for a single bright object."""

    def __init__(self, *, resolution=64, shapes=None, extra=(), fail=None):
        self.model = SimpleNamespace(model=FakeNetwork(shapes or {**STATE, "class_embed.bias": [3]}),
                                     resolution=resolution)
        self.extra = list(extra)
        self.fail = fail
        self.calls = []

    def __call__(self, pretrain_weights, device):
        self.calls.append((pretrain_weights, device))
        return self

    def predict(self, image, threshold=0.5):
        if self.fail:
            raise self.fail
        self.calls.append(("predict", image.size, threshold))
        width, height = image.size
        resolution = self.model.resolution
        square = np.asarray(image.resize((resolution, resolution), Image.BILINEAR).convert("L"))
        rows, columns = np.nonzero(square > 127)
        rows_out, scores, labels = [], [], []
        if len(rows):
            # Box in the resized input, normalized, then scaled to the given size.
            normalized = np.array([columns.min(), rows.min(), columns.max() + 1, rows.max() + 1]) / resolution
            rows_out.append(normalized * [width, height, width, height])
            scores.append(0.9)
            labels.append(2)
        for box, score, label in self.extra:
            rows_out.append(box)
            scores.append(score)
            labels.append(label)
        keep = [score > threshold for score in scores]
        return SimpleNamespace(
            xyxy=np.array(rows_out, dtype=float).reshape(-1, 4)[keep],
            confidence=np.array(scores, dtype=float)[keep],
            class_id=np.array(labels, dtype=int)[keep],
        )


def rgb(width=200, height=80, box=(50, 20, 150, 60)):
    image = Image.new("RGB", (width, height))
    left, top, right, bottom = box
    image.paste((255, 255, 255), (left, top, right, bottom))
    return RGBImage(width, height, image.tobytes())


@pytest.fixture
def checkpoint(tmp_path):
    return write_fake_checkpoint(tmp_path / "best.pth", state=STATE, stripped=True)


def load(checkpoint, model=None, classes=CLASSES, variant="nano"):
    return load_detector(checkpoint, classes, variant=variant, loader=load_fake,
                         model_factory=model or FakeRFDETR())


def test_compatible_checkpoint_and_class_map_load(checkpoint):
    model = FakeRFDETR()
    detector = load(checkpoint, model)
    assert model.calls == [(str(checkpoint), "cpu")]
    assert detector.class_map == CLASSES
    assert detector.model_label.startswith("rfdetr-nano:best.pth@")


@pytest.mark.parametrize("names,logits,message", [
    (("B", "A"), None, "do not match configured classes"),
    (("A",), 3, "do not match configured classes"),
    (("A", "B"), 2, "uses category ID 2"),
])
def test_incompatible_class_metadata_fails_before_model_is_built(tmp_path, names, logits, message):
    path = write_fake_checkpoint(tmp_path / "x.pth", class_names=names, logits=logits, state=STATE)
    model = FakeRFDETR()
    with pytest.raises(IncompatibleCheckpointError, match=message):
        load(path, model)
    assert model.calls == []


def test_checkpoint_without_class_names_is_rejected(tmp_path):
    path = write_fake_checkpoint(tmp_path / "x.pth", state=STATE)
    payload = json.loads(path.read_text())
    payload["args"].pop("class_names")
    path.write_text(json.dumps(payload))
    with pytest.raises(IncompatibleCheckpointError, match="no class names"):
        load(path)


@pytest.mark.parametrize("content,error", [
    (None, "missing"), (b"", "empty"), (b'{"model": {"class_', "unreadable or incomplete"),
    (b'{"weights": {}}', "no 'model' weights"),
])
def test_missing_unreadable_or_foreign_files_raise_specific_errors(tmp_path, content, error):
    path = tmp_path / "x.pth"
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(CheckpointError, match=error) as raised:
        load(path)
    assert raised.value.path == path


def test_weights_from_another_variant_are_rejected(checkpoint):
    other_variant = FakeRFDETR(shapes={**STATE, "backbone.weight": [6, 4], "class_embed.bias": [3],
                                       "decoder.extra": [2]})
    with pytest.raises(IncompatibleCheckpointError, match="1 missing tensors .*decoder.extra.*1 shape mismatches"):
        load(checkpoint, other_variant)


def test_rfdetr_load_failure_is_reported_as_incompatible(checkpoint):
    with pytest.raises(IncompatibleCheckpointError, match="size mismatch"):
        load(checkpoint, Mock(side_effect=RuntimeError("size mismatch for class_embed.weight")))


@pytest.mark.parametrize("classes,variant", [({}, "nano"), ({1: " "}, "nano"), (CLASSES, "tiny")])
def test_invalid_configuration_is_rejected(checkpoint, classes, variant):
    with pytest.raises(ValueError):
        load(checkpoint, classes=classes, variant=variant)


def test_square_resize_is_mapped_back_to_original_pixels(checkpoint):
    model = FakeRFDETR(resolution=64)
    result = load(checkpoint, model).predict(rgb(200, 80, (50, 20, 150, 60)), confidence_floor=0.5)
    assert model.calls[-1] == ("predict", (200, 80), 0.5)  # full original image, not a resized copy
    (detection,) = result.detections
    assert detection.bbox_xyxy == pytest.approx((50, 20, 150, 60), abs=200 / 64)
    assert (result.image_width, result.image_height) == (200, 80)


def test_detections_have_configured_class_unit_confidence_and_in_bounds_boxes(checkpoint):
    model = FakeRFDETR(extra=[
        ((-4.0, -2.0, 30.0, 40.0), 0.8, 1),     # slightly outside: clipped
        ((10.0, 10.0, 20.0, 20.0), 0.95, 0),    # untrained logit index 0: dropped
        ((210.0, 5.0, 230.0, 30.0), 0.7, 1),    # entirely outside: dropped
    ])
    result = load(checkpoint, model).predict(rgb(), confidence_floor=0.5)
    assert [d.category_id for d in result.detections] == [2, 1]
    assert result.detections[1].bbox_xyxy == (0.0, 0.0, 30.0, 40.0)
    for detection in result.detections:
        left, top, right, bottom = detection.bbox_xyxy
        assert detection.category_id in CLASSES
        assert 0 <= detection.confidence <= 1
        assert 0 <= left < right <= 200 and 0 <= top < bottom <= 80
    validate_result(result, image_width=200, image_height=80, confidence_floor=0.5, classes=CLASSES)


def test_result_records_requested_floor_and_omits_lower_scores(checkpoint):
    model = FakeRFDETR(extra=[((1.0, 1.0, 5.0, 5.0), 0.6, 1)])
    detector = load(checkpoint, model)
    assert detector.predict(rgb(), confidence_floor=0.75).confidence_floor == 0.75
    assert [d.confidence for d in detector.predict(rgb(), confidence_floor=0.75).detections] == [0.9]
    assert len(detector.predict(rgb(), confidence_floor=0.3).detections) == 2


def test_no_detections_is_a_successful_empty_result(checkpoint):
    result = load(checkpoint).predict(rgb(box=(0, 0, 0, 0)), confidence_floor=0.4)
    assert result.detections == () and result.confidence_floor == 0.4


def test_inference_failure_is_distinguishable_from_empty_result(checkpoint):
    detector = load(checkpoint, FakeRFDETR(fail=RuntimeError("CUDA error")))
    with pytest.raises(InferenceError, match="CUDA error"):
        detector.predict(rgb(), confidence_floor=0.5)


@pytest.mark.parametrize("output", [
    SimpleNamespace(xyxy=np.array([[1.0, 1.0, np.nan, 5.0]]), confidence=np.array([0.9]), class_id=np.array([1])),
    SimpleNamespace(xyxy=np.array([[1.0, 1.0, 5.0, 5.0]]), confidence=np.array([1.5]), class_id=np.array([1])),
    SimpleNamespace(xyxy=np.zeros((2, 4)), confidence=np.array([0.9]), class_id=np.array([1])),
    SimpleNamespace(xyxy=np.array([[1.0, 1.0, 5.0, 5.0]]), confidence=np.array([0.9]), class_id=np.array([1.5])),
    "not detections",
])
def test_unusable_model_output_is_an_inference_error(checkpoint, output):
    model = FakeRFDETR()
    model.predict = lambda image, threshold: output
    with pytest.raises(InferenceError):
        load(checkpoint, model).predict(rgb(), confidence_floor=0.5)


@pytest.mark.parametrize("floor", [-0.1, 1.1, float("nan"), "0.5", True])
def test_invalid_floor_is_rejected_before_prediction(checkpoint, floor):
    model = FakeRFDETR()
    with pytest.raises(ValueError):
        load(checkpoint, model).predict(rgb(), confidence_floor=floor)
    assert not any(call[0] == "predict" for call in model.calls)


def test_best_checkpoint_recorded_by_a_run_reloads_through_adapter(tmp_path):
    """SCRUM-59 acceptance 3: the run's best reference loads in the adapter."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(json.dumps({"run_id": "r1", "dataset": {"class_map": {"1": "A", "2": "B"}}}))
    write_metrics_log(run / "log.txt", [0.1, 0.2])
    write_fake_checkpoint(run / "checkpoint.pth", epoch=1, state=STATE)
    write_fake_checkpoint(run / "checkpoint_best_total.pth", state=STATE, stripped=True)
    record = finalize_run(run, loader=load_fake)
    classes = {int(key): name for key, name in record["dataset"]["class_map"].items()}
    detector = load(run / record["checkpoints"]["best"]["path"], classes=classes)
    assert detector.checkpoint.sha256 == record["checkpoints"]["best"]["sha256"]


def test_example_config_and_command_line(tmp_path, monkeypatch, capsys, checkpoint):
    example = json.loads(open("configs/inference.example.json").read())
    assert set(example) == {"checkpoint", "class_map", "variant", "device", "confidence_floor"}

    (tmp_path / "classes.json").write_text(json.dumps({"1": "A", "2": "B"}))
    (tmp_path / "inference.json").write_text(json.dumps(dict(
        checkpoint="best.pth", class_map="classes.json", variant="nano", device="cpu",
        confidence_floor=0.6)))
    Image.frombytes("RGB", (200, 80), rgb().pixels).save(tmp_path / "image.png")
    monkeypatch.setattr("app.rfdetr_adapter.load_detector",
                        lambda path, classes, **kw: load(path, classes=classes))
    assert main([str(tmp_path / "inference.json"), str(tmp_path / "image.png")]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["confidence_floor"] == 0.6
    assert output["detections"][0]["class_name"] == "B"


@pytest.mark.parametrize("change,field", [
    ({"confidence_floor": 2}, "confidence_floor"), ({"extra": 1}, "config"),
])
def test_invalid_inference_config_names_the_problem(tmp_path, change, field):
    (tmp_path / "classes.json").write_text(json.dumps({"1": "A"}))
    raw = {**dict(checkpoint="x.pth", class_map="classes.json", variant="nano", device="cpu",
                  confidence_floor=0.5), **change}
    (tmp_path / "inference.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=field):
        load_inference_config(tmp_path / "inference.json")
