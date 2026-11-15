"""Deterministic benchmark checks: no weights, accelerator, or sleeps."""
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest

from app.benchmark import benchmark_validation, runtime_context
from app.inference import RGBImage, InferenceResult


def test_streaming_timing_and_statistics():
    events, references = [], []
    sizes = [(2, 3), (4, 5)]
    images = [(SimpleNamespace(image_id=str(i), width=w, height=h), i)
              for i, (w, h) in enumerate(sizes)]

    def read(index):
        assert all(ref() is None for ref in references)
        events.append("read")
        w, h = sizes[index]
        image = RGBImage(w, h, bytes(w*h*3))
        references.append(weakref.ref(image))
        return image

    def predict(image, *, confidence_floor):
        events.append("predict")
        return InferenceResult(image.width, image.height, confidence_floor)

    ticks = iter([0, 10_000_000, 20_000_000, 50_000_000])

    def clock():
        events.append("clock")
        return next(ticks)

    result = benchmark_validation(SimpleNamespace(predict=predict, input_resolution=64),
                                  images, image_reader=read, class_map={1: "A"},
                                  confidence_floor=.5, device="cpu", warmup_runs=2,
                                  clock=clock, runtime_provider=lambda _: (
                                      {}, lambda: events.append("sync")))
    assert events == ["read", "predict", "predict", "sync", "clock", "predict", "sync", "clock",
                      "read", "sync", "clock", "predict", "sync", "clock"]
    assert result["sample_count"] == 2
    assert result["p50_latency_ms"] == 20
    assert result["p95_latency_ms"] == 29
    assert result["throughput_images_per_second"] == 50
    assert result["model_input_resolution"] == 64
    assert result["resolutions"] == [dict(width=2, height=3, count=1), dict(width=4, height=5, count=1)]


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_duration(duration):
    ticks = iter([0, duration])
    detector = SimpleNamespace(predict=lambda image, **kw: InferenceResult(1, 1, .5))
    with pytest.raises(ValueError, match="duration"):
        benchmark_validation(detector, [(SimpleNamespace(image_id="a", width=1, height=1), "a")],
                             image_reader=lambda _: RGBImage(1, 1, bytes(3)), class_map={1: "A"},
                             confidence_floor=.5, device="cpu", warmup_runs=0,
                             clock=lambda: next(ticks), runtime_provider=lambda _: ({}, lambda: None))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_device_completion_barrier(device, monkeypatch):
    torch = Mock()
    torch.version.cuda = "fixture-version"
    torch.cuda.get_device_name.return_value = "fixture-GPU"
    importer = Mock(return_value=torch)
    monkeypatch.setattr("app.benchmark.importlib.import_module", importer)
    metadata, sync = runtime_context(device)
    sync()
    if device == "cpu":
        importer.assert_not_called()
    elif device == "cuda":
        torch.cuda.synchronize.assert_called_once()
        assert metadata["accelerator_name"] == "fixture-GPU"
    else:
        torch.mps.synchronize.assert_called_once()


def test_unavailable_cuda_fails(monkeypatch):
    torch = Mock()
    torch.cuda.is_available.return_value = False
    monkeypatch.setattr("app.benchmark.importlib.import_module", lambda _: torch)
    with pytest.raises(ValueError, match="unavailable"):
        runtime_context("cuda")


def test_missing_packages_are_unknown(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(_):
        raise PackageNotFoundError

    monkeypatch.setattr("app.benchmark.version", missing)
    metadata, _ = runtime_context("cpu")
    assert all(value is None for value in metadata["packages"].values())
