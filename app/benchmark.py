"""Streaming, batch-one detector timings; callers supply guarded validation metadata."""

import importlib
from importlib.metadata import PackageNotFoundError, version
import math
import os
import platform
from time import perf_counter_ns

from app.inference import validate_result


def runtime_context(device):
    """Collect installed versions and return the device completion barrier."""
    packages = {}
    for name in ("rfdetr", "torch", "numpy", "Pillow"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    machine=platform.machine(), processor=platform.processor() or None,
                    logical_cpu_count=os.cpu_count(), packages=packages, device=device,
                    cuda_version=None, accelerator_name=None)
    synchronize = lambda: None
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError(f"Unsupported benchmark device: {device}")
    if device != "cpu":
        torch = importlib.import_module("torch")
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable for benchmarking")
            metadata.update(cuda_version=torch.version.cuda,
                            accelerator_name=torch.cuda.get_device_name())
            synchronize = torch.cuda.synchronize
        else:
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is unavailable for benchmarking")
            synchronize = torch.mps.synchronize
    return metadata, synchronize


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def benchmark_validation(detector, images, *, image_reader, class_map,
                         confidence_floor, device, warmup_runs=5,
                         clock=perf_counter_ns, runtime_provider=runtime_context):
    """Images is an iterable of (accepted-image metadata, resolved path).

    Only one decoded image is retained. No path selection or model loading is
    performed here; the evaluator remains responsible for frozen-test guards.
    """
    if type(warmup_runs) is not int or warmup_runs < 0:
        raise ValueError("warmup_runs must be a nonnegative integer")
    runtime, synchronize = runtime_provider(device)
    samples, resolutions = [], {}
    for metadata, path in images:
        image = image_reader(path)
        if (image.width, image.height) != (metadata.width, metadata.height):
            raise ValueError(f"Benchmark image dimensions changed: {metadata.image_id}")

        def checked(result):
            validate_result(result, image_width=image.width, image_height=image.height,
                            confidence_floor=confidence_floor, classes=class_map)

        if not samples:
            for _ in range(warmup_runs):
                checked(detector.predict(image, confidence_floor=confidence_floor))
        synchronize()
        start = clock()
        result = detector.predict(image, confidence_floor=confidence_floor)
        synchronize()
        elapsed = (clock() - start) / 1_000_000
        checked(result)
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Benchmark duration must be finite and positive")
        samples.append(dict(image_id=metadata.image_id, width=image.width,
                            height=image.height, latency_ms=elapsed))
        size = (image.width, image.height)
        resolutions[size] = resolutions.get(size, 0) + 1
        del image, result
    if not samples:
        raise ValueError("Benchmark requires validation images")
    latencies = [sample["latency_ms"] for sample in samples]
    return dict(warmup_runs=warmup_runs, warmup_image_id=samples[0]["image_id"],
                sample_count=len(samples), batch_size=1,
                timing_scope="Detector.predict plus device completion; excludes image decoding",
                prior_inference="complete validation evaluation on the same detector",
                percentile_method="linear interpolation at (n-1)*q",
                model_input_resolution=getattr(detector, "input_resolution", None),
                resolutions=[dict(width=w, height=h, count=count)
                             for (w, h), count in sorted(resolutions.items())],
                samples=samples, p50_latency_ms=percentile(latencies, .5),
                p95_latency_ms=percentile(latencies, .95),
                total_timed_seconds=sum(latencies) / 1000,
                throughput_images_per_second=len(samples) * 1000 / sum(latencies),
                runtime=runtime)
