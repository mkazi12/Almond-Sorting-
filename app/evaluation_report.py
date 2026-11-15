"""Render saved evaluation and benchmark JSON as Markdown without running inference."""

import argparse
from collections import Counter
import hashlib
import html
import json
import math
import os
from pathlib import Path
import tempfile


class ReportError(ValueError):
    """Saved artifacts cannot safely produce a report."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReportError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path):
    data = Path(path).read_bytes()
    def invalid(value):
        raise ReportError(f"Nonfinite JSON number: {value}")
    return json.loads(data, object_pairs_hook=_object, parse_constant=invalid), data


def _text(value):
    if value is None:
        return "N/A"
    if isinstance(value, (dict, list, bool)):
        raise ReportError("Expected a scalar report value")
    if isinstance(value, float) and not math.isfinite(value):
        raise ReportError("Nonfinite report value")
    # Keep artifact text inside its table cell; do not interpret embedded HTML/Markdown.
    text = html.escape(str(value), quote=True)
    for char in "\\`*_[]|":
        text = text.replace(char, f"&#{ord(char)};")
    return text.replace("\r", " ").replace("\n", " ")


def _number(value, label, *, count=False, nullable=False, unit=False):
    if value is None and nullable:
        return
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ReportError(f"Invalid {label}")
    if (count and type(value) is not int) or (unit and value > 1):
        raise ReportError(f"Invalid {label}")


def _render(evaluation, benchmark, digest):
    for document in (evaluation, benchmark):
        if type(document["schema_version"]) is not int or document["schema_version"] != 1:
            raise ReportError("Unsupported artifact schema_version")
    for key in ("evaluation_id", "checkpoint", "dataset", "settings"):
        if evaluation[key] != benchmark[key]:
            raise ReportError(f"Artifact mismatch: {key}")
    if not isinstance(evaluation["evaluation_id"], str) or not evaluation["evaluation_id"]:
        raise ReportError("Missing evaluation identity")
    if benchmark["evaluation_sha256"] != digest:
        raise ReportError("Evaluation JSON hash mismatch")
    if benchmark["evaluation_created_at"] != evaluation["created_at"]:
        raise ReportError("Evaluation timestamp mismatch")
    checkpoint, dataset, settings = (evaluation[key] for key in ("checkpoint", "dataset", "settings"))
    if settings["split"] != "valid":
        raise ReportError("Report requires validation-only artifacts")
    for key in ("confidence_floor", "iou_threshold"):
        _number(settings[key], key, unit=True)
    measurements = benchmark["benchmark"]
    runtime = measurements["runtime"]
    if runtime["device"] != checkpoint["device"]:
        raise ReportError("Runtime device mismatch")
    ids = dataset["validation_image_ids"]
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ReportError("Invalid validation image IDs")
    if [row["image_id"] for row in measurements["samples"]] != ids:
        raise ReportError("Benchmark sample membership mismatch")
    if [row["image_id"] for row in evaluation["records"]] != ids:
        raise ReportError("Evaluation record membership mismatch")
    sample_resolutions = Counter()
    for sample, record in zip(measurements["samples"], evaluation["records"]):
        for key in ("width", "height"):
            _number(sample[key], key, count=True)
            _number(record[key], f"evaluation {key}", count=True)
            if sample[key] <= 0 or sample[key] != record[key]:
                raise ReportError("Sample resolution mismatch")
        sample_resolutions[(sample["width"], sample["height"])] += 1
        _number(sample["latency_ms"], "sample latency")
        if sample["latency_ms"] <= 0:
            raise ReportError("Invalid sample latency")
    resolution = measurements["model_input_resolution"]
    # Generic/fake detectors may not expose model input size; null means unknown.
    if resolution is not None:
        _number(resolution, "model input resolution", count=True)
        if resolution == 0:
            raise ReportError("Invalid model input resolution")
    summary = measurements["resolutions"]
    if not isinstance(summary, list):
        raise ReportError("Resolution summary must be a list")
    summary_resolutions = {}
    for row in summary:
        for key in ("width", "height", "count"):
            _number(row[key], f"resolution summary {key}", count=True)
            if row[key] == 0:
                raise ReportError(f"Invalid resolution summary {key}")
        size = (row["width"], row["height"])
        if size in summary_resolutions:
            raise ReportError("Duplicate resolution summary entry")
        summary_resolutions[size] = row["count"]
    if summary_resolutions != sample_resolutions:
        raise ReportError("Resolution summary disagrees with saved benchmark samples")
    if measurements["sample_count"] != len(ids) or evaluation["summary"]["images"] != len(ids):
        raise ReportError("Image count mismatch")
    for key in ("warmup_runs", "sample_count", "batch_size"):
        _number(measurements[key], key, count=True)
    if measurements["batch_size"] != 1 or measurements["warmup_image_id"] != ids[0]:
        raise ReportError("Unsupported benchmark batch or warm-up image")
    for key in ("p50_latency_ms", "p95_latency_ms", "total_timed_seconds", "throughput_images_per_second"):
        _number(measurements[key], key)
        if measurements[key] == 0:
            raise ReportError(f"Invalid {key}")
    metrics = evaluation["metrics"]
    ap, operating = metrics["average_precision"], metrics["operating_point"]
    if (ap["prediction_confidence_floor"] != settings["confidence_floor"]
            or operating["confidence_threshold"] != settings["confidence_floor"]
            or operating["iou_threshold"] != settings["iou_threshold"]):
        raise ReportError("Metric settings mismatch")
    for key in ("map_50", "map_50_95"):
        _number(ap[key], key, nullable=True, unit=True)
    for key in ("precision", "recall"):
        _number(operating[key], key, nullable=True, unit=True)
    for key in ("labeled_object_count", "prediction_count", "true_positive_count",
                "false_positive_count", "false_negative_count"):
        _number(operating[key], key, count=True)
    for summary_key, metric_key in (("ground_truths", "labeled_object_count"),
            ("predictions", "prediction_count"), ("true_positives", "true_positive_count"),
            ("false_positives", "false_positive_count"), ("false_negatives", "false_negative_count"),
            ("precision", "precision"), ("recall", "recall")):
        if evaluation["summary"][summary_key] != operating[metric_key]:
            raise ReportError("Evaluation summary disagrees with saved metrics")
    if [(row["class_id"], row["class_name"]) for row in metrics["per_class"]] != [
            (row["class_id"], row["class_name"]) for row in evaluation["classes"]]:
        raise ReportError("Class identity mismatch")

    lines = ["# Validation evaluation report", "",
             "Saved validation results; frozen test membership was excluded by the evaluation workflow.",
             "Values below are copied from saved JSON. N/A means unavailable or undefined, not zero.",
             "Accuracy values use the 0–1 scale. This report does not make an operator-trial readiness decision.", ""]
    def table(title, rows):
        lines.extend([f"## {title}", "", "| Field | Saved value |", "| --- | --- |"])
        lines.extend(f"| {_text(label)} | {_text(value)} |" for label, value in rows)
        lines.append("")

    table("Run identity", [("Evaluation ID", evaluation["evaluation_id"]),
          ("Evaluation created", evaluation["created_at"]), ("Benchmark created", benchmark["created_at"]),
          ("Evaluation JSON SHA-256", digest)] +
          [(f"Checkpoint {key}", checkpoint[key]) for key in ("path", "sha256", "variant", "model_label", "device")] +
          [(f"Dataset {key}", dataset[key]) for key in ("version", "content_hash", "frozen_test_version", "frozen_test_lock_hash")] +
          [("Validation images", evaluation["summary"]["images"]),
           ("Confidence threshold", settings["confidence_floor"]), ("Operating IoU threshold", settings["iou_threshold"]),
           ("AP prediction confidence floor", ap["prediction_confidence_floor"]),
           ("AP rule", settings["average_precision_rule"]), ("Matching rule", settings["matching_rule"])])
    table("Detection quality", [("Box mAP@0.50", ap["map_50"]), ("Box mAP@0.50:0.95", ap["map_50_95"])] +
          [(key, operating[key]) for key in ("precision", "recall", "labeled_object_count", "prediction_count",
                                             "true_positive_count", "false_positive_count", "false_negative_count")])
    lines.extend(["## Class results", "", "| ID | Class | Labeled objects | Predictions | AP@0.50 | AP@0.50:0.95 | Precision | Recall |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for row in metrics["per_class"]:
        for key in ("ground_truth_count", "prediction_count"):
            _number(row[key], key, count=True)
        values = [row["average_precision"]["ap_50"], row["average_precision"]["ap_50_95"],
                  row["operating_precision"], row["operating_recall"]]
        for value in values:
            _number(value, "class metric", nullable=True, unit=True)
        lines.append("| " + " | ".join(_text(value) for value in
                     [row["class_id"], row["class_name"], row["ground_truth_count"], row["prediction_count"], *values]) + " |")
    lines.append("")
    table("Inference performance", [(key, measurements[key]) for key in (
        "warmup_runs", "warmup_image_id", "sample_count", "batch_size", "model_input_resolution",
        "p50_latency_ms", "p95_latency_ms", "total_timed_seconds", "throughput_images_per_second",
        "timing_scope", "prior_inference", "percentile_method")])
    table("Original image resolutions", [(f'{row["width"]} x {row["height"]}', row["count"])
                                         for row in measurements["resolutions"]])
    table("Hardware and runtime", [(key, runtime[key]) for key in (
        "device", "python", "platform", "machine", "processor", "logical_cpu_count", "cuda_version", "accelerator_name")] +
        [(f"Package {key}", value) for key, value in sorted(runtime["packages"].items())])
    lines.extend(["Throughput covers sequential detector calls only. Small sample sets provide limited tail-latency evidence.", ""])
    return "\n".join(lines)


def generate_report(evaluation_path, benchmark_path, output_path):
    """Verify saved artifacts, then publish a new report without overwriting files."""
    temporary = None
    try:
        target = Path(output_path).expanduser().resolve()
        if target.exists() or target in {Path(evaluation_path).resolve(), Path(benchmark_path).resolve()}:
            raise ReportError("Report output must be a new file distinct from its inputs")
        evaluation, data = _load(evaluation_path)
        benchmark, _ = _load(benchmark_path)
        text = _render(evaluation, benchmark, hashlib.sha256(data).hexdigest())
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic publication with no overwrite, including a concurrently created target.
        os.link(temporary, target)
        return target
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise ReportError(f"Cannot generate Markdown report: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = generate_report(args.evaluation, args.benchmark, args.output)
    except ReportError as error:
        parser.exit(2, f"{error}\n")
    print(f"Saved report: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
