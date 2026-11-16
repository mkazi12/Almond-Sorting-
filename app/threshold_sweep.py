"""Sweep operating confidence thresholds over a saved validation evaluation (SCRUM-67).

Example::

    python -m app.threshold_sweep results/evaluation.json results/sweep.json \\
        --min-confidence 0.5 --max-confidence 0.95 --step 0.05

The sweep reads only the evaluation JSON written by ``app.evaluate``; it loads
no model and opens no images. At each threshold it keeps the saved validation
predictions with ``confidence >= threshold`` (the adapter's inclusive floor
rule), re-matches them with the evaluation's matching rule and IoU threshold,
and records overall and per-class precision, recall and counts.

The evaluation only kept predictions at or above its confidence floor, so a
threshold below that floor is rejected: re-run ``app.evaluate`` with a lower
``--confidence-floor`` to sweep lower.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.evaluate import MATCHING_RULE, _match, _ratio
from app.inference import Detection
from app.run_artifacts import RunArtifactError, new_run_id, write_json_atomic


THRESHOLD_RULE = "keep prediction when confidence >= threshold"


class SweepError(ValueError):
    """The saved evaluation or the requested threshold range cannot be swept."""


def _decimal(value, name: str) -> Decimal:
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise SweepError(f"{name} must be a number.") from error
    if not number.is_finite():
        raise SweepError(f"{name} must be a finite number.")
    return number


def threshold_range(minimum, maximum, step) -> tuple[float, ...]:
    """Return ``minimum, minimum + step, ..., maximum`` (inclusive).

    Decimal arithmetic keeps each threshold exact: in floats, 0.5 + 3 * 0.1 is
    0.8000000000000002, which would wrongly drop a prediction scoring 0.8.
    """
    low, high = _decimal(minimum, "Minimum confidence"), _decimal(maximum, "Maximum confidence")
    if not 0 <= low <= high <= 1:
        raise SweepError("Confidence range must satisfy 0 <= minimum <= maximum <= 1.")
    if low == high:
        return (float(low),)
    size = _decimal(step, "Confidence step")
    if size <= 0:
        raise SweepError("Confidence step must be greater than 0.")
    steps = (high - low) / size
    if steps != steps.to_integral_value():
        raise SweepError(f"Confidence range {low}-{high} is not a whole number of {size} steps.")
    return tuple(float(low + size * index) for index in range(int(steps) + 1))


def _detections(record: dict, threshold: float = 0.0) -> tuple[Detection, ...]:
    return tuple(
        Detection(item["category_id"], item["confidence"], tuple(item["bbox_xyxy"]))
        for item in record["predictions"] if item["confidence"] >= threshold
    )


def _check_evaluation(evaluation: dict) -> tuple[float, float, dict[int, str]]:
    """Reject anything but an intact validation evaluation produced by the matching rule."""
    try:
        settings = evaluation["settings"]
        if evaluation["schema_version"] != 1 or settings["split"] != "valid":
            raise SweepError("Threshold sweeps only accept validation evaluation schema 1.")
        if settings["matching_rule"] != MATCHING_RULE:
            raise SweepError(f"Unsupported matching rule: {settings['matching_rule']}.")
        floor, iou_threshold = float(settings["confidence_floor"]), float(settings["iou_threshold"])
        classes = {row["class_id"]: row["class_name"] for row in evaluation["classes"]}
        if [record["image_id"] for record in evaluation["records"]] != \
                evaluation["dataset"]["validation_image_ids"]:
            raise SweepError("Evaluation records differ from its validation membership.")
        for record in evaluation["records"]:
            image_id = record["image_id"]
            detections = _detections(record)
            if any(item.confidence < floor for item in detections):
                raise SweepError(f"Image {image_id} has a prediction below the confidence floor.")
            if list(detections) != sorted(detections, key=lambda item: -item.confidence):
                raise SweepError("Saved predictions must retain descending-confidence order.")
            used = {item.category_id for item in detections}
            used |= {label["category_id"] for label in record["labels"]}
            if not used <= classes.keys():
                raise SweepError(f"Image {image_id} uses a category outside the class map.")
            expected = _match(detections, record["labels"], iou_threshold)
            if any(record[key] != value for key, value in expected.items()):
                raise SweepError(f"Saved matches for image {image_id} disagree with the matching rule.")
    except SweepError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise SweepError(f"Malformed evaluation: {type(error).__name__}: {error}") from error
    return floor, iou_threshold, classes


def _counts(ground_truth: int, predictions: int, true_positives: int) -> dict:
    return {
        "ground_truth_count": ground_truth,
        "prediction_count": predictions,
        "true_positive_count": true_positives,
        "false_positive_count": predictions - true_positives,
        "false_negative_count": ground_truth - true_positives,
        "precision": _ratio(true_positives, predictions),  # null: no predictions
        "recall": _ratio(true_positives, ground_truth),     # null: no ground truth
    }


def _result_at(evaluation: dict, classes: dict[int, str], threshold: float,
               iou_threshold: float) -> dict:
    totals = {category_id: [0, 0, 0] for category_id in classes}  # ground truth, predictions, TP
    for record in evaluation["records"]:
        labels, kept = record["labels"], _detections(record, threshold)
        for label in labels:
            totals[label["category_id"]][0] += 1
        for detection in kept:
            totals[detection.category_id][1] += 1
        for match in _match(kept, labels, iou_threshold)["matches"]:
            totals[labels[match["label_index"]]["category_id"]][2] += 1
    return {
        "confidence_threshold": threshold,
        "overall": {
            "image_count": len(evaluation["records"]),
            **_counts(*(sum(values[index] for values in totals.values()) for index in range(3))),
        },
        "per_class": [
            {"class_id": category_id, "class_name": name, **_counts(*totals[category_id])}
            for category_id, name in sorted(classes.items())
        ],
    }


def sweep_evaluation(evaluation: dict, thresholds, *, evaluation_sha256: str | None = None,
                     evaluation_path: str | None = None) -> dict:
    """Return sweep results for a checked evaluation at each requested threshold."""
    floor, iou_threshold, classes = _check_evaluation(evaluation)
    values = tuple(thresholds)
    if not values or list(values) != sorted(set(values)):
        raise SweepError("Confidence thresholds must be nonempty, unique and ascending.")
    if values[0] < floor:
        raise SweepError(
            f"Confidence threshold {values[0]} is below the evaluation's prediction confidence "
            f"floor {floor}; predictions under the floor were not retained. Re-run "
            f"app.evaluate with --confidence-floor {values[0]} or lower to sweep this range."
        )
    return {
        "schema_version": 1,
        "sweep_id": new_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": {"evaluation_id": evaluation["evaluation_id"],
                       "path": evaluation_path, "sha256": evaluation_sha256},
        "checkpoint": evaluation["checkpoint"],
        "dataset": evaluation["dataset"],
        "settings": {
            "split": "valid",
            "prediction_confidence_floor": floor,
            "confidence_thresholds": list(values),
            "threshold_rule": THRESHOLD_RULE,
            "iou_threshold": iou_threshold,
            "matching_rule": MATCHING_RULE,
            "matching_tie_break_rule": evaluation["settings"].get("matching_tie_break_rule"),
        },
        "results": [_result_at(evaluation, classes, value, iou_threshold) for value in values],
    }


def run_sweep(evaluation_path, output_path, *, minimum, maximum, step) -> dict:
    """Read a saved evaluation, sweep the configured range and write a new JSON file."""
    thresholds = threshold_range(minimum, maximum, step)
    source = Path(evaluation_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    if source == target or target.exists():
        raise SweepError(f"Sweep output must be a new file: {target}")
    try:
        data = source.read_bytes()
        evaluation = json.loads(data)
    except (OSError, ValueError) as error:
        raise SweepError(f"Cannot read evaluation {source}: {error}") from error
    sweep = sweep_evaluation(evaluation, thresholds, evaluation_path=str(source),
                             evaluation_sha256=hashlib.sha256(data).hexdigest())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, sweep, operation="write threshold sweep", overwrite=False)
    except (OSError, RunArtifactError) as error:
        raise SweepError(f"Cannot write threshold sweep: {error}") from error
    return sweep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evaluation", type=Path, help="evaluation JSON written by app.evaluate")
    parser.add_argument("output", type=Path, help="new sweep JSON file")
    parser.add_argument("--min-confidence", required=True,
                        help="lowest threshold; must be >= the evaluation's confidence floor")
    parser.add_argument("--max-confidence", required=True, help="highest threshold (inclusive)")
    parser.add_argument("--step", default="0.05", help="threshold increment (default 0.05)")
    args = parser.parse_args(argv)
    try:
        sweep = run_sweep(args.evaluation, args.output, minimum=args.min_confidence,
                          maximum=args.max_confidence, step=args.step)
    except SweepError as error:
        parser.exit(2, f"Threshold sweep failed: {error}\n")
    print(f"Swept {len(sweep['results'])} thresholds; wrote {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
