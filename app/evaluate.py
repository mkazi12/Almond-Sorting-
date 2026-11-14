"""Evaluate one RF-DETR checkpoint on a guarded validation split.

Example::

    python -m app.evaluate DATASET_ROOT MANIFEST SPLIT_MANIFEST TEST_LOCK \\
        CHECKPOINT CLASS_MAP OUTPUT --variant nano --confidence-floor 0.5 \\
        --iou-threshold 0.5

The frozen test lock is mandatory. Validation membership is checked before
decoding images, and the output is published only after every selected image
has produced a validated inference result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from PIL import Image, UnidentifiedImageError

from app.checkpoint import CheckpointError, sha256_file
from app.dataset.frozen_test_set import (
    FrozenSetError,
    FrozenSetLeakError,
    assert_excluded,
    load_test_lock,
)
from app.dataset.paths import resolve_dataset_path
from app.dataset.records import AcceptedImage, DatasetInput, GroundTruthObject, objects_for_image
from app.dataset.rfdetr_export import (
    RfdetrExportError,
    SplitAssignment,
    load_class_map,
    load_file_inputs,
)
from app.inference import (
    Detection,
    InferenceError,
    RGBImage,
    confidence_value,
    validate_class_map,
    validate_result,
)
from app.run_artifacts import new_run_id, write_json_atomic
from app.rfdetr_adapter import load_detector


MATCHING_RULE = "greedy_one_to_one_class_aware_descending_confidence"
AP_IOU_THRESHOLDS = tuple(round(0.5 + index * 0.05, 2) for index in range(10))
AP_RECALL_POINTS = tuple(index / 100 for index in range(101))


class EvaluationError(ValueError):
    """The evaluation inputs or a complete evaluation run are unusable."""


def _iou(first: tuple[float, float, float, float],
         second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _label_box(obj: GroundTruthObject, image: AcceptedImage) -> tuple[float, float, float, float]:
    x, y, width, height = obj.bbox_xywh
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise EvaluationError(f"Label annotation {obj.annotation_id} has non-finite box coordinates.")
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > image.width or y + height > image.height:
        raise EvaluationError(
            f"Label annotation {obj.annotation_id} has a box outside image {image.image_id}."
        )
    return x, y, x + width, y + height


def _match(predictions: tuple[Detection, ...], labels: list[dict], iou_threshold: float) -> dict:
    unmatched = set(range(len(labels)))
    matches = []
    false_positives = []
    for prediction_index, prediction in enumerate(predictions):
        candidates = [
            (_iou(prediction.bbox_xyxy, label["bbox_xyxy"]), label_index)
            for label_index, label in enumerate(labels)
            if label_index in unmatched and label["category_id"] == prediction.category_id
        ]
        if candidates:
            overlap, label_index = max(candidates, key=lambda item: (item[0], -item[1]))
        else:
            overlap, label_index = 0.0, -1
        if label_index >= 0 and overlap >= iou_threshold:
            unmatched.remove(label_index)
            matches.append({
                "prediction_index": prediction_index,
                "label_index": label_index,
                "iou": overlap,
            })
        else:
            false_positives.append(prediction_index)
    return {
        "matches": matches,
        "false_positive_prediction_indices": false_positives,
        "false_negative_label_indices": sorted(unmatched),
    }


def _average_precision(records: list[dict], category_id: int, iou_threshold: float,
                       ground_truth_count: int) -> float | None:
    if ground_truth_count == 0:
        return None

    labels_by_image = {
        record["image_id"]: [
            label for label in record["labels"] if label["category_id"] == category_id
        ]
        for record in records
    }
    predictions = [
        (prediction["confidence"], record["image_id"], prediction_index, prediction["bbox_xyxy"])
        for record in records
        for prediction_index, prediction in enumerate(record["predictions"])
        if prediction["category_id"] == category_id
    ]
    predictions.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched_by_image: dict[str, set[int]] = {image_id: set() for image_id in labels_by_image}
    true_positives = []
    false_positives = []
    for _, image_id, _, box in predictions:
        labels = labels_by_image[image_id]
        candidates = [
            (_iou(box, label["bbox_xyxy"]), label_index)
            for label_index, label in enumerate(labels)
            if label_index not in matched_by_image[image_id]
        ]
        if candidates:
            overlap, label_index = max(candidates, key=lambda item: (item[0], -item[1]))
        else:
            overlap, label_index = 0.0, -1
        if label_index >= 0 and overlap >= iou_threshold:
            matched_by_image[image_id].add(label_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    cumulative_tp = 0
    cumulative_fp = 0
    recalls = []
    precisions = []
    for is_true_positive, is_false_positive in zip(true_positives, false_positives):
        cumulative_tp += is_true_positive
        cumulative_fp += is_false_positive
        recalls.append(cumulative_tp / ground_truth_count)
        precisions.append(cumulative_tp / (cumulative_tp + cumulative_fp))

    interpolated = [
        max((precision for precision, recall in zip(precisions, recalls) if recall >= target), default=0.0)
        for target in AP_RECALL_POINTS
    ]
    return sum(interpolated) / len(AP_RECALL_POINTS)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _calculate_metrics(records: list[dict], class_map: dict[int, str],
                       confidence_floor: float, operating_iou_threshold: float) -> dict:
    counts = {
        category_id: {"ground_truth": 0, "predictions": 0, "true_positives": 0}
        for category_id in class_map
    }
    for record in records:
        for label in record["labels"]:
            counts[label["category_id"]]["ground_truth"] += 1
        for prediction in record["predictions"]:
            counts[prediction["category_id"]]["predictions"] += 1
        for match in record["matches"]:
            label = record["labels"][match["label_index"]]
            counts[label["category_id"]]["true_positives"] += 1

    per_class = []
    included_ap50 = []
    included_ap_range = []
    for category_id, class_name in sorted(class_map.items()):
        class_count = counts[category_id]
        ground_truth_count = class_count["ground_truth"]
        prediction_count = class_count["predictions"]
        true_positive_count = class_count["true_positives"]
        false_positive_count = prediction_count - true_positive_count
        false_negative_count = ground_truth_count - true_positive_count
        by_iou = [
            {
                "iou_threshold": threshold,
                "average_precision": _average_precision(records, category_id, threshold, ground_truth_count),
            }
            for threshold in AP_IOU_THRESHOLDS
        ]
        ap50 = by_iou[0]["average_precision"]
        ap50_95 = (
            sum(item["average_precision"] for item in by_iou) / len(by_iou)
            if ground_truth_count else None
        )
        if ground_truth_count:
            included_ap50.append(ap50)
            included_ap_range.append(ap50_95)
        per_class.append({
            "class_id": category_id,
            "class_name": class_name,
            "ground_truth_count": ground_truth_count,
            "prediction_count": prediction_count,
            "true_positive_count": true_positive_count,
            "false_positive_count": false_positive_count,
            "false_negative_count": false_negative_count,
            "operating_precision": _ratio(true_positive_count, prediction_count),
            "operating_recall": _ratio(true_positive_count, ground_truth_count),
            "average_precision": {
                "ap_50": ap50,
                "ap_50_95": ap50_95,
                "by_iou_threshold": by_iou,
            },
        })

    total_ground_truths = sum(item["ground_truth"] for item in counts.values())
    total_predictions = sum(item["predictions"] for item in counts.values())
    total_true_positives = sum(item["true_positives"] for item in counts.values())
    total_false_positives = total_predictions - total_true_positives
    total_false_negatives = total_ground_truths - total_true_positives
    return {
        "task": "bounding_box_detection",
        "metric_names": {
            "map_50": "mAP@0.50",
            "map_50_95": "mAP@0.50:0.95",
            "class_ap": "AP",
            "operating_precision": "Operating precision",
            "operating_recall": "Operating recall",
        },
        "average_precision": {
            "rule": "101_point_interpolated_precision_recall",
            "matching_rule": MATCHING_RULE,
            "tie_break_rule": "image_id_then_prediction_index; label_index_for_equal_iou",
            "prediction_confidence_floor": confidence_floor,
            "iou_thresholds": list(AP_IOU_THRESHOLDS),
            "map_50": sum(included_ap50) / len(included_ap50) if included_ap50 else None,
            "map_50_95": sum(included_ap_range) / len(included_ap_range) if included_ap_range else None,
            "included_class_ids": [
                category_id for category_id in sorted(class_map)
                if counts[category_id]["ground_truth"] > 0
            ],
        },
        "operating_point": {
            "confidence_threshold": confidence_floor,
            "iou_threshold": operating_iou_threshold,
            "matching_rule": MATCHING_RULE,
            "labeled_object_count": total_ground_truths,
            "prediction_count": total_predictions,
            "true_positive_count": total_true_positives,
            "false_positive_count": total_false_positives,
            "false_negative_count": total_false_negatives,
            "precision": _ratio(total_true_positives, total_predictions),
            "recall": _ratio(total_true_positives, total_ground_truths),
        },
        "class_ids": sorted(class_map),
        "per_class": per_class,
    }


def _read_rgb(path: Path) -> RGBImage:
    try:
        with Image.open(path) as opened:
            if opened.format not in {"JPEG", "PNG"}:
                raise EvaluationError(f"Image contents are not JPEG or PNG: {path}")
            image = opened.convert("RGB")
            image.load()
            return RGBImage(image.width, image.height, image.tobytes())
    except (OSError, UnidentifiedImageError) as error:
        raise EvaluationError(f"Image cannot be decoded: {path}: {error}") from error


def _select_validation(
    dataset: DatasetInput,
    assignments: tuple[SplitAssignment, ...],
    frozen_lock: dict,
) -> tuple[tuple[AcceptedImage, SplitAssignment], ...]:
    assignment_by_id: dict[str, SplitAssignment] = {}
    for assignment in assignments:
        if assignment.image_id in assignment_by_id:
            raise EvaluationError(f"Split manifest repeats image_id {assignment.image_id}.")
        assignment_by_id[assignment.image_id] = assignment

    images_by_id = {image.image_id: image for image in dataset.accepted}
    missing = sorted(set(images_by_id) - set(assignment_by_id))
    unknown = sorted(set(assignment_by_id) - set(images_by_id))
    if missing or unknown:
        raise EvaluationError(
            f"Split manifest must assign every manifest ID exactly once; missing={missing}, unknown={unknown}."
        )

    selected = tuple(
        (images_by_id[image_id], assignment)
        for image_id, assignment in sorted(assignment_by_id.items())
        if assignment.split == "valid"
    )
    if not selected:
        raise EvaluationError("Split manifest selects no validation images.")

    test_ids = {item.image_id for item in assignments if item.split == "test"}
    test_groups = {item.source_group_id for item in assignments if item.split == "test"}
    frozen_ids = {
        item["image_id"] for item in frozen_lock["files"] if isinstance(item.get("image_id"), str)
    }
    frozen_groups = set(frozen_lock["groups"])
    for image, assignment in selected:
        if image.image_id in frozen_ids or image.image_id in test_ids:
            raise EvaluationError(
                f"Frozen test membership conflict: validation image_id {image.image_id}."
            )
        if assignment.source_group_id in frozen_groups or assignment.source_group_id in test_groups:
            raise EvaluationError(
                f"Frozen test membership conflict: validation group_id {assignment.source_group_id} "
                f"for image_id {image.image_id}."
            )
        path = resolve_dataset_path(dataset.root, image.image_path)
        if not path.is_file():
            raise EvaluationError(f"Validation image is missing for image_id {image.image_id}: {path}")

    splits_by_group: dict[str, set[str]] = {}
    for assignment in assignments:
        splits_by_group.setdefault(assignment.source_group_id, set()).add(assignment.split)
    for group_id, splits in sorted(splits_by_group.items()):
        if len(splits) > 1:
            raise EvaluationError(
                f"Source group {group_id} appears in multiple splits: {', '.join(sorted(splits))}."
            )

    try:
        assert_excluded(
            frozen_lock,
            paths=[dataset.resolve(image.image_path) for image, _ in selected],
            group_ids=[assignment.source_group_id for _, assignment in selected],
        )
    except (FrozenSetError, FrozenSetLeakError) as error:
        raise EvaluationError(f"Frozen test membership check failed: {error}") from error
    return selected


def _labels_for(image: AcceptedImage, dataset: DatasetInput, class_map: dict[int, str]) -> list[dict]:
    labels = []
    for obj in sorted(objects_for_image(dataset, image), key=lambda item: item.annotation_id):
        if obj.category_id not in class_map:
            raise EvaluationError(
                f"Label annotation {obj.annotation_id} for image {image.image_id} "
                f"uses unknown category_id {obj.category_id}."
            )
        labels.append({
            "annotation_id": obj.annotation_id,
            "category_id": obj.category_id,
            "bbox_xyxy": list(_label_box(obj, image)),
        })
    return labels


def evaluate_validation(
    dataset_root,
    manifest_path,
    split_manifest_path,
    frozen_test_lock_path,
    checkpoint_path,
    class_map_path,
    output_path,
    *,
    variant: str,
    device: str = "cpu",
    confidence_floor: float,
    iou_threshold: float,
    detector_loader: Callable | None = None,
    image_reader: Callable[[Path], RGBImage] | None = None,
    benchmark_output=None,
    warmup_runs: int | None = None,
    report_output=None,
) -> dict:
    """Run validation, then atomically write and return its complete report."""
    floor = confidence_value(confidence_floor)
    if report_output is not None and benchmark_output is None:
        raise EvaluationError("report_output requires benchmark_output")
    if warmup_runs is not None and benchmark_output is None:
        raise EvaluationError("warmup_runs requires benchmark_output")
    if warmup_runs is not None and (type(warmup_runs) is not int or warmup_runs < 0):
        raise EvaluationError("warmup_runs must be a nonnegative integer")
    if isinstance(iou_threshold, bool) or not isinstance(iou_threshold, (int, float)):
        raise EvaluationError("IoU threshold must be a finite number between 0 and 1.")
    iou_threshold = float(iou_threshold)
    if not math.isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
        raise EvaluationError("IoU threshold must be a finite number between 0 and 1.")

    try:
        dataset, manifest, assignments = load_file_inputs(
            dataset_root, manifest_path, split_manifest_path
        )
    except (OSError, RfdetrExportError, KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"Validation manifest or label failure: {error}") from error
    try:
        frozen_lock = load_test_lock(frozen_test_lock_path)
    except (OSError, FrozenSetError) as error:
        raise EvaluationError(f"Frozen test lock failure: {error}") from error

    selected = _select_validation(dataset, assignments, frozen_lock)
    try:
        class_map = load_class_map(class_map_path)
        validate_class_map(class_map)
        for parsed in dataset.annotations:
            for class_spec in parsed.classes:
                if class_map.get(class_spec.category_id) != class_spec.name:
                    raise EvaluationError(
                        f"Class map does not match labels for category_id {class_spec.category_id}."
                    )
    except (OSError, RfdetrExportError, ValueError, TypeError) as error:
        raise EvaluationError(f"Class map or label failure: {error}") from error

    prepared = []
    for image, assignment in selected:
        image_path = resolve_dataset_path(dataset.root, image.image_path)
        if not image_path.is_file():
            raise EvaluationError(f"Validation image is missing for image_id {image.image_id}: {image_path}")
        prepared.append((image, assignment, image_path, _labels_for(image, dataset, class_map)))

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if benchmark_output is not None:
        targets = [Path(p).expanduser().resolve() for p in (output_path, benchmark_output)]
        if report_output is not None:
            targets.append(Path(report_output).expanduser().resolve())
        protected = {Path(p).resolve() for p in
                     (manifest_path, split_manifest_path, frozen_test_lock_path,
                      checkpoint_path, class_map_path)}
        protected.update(dataset.resolve(image.image_path) for image in dataset.accepted)
        protected.update(dataset.resolve(image.annotation_path) for image in dataset.accepted)
        if len(set(targets)) != len(targets) or any(p in protected for p in targets):
            raise EvaluationError("Benchmark outputs must be distinct from each other and inputs")
        for target in targets:
            if target.exists():
                raise EvaluationError(f"Benchmark-enabled runs require new output files: {target}")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                raise EvaluationError(f"Invalid output directory: {target}: {error}") from error
    if not checkpoint.is_file():
        raise EvaluationError(f"Checkpoint file is missing: {checkpoint}")
    try:
        detector = (detector_loader or load_detector)(
            checkpoint, class_map, variant=variant, device=device
        )
        checkpoint_digest = getattr(getattr(detector, "checkpoint", None), "sha256", None)
        if not isinstance(checkpoint_digest, str) or not checkpoint_digest:
            checkpoint_digest = sha256_file(checkpoint)
    except (CheckpointError, OSError, ImportError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationError(f"Checkpoint load failure: {error}") from error

    read_image = image_reader or _read_rgb
    records = []
    for image, assignment, image_path, labels in prepared:
        try:
            rgb_image = read_image(image_path)
            if (rgb_image.width, rgb_image.height) != (image.width, image.height):
                raise EvaluationError(
                    f"Image dimensions do not match validation labels for image_id {image.image_id}."
                )
            result = detector.predict(rgb_image, confidence_floor=floor)
            validate_result(
                result,
                image_width=image.width,
                image_height=image.height,
                confidence_floor=floor,
                classes=class_map,
            )
        except (InferenceError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise EvaluationError(f"Evaluation failed for image_id {image.image_id}: {error}") from error

        predictions = tuple(sorted(result.detections, key=lambda item: -item.confidence))
        matched = _match(predictions, labels, iou_threshold)
        records.append({
            "image_id": image.image_id,
            "image_path": image.image_path.as_posix(),
            "source_group_id": assignment.source_group_id,
            "width": image.width,
            "height": image.height,
            "predictions": [
                {"category_id": item.category_id, "confidence": item.confidence,
                 "bbox_xyxy": list(item.bbox_xyxy)}
                for item in predictions
            ],
            "labels": labels,
            **matched,
        })
        del rgb_image

    metrics = _calculate_metrics(records, class_map, floor, iou_threshold)
    operating_point = metrics["operating_point"]
    report = {
        "schema_version": 1,
        "evaluation_id": new_run_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_digest,
            "variant": variant,
            "device": device,
            "model_label": getattr(detector, "model_label", None),
        },
        "dataset": {
            "version": manifest.dataset_version,
            "content_hash": manifest.content_hash,
            "frozen_test_version": frozen_lock["version"],
            "frozen_test_lock_hash": frozen_lock["lock_hash"],
            "validation_image_ids": [image.image_id for image, _ in selected],
        },
        "settings": {
            "split": "valid",
            "confidence_floor": floor,
            "matching_rule": MATCHING_RULE,
            "iou_threshold": iou_threshold,
            "average_precision_rule": metrics["average_precision"]["rule"],
            "average_precision_iou_thresholds": metrics["average_precision"]["iou_thresholds"],
            "matching_tie_break_rule": metrics["average_precision"]["tie_break_rule"],
        },
        "classes": [
            {"class_id": category_id, "class_name": name}
            for category_id, name in sorted(class_map.items())
        ],
        "metrics": metrics,
        "summary": {
            "images": len(records),
            "ground_truths": operating_point["labeled_object_count"],
            "predictions": operating_point["prediction_count"],
            "true_positives": operating_point["true_positive_count"],
            "false_positives": operating_point["false_positive_count"],
            "false_negatives": operating_point["false_negative_count"],
            "precision": operating_point["precision"],
            "recall": operating_point["recall"],
        },
        "records": records,
    }
    try:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(target, report, operation="write evaluation report",
                          overwrite=benchmark_output is None)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationError(f"Cannot write complete evaluation report: {error}") from error
    if benchmark_output is not None:
        from app.benchmark import benchmark_validation

        try:
            _select_validation(dataset, assignments, frozen_lock)
            measurements = benchmark_validation(
                detector, ((image, path) for image, _, path, _ in prepared),
                image_reader=read_image, class_map=class_map, confidence_floor=floor,
                device=device, warmup_runs=5 if warmup_runs is None else warmup_runs,
            )
            benchmark = {
                "schema_version": 1, "evaluation_id": report["evaluation_id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_created_at": report["created_at"],
                "evaluation_sha256": sha256_file(target),
                "checkpoint": report["checkpoint"], "dataset": report["dataset"],
                "settings": report["settings"], "benchmark": measurements,
            }
            write_json_atomic(targets[1], benchmark, operation="write benchmark", overwrite=False)
        except (OSError, RuntimeError, ImportError, AttributeError, TypeError, ValueError) as error:
            raise EvaluationError(
                f"Benchmark failed; valid evaluation retained at {target}: {error}"
            ) from error
    if report_output is not None:
        from app.evaluation_report import ReportError, generate_report
        try:
            generate_report(target, targets[1], targets[2])
        except ReportError as error:
            raise EvaluationError(f"Reporting failed; evaluation and benchmark JSON retained: {error}") from error
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("split_manifest", type=Path)
    parser.add_argument("frozen_test_lock", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("class_map", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", required=True, choices=("nano", "small", "medium", "base", "large"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--confidence-floor", required=True, type=float)
    parser.add_argument("--iou-threshold", required=True, type=float)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--warmup-runs", type=int)
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = evaluate_validation(
            args.dataset_root, args.manifest, args.split_manifest, args.frozen_test_lock,
            args.checkpoint, args.class_map, args.output, variant=args.variant, device=args.device,
            confidence_floor=args.confidence_floor, iou_threshold=args.iou_threshold,
            benchmark_output=args.benchmark_output, warmup_runs=args.warmup_runs,
            report_output=args.report_output,
        )
    except EvaluationError as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    print(json.dumps({"evaluation_id": report["evaluation_id"], "summary": report["summary"],
                      "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
